import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

#                                     MiniMind Config
class MyGptConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 这里是标记为可学习参数的意思
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1).unsqueeze(-1) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

class GroupQueryAttention(nn.Module):
    def __init__(self, config: MyGptConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.query_head_num = config.num_attention_heads
        self.key_value_head_num = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.query_head_num // self.key_value_head_num
        self.drop_out = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.query_head_num * self.head_dim, bias = False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_value_head_num * self.head_dim, bias = False)
        self.v_proj = nn.Linear(self.hidden_size, self.key_value_head_num * self.head_dim, bias = False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias = False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.is_causal = True

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def repeat(self, x):
        # input batch, seq, head_num, head_dim
        # 先转成 batch seq, head_num, rep_num, head_dim
        # 然后再做reshape
        batch, seq, head_num, head_dim = x.size()
        if (self.n_rep == 1):
            return x
        return x.unsqueeze(3).expand(batch, seq, head_num, self.n_rep, head_dim).reshape(batch, seq, -1, head_dim)

    def forward(self, x, position_embeddings, past_key_value = None, use_cache: bool=False, attention_mask = None):
        batch, seq, _ = x.size()
        xq = self.q_proj(x).view(batch, seq, self.query_head_num, self.head_dim)
        xk = self.k_proj(x).view(batch, seq, self.key_value_head_num, self.head_dim)
        xv = self.v_proj(x).view(batch, seq, self.key_value_head_num, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # (2, batch, seq, heddiendim)
        if past_key_value is not None:
            # batch, post_seq + seq, kv_hiddendim
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # batch, head_num, seq, head_dim
        xq = xq.transpose(1,2)
        # batch, head_num, post + seq, head_dim
        xk = self.repeat(xk).transpose(1, 2)
        xv = self.repeat(xv).transpose(1, 2)
        if self.flash and (seq > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.drop_out if self.training else 0.0, is_causal=self.is_causal)
        else:
            # batch, head_num, seq, past_len + seq
            attention_weight = (xq @ xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            kv_seq_len = attention_weight.shape[-1]
            if self.is_causal:
                # 因果 mask 只作用于右侧新 token 的 key 列，左侧 past 列是历史 key，全部可见
                causal_mask = torch.zeros(seq, kv_seq_len, device=attention_weight.device, dtype=attention_weight.dtype)
                causal_mask[:, kv_seq_len - seq:] = torch.full((seq, seq), float("-inf"), device=attention_weight.device, dtype=attention_weight.dtype).triu(1)
                attention_weight = attention_weight + causal_mask.unsqueeze(0).unsqueeze(0)
            # 这里的 attention mask 维度是 batch * seq，是上层应用传入表示哪些有效，所以我们需要把 score 中无效的 token 设置为 -inf
            if (attention_mask is not None):
                # batch, seq -> batch, 1, 1, seq，再利用 broadcast 对齐到含 past 的 key 维度
                pad_mask = (1 - attention_mask).unsqueeze(1).unsqueeze(1)
                full_mask = torch.zeros(batch, 1, 1, kv_seq_len, device=attention_weight.device, dtype=attention_weight.dtype)
                full_mask[..., kv_seq_len - seq:] = pad_mask
                attention_weight = attention_weight + full_mask * float('-inf')
            # batch, head_num, seq, head_dim
            output = (self.attn_dropout(attention_weight) @ xv).transpose(1, 2).contiguous().view(batch, seq, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MyGptConfig, intermediate_size = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MoeFeedForward(nn.Module):
    def __init__(self, config: MyGptConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size = config.moe_intermediate_size) for _ in range(config.num_experts)])

    def forward(self, x):
        # batch, seq, expert_num
        batch, seq, _ = x.size()
        x_flat = x.view(batch * seq, -1)
        # batch * seq, expert_num
        expert_logits = torch.softmax(self.gate(x_flat), dim=-1)
        # batch * seq, top_k
        topk_wegiht, topk_indices = torch.topk(expert_logits, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        if (self.config.norm_topk_prob):
            topk_wegiht = topk_wegiht / (torch.sum(topk_wegiht, dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)

        for (expert_idx, expert) in enumerate(self.experts):
            seq_mask = topk_indices == expert_idx
            if torch.any(seq_mask):
                # 这里的 any 调用会降低一个维度，nonzero会升高一个维度，flatten会重新拍平
                selected_seq = torch.any(seq_mask, dim = -1).nonzero().flatten()
                # 这里的布尔索引会选中所有为 True 的元素，结果压平为一维
                # selected_num, 1
                selected_weight = topk_wegiht[seq_mask].unsqueeze(1)
                y.index_add_(0, selected_seq, (expert(x_flat[selected_seq]) * selected_weight).to(y.dtype))
            elif self.training:
                # 这个地方时防止空载报错
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        
        if self.training and self.config.router_aux_loss_coef > 0:
            # one_hot 以后是 (seq * batch, topk, num_experts)
            # mean(0)延0维度，消除掉0维，从而变成(topk, num_experts)
            load = F.one_hot(topk_indices, self.config.num_experts).float().mean(0)
            # expert_logits 维度是 (batch*seq, expert_num) mean 之后 变成 (expert_num)
            # 广播规则会默认在高纬度补充1
            # 乘法完成之后变成了 topk, expert_num
            # sum后变成标量
            self.aux_loss = (load * expert_logits.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            # 
            self.aux_loss = expert_logits.new_zeros(1).squeeze()

        return y.view(batch, seq, -1)

class LLMBlock(nn.Module):
    def __init__(self, config: MyGptConfig):
        super().__init__()
        self.pre_attention_layernorm = RMSNorm(config.hidden_size)
        self.pre_forward_layernorm = RMSNorm(config.hidden_size)
        
        self.attn = GroupQueryAttention(config)
        self.feedForward = MoeFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, x, position_embeddings, past_key_value = None, use_cache: bool=False, attention_mask = None):
        residual = x
        x, presnt_key_value = self.attn(
            self.pre_attention_layernorm(x),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        x += residual

        x += self.feedForward(self.pre_forward_layernorm(x))

        return x, presnt_key_value

class MyGpt(nn.Module):
    def __init__(self, config: MyGptConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.layer_num = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.drop_out = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size)
        self.block_layers = nn.ModuleList([LLMBlock(config) for _ in range(self.layer_num)])
        
        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x, past_key_values = None, use_cache: bool=False, attention_mask = None):
        # 这个地方用来判断是否是huging face的past_key_value对象
        batch, seq = x.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.block_layers)
        # past_key_values 的维度是 layer, k/v, batch, seq, himddendim
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.drop_out(self.embedding(x))
        # freqs_cos/sin 是注册的 buffer，会随模型 .to(device) 自动迁移，直接切片即可
        position_embedding = (self.freqs_cos[start_pos : start_pos + seq], self.freqs_sin[start_pos : start_pos + seq])

        present_key_vlaues = []
        for block, past_key_value in zip(self.block_layers, past_key_values):
            hidden_states, present_key_value = block(
                hidden_states, position_embedding, past_key_value, use_cache, attention_mask
            )
            
            present_key_vlaues.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.feedForward.aux_loss for l in self.block_layers if isinstance(l.feedForward, MoeFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, present_key_vlaues, aux_loss

class MyGptForCausalLLM(PreTrainedModel, GenerationMixin):
    config_class = MyGptConfig
    _tied_weights_keys = {"lm_head.weight" : "model.embedding.weight"}
    def __init__(self, config: MyGptConfig=None):
        self.config = config or MyGptConfig()
        super().__init__(self.config)
        self.model = MyGpt(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        if self.tie_word_embeddings:
            self.model.embedding.weight = self.lm_head.weight
        self.post_init()
        
    def forward(self, x, past_key_values = None, use_cache: bool=False, attention_mask = None, logits_to_keep = 0, labels = None):
        hidden_states, past_key_values, aux_loss = self.model(x, past_key_values, use_cache, attention_mask)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # batch, seq, vocab_size
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss = loss, aux_loss = aux_loss, logits = logits, past_key_values = past_key_values, hidden_states = hidden_states)

    @torch.inference_mode()
    def generate(self, inputs = None, attention_mask = None, max_new_tokens = 8192, eos_token_id = 2, temperature = 0.85, top_p=0.85, top_k=50, num_return_seqs = 1, use_cache: bool=True, do_sample=True, repetition_penalty = 1.0, **kwargs):
        # 第0维重复num_return_seqs次，第一维重复1次
        input_ids = kwargs.pop("input_idx", inputs)
        if input_ids is None:
            raise ValueError("generate() 需要传入 inputs（或 kwargs 中的 input_idx）")
        input_ids = input_ids.repeat(num_return_seqs, 1)
        attention_mask = attention_mask.repeat(num_return_seqs, 1) if attention_mask is not None else attention_mask
        past_key_values = kwargs.get("past_key_values", None)
        streamer = kwargs.pop("streamer", None)
        finished = torch.zeros(input_ids.shape[0], dtype = torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            # 这里在第0步可以处理，如果初始输入没有kvcache的话，第一轮可以计算出来
            # 如果初始有的话 past_key_values 不为空，则可以直接跳过去第一轮kv计算，直接走到输出流程，逐个计算下一个token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 除了第一步外，每一步之处理一个token，因为第一步的past_len是0
            # 这里的言外之意是只计算kvchache中不包含的输入
            outputs = self.forward(
                input_ids[:, past_len:],
                past_key_values,
                use_cache,
                attention_mask[:, past_len:] if attention_mask is not None else None,
            )
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # batch, vocab_size
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)

            if top_k > 0:
                # 只保留 top_k 的 logit，其余置为 -inf
                logits[logits < torch.topk(logits, top_k)[0][..., -1].unsqueeze(-1)] = float("-inf")

            if top_p > 0:
                # batch, vocab_size
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim = -1), dim = -1) > top_p
                # 向右平移一位：mask[..., 1:] = 原 mask[..., :-1]，且首位置 0，
                # 保证恰好覆盖累积概率 <= top_p 的那部分 token
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 把平移后的 mask 按原 token 位置放回去（scatter），再置 -inf
                logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(1), next_token.new_full((input_ids.shape[0], 1), eos_token_id), next_token)

            input_ids = torch.cat([input_ids, next_token], dim = -1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()

        if kwargs.get("return_kv"):
            return {"generated_ids" : input_ids, "past_kv" : past_key_values}

        return input_ids
