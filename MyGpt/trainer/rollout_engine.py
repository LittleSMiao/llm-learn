import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor

# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass

class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        # 生成和重算 logp 都必须在 eval 下做：train 模式下 dropout 是随机的，
        # 采样用一套 mask、重算 logp 用另一套，算出来的 logp 就不是采样时那个分布的 logp 了
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), ctx:
                # 注意 MyGpt.generate 的参数名是 inputs / num_return_seqs，且它没有 pad_token_id 参数，
                # 多传的关键字会被 **kwargs 静默吞掉。这里 batch 已经手动 repeat_interleave，
                # 采样结果的排布是 [p0 * N, p1 * N, ...]，正是后续按 prompt 分组算 advantage 需要的顺序
                output_ids = model.generate(
                    inputs=prompt_ids.repeat_interleave(num_generations, dim=0),   # 创建多个batch，方便后续分别进行采样
                    attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                ).clone()  # Batch * N, Prompt + Request
                prompt_len = prompt_ids.size(1)
                completion_ids = output_ids[:, prompt_len :] # B * N, R
                completion_len = completion_ids.size(1)

                # generate 结束后是用 eos 把后面的位置补齐的（不是 pad），所以按「第一个 eos」截断：
                # eos 本身参与 loss，它之后的位置不参与
                eos_id = self.tokenizer.eos_token_id
                if eos_id is None:
                    completion_mask = completion_ids.new_ones(completion_ids.shape)
                else:
                    pos = torch.arange(completion_len, device=completion_ids.device).unsqueeze(0)
                    # 整条都没出现 eos 的序列给 completion_len，等价于全部保留
                    first_eos = torch.where(completion_ids == eos_id, pos, completion_len).min(dim=1, keepdim=True).values
                    completion_mask = (pos <= first_eos).long()

                # attention_mask 必须由「prompt 的真实 mask + completion mask」拼出来。
                # 不能拿 (output_ids != pad_token_id) 反推：generate 不产生 pad，
                # 且很多 tokenizer 的 pad_token_id 就是 eos_token_id（也有的根本没有 pad_token_id）
                full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), completion_mask], dim=1)
                per_token_logps = self.get_per_token_logits(full_mask, output_ids, completion_len)
        finally:
            if was_training:
                model.train()
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                        prompt_ids.new_full((output_ids.size(0),), prompt_len),
                        completion_mask)


    def update_policy(self, policy_model: torch.nn.Module):
        self.policy_model = policy_model

    # must under no_grad ctx
    def get_per_token_logits(self, attention_mask, output_ids, completion_len):
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        input_ids = output_ids.detach().clone() if output_ids.is_inference() else output_ids

        # B * N, R, vocab_size
        completion_logits = model.forward(input_ids, attention_mask=attention_mask, logits_to_keep=completion_len + 1).logits[:, :-1, :]

        per_batch_token_logits = []

        for batch_logits, batch_token in zip(completion_logits, output_ids[:, -completion_len:]):
            # batch_logits: (Request, vocab_size)
            # batch_token: (Request)
            # batch_token_logit: (Request)
            batch_token_logits = torch.gather(batch_logits.log_softmax(dim=-1), dim=1, index=batch_token.unsqueeze(-1)).squeeze(-1)
            per_batch_token_logits.append(batch_token_logits)

        return torch.stack(per_batch_token_logits)
