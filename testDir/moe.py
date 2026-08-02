import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__()
        self.proj = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        return self.proj(x)

class SimpleMoe(nn.Module):
    def __init__(self, feature_in: int, feature_out: int, expert_num: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_num)
            ]
        )
        self.gate = nn.Linear(feature_in, expert_num)

    def forward(self, x):
        # batch, expertnum
        expert_weights = self.gate(x)

        expert_outs = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]

        # batch, expert_num, feature_out
        expert_outs = torch.concat(expert_outs, dim=1)

        # batch, 1, expert_num
        expert_weights = expert_weights.unsqueeze(1)

        # (b, 1, exprt_num) @ (b, expert_num, feature_out) = ()b, 1, feature_out
        x = expert_weights @ expert_outs

        return x.squeeze()

class SparseMoeRouter(nn.Module):
    def __init__(self, hiddendim, expert_num, top_k):
        self.router = nn.Linear(hiddendim, expert_num)
        self.top_k = top_k
        self.expert_num = expert_num
        self.hiddendim = hiddendim

    def forward(self, x):

        # batch, expert_num
        logits = self.router(x)

        prop = torch.softmax(logits, dim = -1, dtype=torch.float)

        # (b * s, top_k)
        router_weights, selected_expert = torch.topk(prop, self.expert_num, dim = -1)
        
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim = True)
        router_weights = router_weights.to(x.dtype)

        # (b * s, top_k, expert_num)
        expert_mask = F.one_hot(selected_expert, numclasses = self.expert_num)
        expert_mask = expert_mask.permute(2, 1, 0) # (expert_num, top_k, b*s)

        return logits, router_weights, selected_expert, expert_mask

class MOEConfig:
    def __init__(
            self, 
            hidden_dim, 
            expert_number, 
            top_k, 
            shared_experts_number=2,
        ):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMoe(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.expert_num = config.expert_number
        self.hidden_dim = config.hidden_dim
        self.top_k = config.top_k

        self.router = SparseMoeRouter(self.hidden_dim, self.expert_num, self.top_k)

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_num)
            ]
        )

    def forward(self, x):
        batch, seq, _ = x.size()
        # b*s, h
        x = x.view(batch*seq, -1)

        # (expert_num, top_k, b*s)
        expert_logits, router_weights, selected_expert, expert_mask = self.router(x)

        final_hidden_states = torch.zeros(
            (batch * seq, self.hidden_dim),
            dtype=x.dtype,
            device=x.device
        )

        for expert_idx in range(self.expert_num):
            top_idx, token_idx = torch.where(expert_mask[expert_idx])

            selected_token = x.unsqueeze(0)[:, token_idx, :].reshape(-1, self.hidden_dim)

            # selected_tokens, hidden_dim
            seletced_outputs = self.experts[expert_idx](selected_token)

            seletced_outputs = seletced_outputs * router_weights[token_idx, top_idx].unsqueeze(-1)
            
            final_hidden_states.index_add_(0, top_idx, seletced_outputs.to(x.dtype))

        return final_hidden_states.reshape(batch, seq, self.hidden_dim), expert_logits


def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)

class SharedMoe(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        
        self.router_moe = SparseMOE(config)
        
        self.shared_experts = nn.ModuleList(
            [ BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.shared_experts_number) ]
        )

    def forward(self, x):
        sparse_outs, router_logits = self.router_moe(x)

        shared_outs = [
            expert(x) for expert in self.shared_experts
        ]
        shared_outs = torch.stack(shared_outs, dim = 0).sum(dim=0)

        return sparse_outs + shared_outs, router_logits
