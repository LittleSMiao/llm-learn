import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 512   # max seq len
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    head_size: int = n_embd // n_head
    drop_out: flaot = 0.1
    vocab_size: int = 50257
    pass

class MultiHeadAttention:
    def __init__(self, config):
        self.Q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.K_proj = nn.Linear(config.n_embd, config.n_embd)
        self.V_proj = nn.Linear(config.n_embd, config.n_embd)
        self.head_size = config.head_size

        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))

        self.dropout = nn.Dropout(config.drop_out)

    def forward(self, X):
        # (batch, seq, hiddendim)
        batch, seq, _ = X.size()
        Q = self.Q_proj(X)
        K = self.K_proj(X)
        V = self.V_proj(X)

        Q = Q.view(batch, seq, self.n_head, self.head_size)
        K = K.view(batch, seq, self.n_head, self.head_size)
        V = V.view(batch, seq, self.n_head, self.head_size)

        # batch, head, seq, head_size
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_weight = torch.matmul(
            Q, K.transpose(2, 3)
        )

        attention_weight = attention_weight.masked_fill(
            self.attention_mask[:seq, :seq] == 0,
            float('-inf')
        ) / math.sqrt(self.head_size)

        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )

        attention_weight = self.dropout(attention_weight)

        return attention_weight @ V

class Feedforward:
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd

        self.net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd * 4),
            nn.GELU(),
            nn.Linear(self.n_embd * 4, self.n_embd),
            nn.Dropout(config.drop_out)
        )

    def forward(self, X):
        return self.net(X)

class Block:
    def __init__(self, config):
        super().__init__()

        self.att = MultiHeadAttention(config)
        self.ffn = Feedforward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, X):
        super().__init__()

        X += self.att(self.ln1(X))
        X += self.ffn(self.ln2(X))

        return X

class GPT:
    def __init__(self, config):
        super().__init__()

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(self.n_layer)]
        )

        self.ln_final = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq = idx.size()
        # b, s, h
        token_embedding = self.token_embedding_table(X)
        # s, h
        position_embedding = self.position_embedding_table(
            torch.arange(seq, device=idx.device)
        )

        x = token_embedding + position_embedding # b, s, h

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # b, s, vocab_size

        if targets is None:
            loss = None:
        else:
            batch, seq, vocab_size = logits.size()
            logits = logits.view(batch*seq, -1)
            targets = targets.view(batch * seq)
            loss = F.cross_entropy(logits, targets)

        return logits, loss



