"""Small decoder-only Transformer: RoPE, RMSNorm, SwiGLU, tied embeddings."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {
    "tiny": {"dim": 128, "layers": 8, "heads": 4, "hidden_dim": 320},
    "small": {"dim": 256, "layers": 9, "heads": 8, "hidden_dim": 704},
    "base": {"dim": 384, "layers": 10, "heads": 8, "hidden_dim": 1024},
}
MAX_PARAMETERS = 20_000_000


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 128
    layers: int = 8
    heads: int = 4
    hidden_dim: int = 320
    context_length: int = 128

    def __post_init__(self):
        if not 258 <= self.vocab_size <= 16384:
            raise ValueError("词表大小必须在 258 到 16384 之间。")
        if not 32 <= self.dim <= 512 or not 1 <= self.layers <= 16:
            raise ValueError("隐藏维度需为 32–512，层数需为 1–16。")
        if self.heads < 1 or self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("隐藏维度必须能整除注意力头数，且每个头的维度必须为偶数。")
        if not 32 <= self.hidden_dim <= 2048 or not 16 <= self.context_length <= 1024:
            raise ValueError("前馈维度需为 32–2048，上下文长度需为 16–1024。")
        if self.parameters > MAX_PARAMETERS:
            raise ValueError(f"配置共 {self.parameters:,} 参数，超过 20M；请缩小词表、层数或维度。")

    @property
    def parameters(self):
        return self.vocab_size * self.dim + self.layers * (
            4 * self.dim**2 + 3 * self.dim * self.hidden_dim + 2 * self.dim
        ) + self.dim

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


def rotary(x):
    length, dim = x.shape[-2:]
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angle = torch.outer(torch.arange(length, device=x.device).float(), inv)
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.attn_norm = RMSNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2) for a in (q, k, v)]
        a = F.scaled_dot_product_attention(rotary(q), rotary(k), v, is_causal=True, dropout_p=0.0)
        x = x + self.out(a.transpose(1, 2).contiguous().view(batch, length, dim))
        y = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
        self.apply(self._init)
        # Reduce residual-branch variance as depth increases.
        for block in self.blocks:
            nn.init.normal_(block.out.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.config.context_length:
            raise ValueError("输入超过模型上下文长度。")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
