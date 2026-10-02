"""Motif 아키텍처 기반 SLM (~350M).

Motif-2.6B 기술보고서(arXiv:2508.09148)의 두 핵심 컴포넌트를 구현:
  - Differential Attention: [softmax(Q1K1^T) - lambda * softmax(Q2K2^T)] V
    (Ye et al. 2024, Differential Transformer 방식의 lambda 재파라미터화)
  - PolyNorm: x, x^2, x^3 각각 RMS 정규화 후 학습 가중치로 합성 (최대 3차)

구조: decoder-only, pre-RMSNorm, RoPE, tied embedding.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PerdixConfig:
    vocab_size: int = 49152
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16          # differential attention은 내부에서 head_dim을 반으로 나눠 2쌍 사용
    ffn_dim: int = 4096
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class PolyNorm(nn.Module):
    """poly_norm(x) = w1*rms(x) + w2*rms(x^2) + w3*rms(x^3) + b"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _rms(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def forward(self, x):
        xf = x.float()
        out = (self.weight[0] * self._rms(xf)
               + self.weight[1] * self._rms(xf ** 2)
               + self.weight[2] * self._rms(xf ** 3)
               + self.bias)
        return out.to(x.dtype)


def precompute_rope(head_dim, max_seq_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D) -> 짝/홀 성분 회전
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class DifferentialAttention(nn.Module):
    """두 개의 어텐션 맵 차이로 노이즈를 상쇄하는 어텐션.

    표준 어텐션과 동일한 파라미터 수: head_dim을 반으로 나눠
    (Q1,K1), (Q2,K2) 두 쌍을 만들고 softmax 맵을 lambda 가중으로 뺀다.
    각 항이 표준 softmax(QK^T)V 꼴이므로 SDPA(flash attention) 2회로 계산.
    """

    def __init__(self, cfg: PerdixConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads // 2  # 반으로 쪼개 2쌍
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

        # lambda 재파라미터화 (Differential Transformer eq.2)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        d = self.head_dim
        self.lambda_q1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(d) * 0.1)

        # 헤드별 RMSNorm (논문의 GroupNorm 역할)
        self.subln = RMSNorm(2 * self.head_dim, eps=cfg.norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # (B, T, C) -> (B, 2H, T, D): 헤드당 (Q1,Q2), (K1,K2) 쌍
        q = self.wq(x).view(B, T, 2 * H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, 2 * H, D).transpose(1, 2)
        # V(head_dim 2D)를 D짜리 두 헤드로 펼침 — Q/K/V head_dim을 맞춰야
        # flash attention 커널 자격이 되고(math 폴백 방지), 결과는 수학적으로 동일
        v = self.wv(x).view(B, T, 2 * H, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q1, q2 = q[:, 0::2], q[:, 1::2]   # 각 (B, H, T, D)
        k1, k2 = k[:, 0::2], k[:, 1::2]

        # 같은 어텐션 맵을 V의 두 반쪽에 적용하도록 헤드 축으로 복제
        q1r = q1.repeat_interleave(2, dim=1)   # (B, 2H, T, D)
        k1r = k1.repeat_interleave(2, dim=1)
        q2r = q2.repeat_interleave(2, dim=1)
        k2r = k2.repeat_interleave(2, dim=1)

        a1 = F.scaled_dot_product_attention(q1r, k1r, v, is_causal=True)
        a2 = F.scaled_dot_product_attention(q2r, k2r, v, is_causal=True)
        # (B, 2H, T, D) -> (B, H, T, 2D) 복원
        a1 = a1.view(B, H, 2, T, D).permute(0, 1, 3, 2, 4).reshape(B, H, T, 2 * D)
        a2 = a2.view(B, H, 2, T, D).permute(0, 1, 3, 2, 4).reshape(B, H, T, 2 * D)

        lam1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum().float())
        lam2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum().float())
        lam = (lam1 - lam2 + self.lambda_init).to(x.dtype)

        attn = a1 - lam * a2                     # (B, H, T, 2D)
        attn = self.subln(attn) * (1.0 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        return self.wo(attn)


class FeedForward(nn.Module):
    """Linear -> PolyNorm -> Linear (Motif 그림 1의 비게이트 FFN)"""

    def __init__(self, cfg: PerdixConfig):
        super().__init__()
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.act = PolyNorm()
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: PerdixConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = DifferentialAttention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PerdixSLM(nn.Module):
    def __init__(self, cfg: PerdixConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(
            [Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        # tied embedding: lm_head 가중치 = tok_emb 가중치
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.dim // cfg.n_heads // 2
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 잔차 누적 보정: 깊이에 따른 출력 프로젝션 스케일다운 (GPT-2 방식)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-100)
        return logits, loss

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n


if __name__ == "__main__":
    cfg = PerdixConfig()
    model = PerdixSLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {total/1e6:.1f}M")
    print(f"비임베딩 파라미터: {model.num_params()/1e6:.1f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 128))
    y = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = model(x, y)
    print(f"forward ok: logits {tuple(logits.shape)}, loss {loss.item():.3f}")
    print(f"기대 초기 loss ~ ln(vocab) = {math.log(cfg.vocab_size):.3f}")
