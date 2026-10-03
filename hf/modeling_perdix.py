"""Perdix: Differential Attention + PolyNorm 기반 decoder-only LM (transformers 연동용).

학습 코드(model.py)의 PerdixSLM과 모듈 이름이 같아 state_dict가 그대로 호환된다.
KV 캐시는 구현하지 않았다(생성 시 매 스텝 전체 시퀀스를 다시 계산).
패딩 마스크도 없다: attention_mask는 무시되며 causal 마스크만 적용된다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_perdix import PerdixConfig


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

    def __init__(self, cfg, layer_idx: int):
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

    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.act = PolyNorm()
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = DifferentialAttention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PerdixForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = PerdixConfig
    base_model_prefix = ""
    # transformers 5.x는 {묶이는 키: 원본 키} dict, 4.x는 키 목록을 기대한다. dict는 양쪽 다 동작.
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}
    _supports_cache_class = False

    def __init__(self, config: PerdixConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [Block(config, i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPE 표는 buffer로 두지 않고 첫 forward에서 만든다. transformers 5.x는 모델을 meta 장치에서
        # 만든 뒤 가중치만 채우므로, 저장되지 않는(non-persistent) buffer는 내용이 날아간다.
        self._rope = None
        self.post_init()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.tok_emb(input_ids)
        if self._rope is None or self._rope[0].device != x.device:
            head_dim = self.config.dim // self.config.n_heads // 2
            cos, sin = precompute_rope(head_dim, self.config.max_seq_len, self.config.rope_theta)
            self._rope = (cos.to(x.device), sin.to(x.device))
        for blk in self.blocks:
            x = blk(x, *self._rope)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 캐시가 없으므로 매 스텝 최근 max_seq_len 토큰 전체를 다시 넣는다
        return {"input_ids": input_ids[:, -self.config.max_seq_len:]}
