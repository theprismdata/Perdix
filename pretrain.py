"""PerdixSLM 사전학습 스크립트.

- packed/<source>/*.bin (uint16 토큰)을 memmap으로 읽음
- Motif식 선형 데이터 믹싱 스케줄: 학습 진행률에 따라 소스 비율을
  initial -> final로 선형 보간
- WSD(Warmup-Stable-Decay) 학습률 스케줄
- bf16 autocast, grad accumulation, 체크포인트 재개 지원

실행:
    python3 pretrain.py                     # 기본 설정
    python3 pretrain.py --smoke             # 초소형 스모크 테스트
    python3 pretrain.py --resume ckpt/latest.pt
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from model import PerdixConfig, PerdixSLM

# ---------------- 학습 설정 ----------------
SEQ_LEN = 2048
TOTAL_TOKENS = 30_000_000_000     # 1B 모델 × 30B 토큰 (Chinchilla ~20N의 1.5배)
BATCH_TOKENS = 1_048_576          # 유효 배치 (1M 토큰)
MICRO_BS = 8                      # 1.1B: 활성화 메모리 증가로 bs 축소 (350M은 16이었음)
LR_PEAK = 3e-4                    # 1B급 관례 (350M은 5e-4)
WARMUP_TOKENS = 1_000_000_000     # 1B 토큰 워밍업 (예산 30B의 3.3%)
DECAY_START_FRAC = 0.8            # 마지막 20%에서 감쇠
MIN_LR_RATIO = 0.25               # Motif: 피크의 25%까지 감쇠
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CKPT_DIR = "ckpt"
CKPT_EVERY_STEPS = 50             # 1M 토큰/스텝 × 75초 기준 약 1시간마다
LOG_EVERY_STEPS = 5

# Motif식 선형 믹싱: 초반 영어 위주 -> 후반 한국어/수학 비중 증가
MIX_INITIAL = {"web_en": 0.65, "korean": 0.30, "math": 0.05}
MIX_FINAL   = {"web_en": 0.25, "korean": 0.50, "math": 0.25}
SOURCES = list(MIX_INITIAL)


class PackedSource:
    """한 소스의 .bin 셔드들을 memmap으로 열고 랜덤 윈도우를 샘플링."""

    def __init__(self, name, rng):
        self.name = name
        self.rng = rng
        paths = sorted(glob.glob(os.path.join("packed", name, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"no bins for source {name}")
        self.maps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        sizes = np.array([m.shape[0] for m in self.maps], dtype=np.float64)
        self.probs = sizes / sizes.sum()
        self.total = int(sizes.sum())
        self.consumed = 0

    def sample(self, seq_len):
        i = self.rng.choice(len(self.maps), p=self.probs)
        m = self.maps[i]
        start = self.rng.integers(0, m.shape[0] - seq_len - 1)
        self.consumed += seq_len
        return torch.from_numpy(m[start:start + seq_len + 1].astype(np.int64))


def mix_probs(progress):
    """진행률(0~1)에 따라 소스 비율을 선형 보간."""
    p = np.array([MIX_INITIAL[s] + (MIX_FINAL[s] - MIX_INITIAL[s]) * progress
                  for s in SOURCES])
    return p / p.sum()


def lr_at(tokens_done):
    if tokens_done < WARMUP_TOKENS:
        return LR_PEAK * tokens_done / WARMUP_TOKENS
    decay_start = TOTAL_TOKENS * DECAY_START_FRAC
    if tokens_done < decay_start:
        return LR_PEAK
    frac = (tokens_done - decay_start) / (TOTAL_TOKENS - decay_start)
    return LR_PEAK * (1.0 - (1.0 - MIN_LR_RATIO) * min(frac, 1.0))


def get_batch(sources, rng, progress, micro_bs, seq_len, device):
    probs = mix_probs(progress)
    xs, ys = [], []
    for _ in range(micro_bs):
        src = sources[SOURCES[rng.choice(len(SOURCES), p=probs)]]
        seq = src.sample(seq_len)
        xs.append(seq[:-1])
        ys.append(seq[1:])
    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.stack(ys).to(device, non_blocking=True)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto",
                    help="체크포인트 경로, 'auto'(기본: ckpt/latest.pt 있으면 재개), 'none'")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    global TOTAL_TOKENS, BATCH_TOKENS, WARMUP_TOKENS, CKPT_EVERY_STEPS
    # ~1.1B: dim 2048 / 20층 / 16헤드(diff attn head_dim 64) / FFN 8192
    cfg = PerdixConfig(dim=2048, n_layers=20, n_heads=16, ffn_dim=8192,
                      max_seq_len=SEQ_LEN)
    seq_len, micro_bs = SEQ_LEN, MICRO_BS
    if args.smoke:
        cfg = PerdixConfig(dim=256, n_layers=4, n_heads=4, ffn_dim=1024,
                          max_seq_len=512)
        seq_len, micro_bs = 512, 4
        TOTAL_TOKENS = 2_000_000
        BATCH_TOKENS = 16384
        WARMUP_TOKENS = 200_000
        CKPT_EVERY_STEPS = 50

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    print(f"device={device}, params 초기화 중...", flush=True)

    model = PerdixSLM(cfg).to(device)
    print(f"모델: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params",
          flush=True)
    if args.compile:
        model = torch.compile(model)

    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2]
    other_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
         {"params": other_params, "weight_decay": 0.0}],
        lr=LR_PEAK, betas=(0.9, 0.95), fused=(device == "cuda"))

    rng = np.random.default_rng(1234)
    sources = {s: PackedSource(s, rng) for s in SOURCES}
    for s in SOURCES:
        print(f"  {s}: {sources[s].total/1e9:.2f}B tokens", flush=True)

    grad_accum = max(1, BATCH_TOKENS // (micro_bs * seq_len))
    print(f"grad_accum={grad_accum} "
          f"(유효배치 {grad_accum * micro_bs * seq_len/1e6:.2f}M tokens)",
          flush=True)

    step, tokens_done = 0, 0
    if args.resume == "auto":
        auto_path = os.path.join(CKPT_DIR, "latest.pt")
        args.resume = auto_path if os.path.exists(auto_path) else None
    elif args.resume == "none":
        args.resume = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, tokens_done = ck["step"], ck["tokens_done"]
        rng.bit_generator.state = ck["rng_state"]
        print(f"재개: step {step}, {tokens_done/1e9:.2f}B tokens", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    model.train()
    t_log = time.time()
    tokens_at_log = tokens_done

    while tokens_done < TOTAL_TOKENS:
        progress = tokens_done / TOTAL_TOKENS
        lr = lr_at(tokens_done)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(sources, rng, progress, micro_bs, seq_len, device)
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=autocast_dtype,
                                enabled=(device != "cpu")):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        step += 1
        tokens_done += grad_accum * micro_bs * seq_len

        if step % LOG_EVERY_STEPS == 0:
            dt = time.time() - t_log
            tps = (tokens_done - tokens_at_log) / dt if dt > 0 else 0
            mix = ", ".join(f"{s}:{p:.2f}"
                            for s, p in zip(SOURCES, mix_probs(progress)))
            print(f"step {step} | {tokens_done/1e9:.3f}B tok "
                  f"| loss {loss_acc:.4f} | lr {lr:.2e} "
                  f"| {tps/1e3:.1f}K tok/s | mix [{mix}]", flush=True)
            t_log = time.time()
            tokens_at_log = tokens_done

        if step % CKPT_EVERY_STEPS == 0:
            path = os.path.join(CKPT_DIR, "latest.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "tokens_done": tokens_done,
                        "rng_state": rng.bit_generator.state,
                        "config": cfg.__dict__}, path + ".tmp")
            os.replace(path + ".tmp", path)
            consumed = {s: sources[s].consumed for s in SOURCES}
            with open(os.path.join(CKPT_DIR, "progress.json"), "w") as f:
                json.dump({"step": step, "tokens_done": tokens_done,
                           "consumed": consumed}, f)
            print(f"  ckpt 저장: step {step}", flush=True)

    print("=== 학습 완료 ===", flush=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
               os.path.join(CKPT_DIR, "final.pt"))


if __name__ == "__main__":
    main()
