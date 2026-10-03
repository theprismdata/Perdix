"""Perdix SFT(지시 미세조정) 스크립트.

- ckpt/final.pt(베이스)에서 시작, 임베딩을 대화용 토큰 2개만큼 늘림
- sft_packed/train.npz를 길이 구간(BUCKETS)별로 묶음. 배치 모양을 몇 가지로 고정해
  메모리 조각화를 막는다 (오른쪽 패딩, causal이라 영향 없음)
- 답변 구간만 loss 계산, warmup 후 cosine 감쇠
- ckpt/sft_latest.pt로 재개 지원, 끝나면 ckpt/sft_final.pt 저장

실행:
    python3 sft_train.py --compile       # 본 학습 (sft_latest.pt 있으면 재개)
    python3 sft_train.py --smoke         # 20스텝만 돌려 보기
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from model import PerdixConfig, PerdixSLM

# ---------------- 학습 설정 ----------------
BASE_CKPT = "ckpt/final.pt"
DATA_DIR = "sft_packed"
NEW_VOCAB = 49154                 # 49152 + <|im_start|>, <|im_end|>
EPOCHS = 2
MICRO_TOKENS = 8192               # 마이크로배치 토큰 수. 실측: --compile 36GB, 미사용 시 72GB
BUCKETS = [64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048]   # 배치 길이는 이 값들로만 맞춤
GPU_MEM_FRACTION = 0.75           # GB10은 GPU/시스템 메모리 공유: 넘으면 서버 대신 학습만 OOM으로 멈추게
GRAD_ACCUM = 8                    # 유효 배치 약 65K 토큰
LR_PEAK = 5e-5
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CKPT_DIR = "ckpt"
CKPT_EVERY_STEPS = 500
VAL_EVERY_STEPS = 250
LOG_EVERY_STEPS = 5
PAD_ID = 1


class ChatData:
    def __init__(self, path):
        d = np.load(path)
        self.ids, self.mask, self.off = d["ids"], d["mask"], d["offsets"]
        self.lens = np.diff(self.off)

    def __len__(self):
        return len(self.lens)

    def batches(self, rng):
        """길이 구간별로 모아 섞고, 구간마다 고정 크기(MICRO_TOKENS // 길이)로 자른 뒤 묶음 순서를 섞음."""
        bucket = np.searchsorted(BUCKETS, self.lens)
        out = []
        for k, L in enumerate(BUCKETS):
            idx = rng.permutation(np.nonzero(bucket == k)[0])
            bs = MICRO_TOKENS // L
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                # 마지막 자투리는 빈 행(-1)으로 채워 배치 모양을 고정 (torch.compile 재컴파일 방지)
                out.append(np.concatenate([b, np.full(bs - len(b), -1, dtype=b.dtype)]))
        return [out[j] for j in rng.permutation(len(out))]

    def collate(self, idxs, device):
        L = BUCKETS[int(np.searchsorted(BUCKETS, self.lens[idxs[idxs >= 0]].max()))]
        x = np.full((len(idxs), L), PAD_ID, dtype=np.int64)
        y = np.full((len(idxs), L), -100, dtype=np.int64)
        for r, i in enumerate(idxs):
            if i < 0:
                continue
            a, b = self.off[i], self.off[i + 1]
            ids = self.ids[a:b].astype(np.int64)
            x[r, :b - a] = ids
            y[r, :b - a] = np.where(self.mask[a:b] == 1, ids, -100)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # 입력 x[:, :-1] -> 정답 y[:, 1:]
        return x[:, :-1].to(device), y[:, 1:].to(device)


def lr_at(step, total):
    if step < WARMUP_STEPS:
        return LR_PEAK * (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return LR_PEAK * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def load_base(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    cfg = PerdixConfig(**{**ck["config"], "vocab_size": NEW_VOCAB})
    old = sd["tok_emb.weight"]
    if old.shape[0] < NEW_VOCAB:   # 새 토큰 임베딩은 기존 임베딩 평균으로 시작
        extra = old.mean(0, keepdim=True).repeat(NEW_VOCAB - old.shape[0], 1)
        sd["tok_emb.weight"] = sd["lm_head.weight"] = torch.cat([old, extra])
    model = PerdixSLM(cfg)
    model.load_state_dict(sd)
    return model.to(device), cfg


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    rng = np.random.default_rng(0)
    tot, n = 0.0, 0
    for idxs in data.batches(rng):
        x, y = data.collate(idxs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        k = int((y != -100).sum())
        tot += loss.item() * k; n += k
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)

    train, val = ChatData(f"{DATA_DIR}/train.npz"), ChatData(f"{DATA_DIR}/val.npz")
    raw, cfg = load_base(BASE_CKPT, device)
    print(f"모델: {sum(p.numel() for p in raw.parameters())/1e6:.1f}M params, vocab {cfg.vocab_size}", flush=True)
    model = raw
    if args.compile:   # 배치 모양이 BUCKETS 개수만큼만 나오므로 그만큼만 컴파일된다
        torch._dynamo.config.recompile_limit = 4 * len(BUCKETS)
        model = torch.compile(raw)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": other, "weight_decay": 0.0}],
                            lr=LR_PEAK, betas=(0.9, 0.95), fused=True)

    rng = np.random.default_rng(1234)
    plan = [b for _ in range(EPOCHS) for b in train.batches(rng)]   # 전체 마이크로배치 순서를 미리 고정
    total_steps = len(plan) // GRAD_ACCUM
    print(f"train {len(train)}건 / val {len(val)}건 | 마이크로배치 {len(plan)}개 "
          f"-> {total_steps} 스텝 ({EPOCHS} epoch)", flush=True)
    if args.smoke:
        total_steps = 20

    step = 0
    latest = os.path.join(CKPT_DIR, "sft_latest.pt")
    if os.path.exists(latest) and not args.smoke:
        ck = torch.load(latest, map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"재개: step {step}", flush=True)

    model.train()
    t_log, tok_log = time.time(), 0
    while step < total_steps:
        lr = lr_at(step, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for k in range(GRAD_ACCUM):
            x, y = train.collate(plan[step * GRAD_ACCUM + k], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            loss_acc += loss.item() / GRAD_ACCUM
            tok_log += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1

        if step % LOG_EVERY_STEPS == 0:
            dt = time.time() - t_log
            eta = (total_steps - step) * dt / LOG_EVERY_STEPS / 3600
            print(f"step {step}/{total_steps} | loss {loss_acc:.4f} | lr {lr:.2e} "
                  f"| {tok_log/dt/1e3:.1f}K tok/s | 남은 시간 {eta:.1f}h "
                  f"| mem {torch.cuda.max_memory_allocated()/2**30:.0f}/{torch.cuda.memory_reserved()/2**30:.0f}GB", flush=True)
            t_log, tok_log = time.time(), 0
        if step % VAL_EVERY_STEPS == 0 or step == total_steps:
            print(f"  val loss {evaluate(model, val, device):.4f} (step {step})", flush=True)
            t_log, tok_log = time.time(), 0
        if step % CKPT_EVERY_STEPS == 0 and not args.smoke:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                        "step": step, "config": cfg.__dict__}, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
            print(f"  ckpt 저장: step {step}", flush=True)

    print("=== SFT 완료 ===", flush=True)
    if not args.smoke:
        torch.save({"model": raw.state_dict(), "config": cfg.__dict__},
                   os.path.join(CKPT_DIR, "sft_final.pt"))


if __name__ == "__main__":
    main()
