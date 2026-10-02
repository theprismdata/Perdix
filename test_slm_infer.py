# 학습 중인 350M PerdixSLM 체크포인트 추론 스모크 테스트 (베이스 모델 → 이어쓰기)
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/ws")
from model import PerdixConfig, PerdixSLM

CKPT = "/ws/ckpt/latest.pt"
TOK = "/ws/tokenizer/tokenizer.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("체크포인트 로드")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
log(f"step {ck['step']}, {ck['tokens_done']/1e9:.2f}B tokens 시점")

cfg = PerdixConfig(**ck["config"])
model = PerdixSLM(cfg)
# torch.compile 학습 시 붙는 _orig_mod. 접두사 제거
sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
model.load_state_dict(sd)
model = model.to("cuda", dtype=torch.bfloat16).eval()
log(f"모델 로드 완료 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

tok = Tokenizer.from_file(TOK)

@torch.inference_mode()
def generate(prompt, max_new=120, temp=0.8, top_k=50):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device="cuda")
    t0 = time.time()
    for _ in range(max_new):
        logits, _ = model(x[:, -cfg.max_seq_len:])
        logits = logits[0, -1].float() / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    dt = time.time() - t0
    text = tok.decode(x[0].tolist())
    return text, max_new / dt

PROMPTS = [
    "대한민국의 수도 서울은",
    "김치는 한국의 전통 음식으로",
    "The capital city of France is",
    "숫자 1부터 5까지 더하면",
]

for p in PROMPTS:
    text, tps = generate(p)
    log(f"({tps:.0f} tok/s)")
    print("-" * 60, flush=True)
    print(text, flush=True)
    print("=" * 60, flush=True)

log("350M 추론 테스트 완료")
