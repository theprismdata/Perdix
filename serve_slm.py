# 학습 중인 1.1B PerdixSLM 체크포인트 임시 서빙 (베이스 모델 → 이어쓰기 전용)
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from tokenizers import Tokenizer

sys.path.insert(0, "/ws")
from model import PerdixConfig, PerdixSLM

print("체크포인트 로드 중...", flush=True)
ck = torch.load("/ws/ckpt/latest.pt", map_location="cpu", weights_only=False)
STEP, TOKENS_B = ck["step"], ck["tokens_done"] / 1e9
cfg = PerdixConfig(**ck["config"])
model = PerdixSLM(cfg)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
del ck
model = model.to("cuda", dtype=torch.bfloat16).eval()
tok = Tokenizer.from_file("/ws/tokenizer/tokenizer.json")
print(f"로드 완료: step {STEP} ({TOKENS_B:.2f}B tokens 시점)", flush=True)

app = FastAPI()
lock = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok", "model": f"PerdixSLM-1.1B @ step {STEP} ({TOKENS_B:.2f}B tok)",
            "note": "베이스 모델 — 질문응답 불가, 텍스트 이어쓰기 전용"}


@app.post("/complete")
def complete(body: dict):
    prompt = body.get("prompt", "")
    if not prompt:
        return {"error": "prompt를 넣어주세요"}
    max_new = min(int(body.get("max_tokens") or 150), 500)
    temp = float(body.get("temperature") or 0.8)
    top_k = int(body.get("top_k") or 50)
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device="cuda")
    t0 = time.time()
    with lock, torch.inference_mode():
        for _ in range(max_new):
            logits, _ = model(x[:, -cfg.max_seq_len:])
            logits = logits[0, -1].float() / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            x = torch.cat([x, nxt.view(1, 1)], dim=1)
    dt = time.time() - t0
    full = tok.decode(x[0].tolist())
    return {"prompt": prompt, "completion": full[len(prompt):] if full.startswith(prompt) else full,
            "checkpoint": f"step {STEP} ({TOKENS_B:.2f}B tok)",
            "tokens_per_sec": round(max_new / dt, 1)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
