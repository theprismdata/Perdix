# SFT 체크포인트(ckpt/sft_final.pt)를 OpenAI 호환 chat API로 띄우는 서버
#
#   POST /v1/chat/completions  — messages(ChatML로 변환), temperature(0이면 greedy),
#                                top_k, top_p, repetition_penalty, max_tokens, stream
#   GET  /health
#
# 환경변수: CKPT(기본 /ws/ckpt/sft_final.pt), TOK(기본 /ws/tokenizer/tokenizer_chat.json), PORT(기본 8000)
import json
import os
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PerdixConfig, PerdixSLM

CKPT = os.environ.get("CKPT", "/ws/ckpt/sft_final.pt")
TOK = os.environ.get("TOK", "/ws/tokenizer/tokenizer_chat.json")
PORT = int(os.environ.get("PORT", "8000"))
MODEL_NAME = "perdix-1.1b-sft"
MAX_NEW_TOKENS_CAP = 1024
DEFAULTS = dict(temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1, max_tokens=256)

print("체크포인트 로드 중...", flush=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = PerdixConfig(**ck["config"])
model = PerdixSLM(cfg)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
del ck
model = model.to("cuda", dtype=torch.bfloat16).eval()
tok = Tokenizer.from_file(TOK)
IM_END = tok.token_to_id("<|im_end|>")
assert IM_END is not None and IM_END < cfg.vocab_size, "토크나이저와 모델 vocab이 맞지 않음"
STOP_IDS = {i for i in (IM_END, tok.token_to_id("<|endoftext|>")) if i is not None}
print(f"로드 완료: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
      f"vocab {cfg.vocab_size}, GPU {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

app = FastAPI()
gpu_lock = threading.Lock()  # 한 번에 한 요청만 GPU 사용


def build_prompt(messages):
    """sft_data.py와 같은 ChatML. 마지막에 assistant 헤더를 붙여 답변을 받는다."""
    text = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages)
    return tok.encode(text + "<|im_start|>assistant\n").ids


def param(body, key, cast):
    v = body.get(key)
    return DEFAULTS[key] if v is None else cast(v)


def sample(logits, temp, top_k, top_p):
    if temp <= 0:
        return int(logits.argmax())
    logits = logits / temp
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
    probs = torch.softmax(logits, -1)
    if 0 < top_p < 1:
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p
        probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1))


@torch.inference_mode()
def generate(ids, temp, top_k, top_p, rep, max_new):
    """토큰 id를 하나씩 yield. 종료 토큰은 yield하지 않고 멈춘다."""
    x = torch.tensor([ids], device="cuda")
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -cfg.max_seq_len:])
        logits = logits[0, -1].float()
        if rep != 1.0 and out:
            prev = torch.tensor(sorted(set(out)), device="cuda")
            sc = logits[prev]
            logits[prev] = torch.where(sc > 0, sc / rep, sc * rep)
        nxt = sample(logits, temp, top_k, top_p)
        if nxt in STOP_IDS:
            return
        out.append(nxt)
        yield nxt
        x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], dim=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "defaults": DEFAULTS,
            "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 1)}


@app.post("/v1/chat/completions")
def chat(body: dict):
    messages = body.get("messages") or []
    if not messages or any(m.get("role") not in ("system", "user", "assistant") for m in messages):
        return {"error": "messages는 system/user/assistant role의 목록이어야 합니다"}
    ids = build_prompt(messages)
    if len(ids) >= cfg.max_seq_len - 16:
        return {"error": f"입력이 너무 깁니다 ({len(ids)} 토큰, 최대 {cfg.max_seq_len - 16})"}
    temp = param(body, "temperature", float)
    top_k = param(body, "top_k", int)
    top_p = param(body, "top_p", float)
    rep = param(body, "repetition_penalty", float)
    max_new = min(param(body, "max_tokens", int), MAX_NEW_TOKENS_CAP)
    created = int(time.time())
    rid = f"chatcmpl-{created}"

    if body.get("stream"):
        def sse():
            def chunk(delta, finish=None):
                return "data: " + json.dumps({
                    "id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"
            yield chunk({"role": "assistant", "content": ""})
            n = 0
            with gpu_lock:
                # 한글은 여러 토큰이 한 글자를 이루므로, 디코딩이 완성된 부분만 보낸다
                buf, sent = [], ""
                for t in generate(ids, temp, top_k, top_p, rep, max_new):
                    n += 1
                    buf.append(t)
                    text = tok.decode(buf)
                    if not text.endswith("�") and len(text) > len(sent):
                        yield chunk({"content": text[len(sent):]})
                        sent = text
                text = tok.decode(buf)
                if len(text) > len(sent):
                    yield chunk({"content": text[len(sent):]})
            yield chunk({}, "length" if n >= max_new else "stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    t0 = time.time()
    with gpu_lock:
        out = list(generate(ids, temp, top_k, top_p, rep, max_new))
    dt = time.time() - t0
    return {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": tok.decode(out)},
                         "finish_reason": "length" if len(out) >= max_new else "stop"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                      "total_tokens": len(ids) + len(out),
                      "tokens_per_sec": round(len(out) / max(dt, 1e-6), 1)}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
