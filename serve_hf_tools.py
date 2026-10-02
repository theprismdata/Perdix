# HF causal LM용 OpenAI 호환 추론 서버 (DGX Spark GB10)
# - GDA 구조상 flash_attention_2 필수, bf16 로드 25.4GB
# - 학습(slm-train)과 GPU 공유: 요청 1개씩 직렬 처리, max_tokens 상한으로 점유 제한
import json
import os
import re
import threading
import uuid
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = os.environ.get("MODEL_PATH", "/ws/model")  # HF 모델 디렉토리
MODEL_NAME = os.path.basename(MODEL_PATH.rstrip("/"))
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 4096

print("모델 로드 중...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()
print(f"모델 로드 완료: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

app = FastAPI()
gpu_lock = threading.Lock()  # 한 번에 한 요청만 GPU 사용


def normalize(messages):
    """OpenAI 메시지를 템플릿이 기대하는 형태로: content=None -> "", tool 결과/arguments는 문자열 유지"""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        elif isinstance(m["content"], list):  # [{"type":"text","text":..}]
            m["content"] = "".join(p.get("text", "") for p in m["content"])
        out.append(m)
    return out


def build_inputs(messages, tools=None):
    text = tok.apply_chat_template(normalize(messages), tools=tools or None,
                                   add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    if enc["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise ValueError(f"입력이 {MAX_INPUT_TOKENS} 토큰을 초과합니다")
    return {k: v.to(model.device) for k, v in enc.items()}


def gen_kwargs(body):
    return dict(
        max_new_tokens=min(int(body.get("max_tokens") or 1024), MAX_NEW_TOKENS_CAP),
        do_sample=True,
        temperature=float(body.get("temperature") or 0.6),
        top_p=float(body.get("top_p") or 1.0),
        pad_token_id=tok.eos_token_id,
    )


def split_reasoning(text):
    """deepseek_r1식 <think>...</think> 분리"""
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return None, text.strip()


TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def extract_tool_calls(answer):
    """<tool_call>{json}</tool_call> -> OpenAI tool_calls. 파싱 실패한 블록은 무시."""
    calls = []
    for raw in TOOL_RE.findall(answer):
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": obj["name"],
                                       "arguments": args if isinstance(args, str)
                                       else json.dumps(args, ensure_ascii=False)}})
        except Exception:
            continue
    content = TOOL_RE.sub("", answer).strip()
    return calls, content


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME,
            "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 1)}


@app.post("/v1/chat/completions")
def chat(body: dict):
    try:
        enc = build_inputs(body["messages"], body.get("tools"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    show_reasoning = body.get("show_reasoning", True)  # false면 <think> 구간 숨김
    kwargs = gen_kwargs(body)
    created = int(time.time())

    if body.get("stream") and body.get("tools"):
        # 툴콜은 완성된 JSON이 필요하므로 전체 생성 후 한 번에 전송
        with gpu_lock, torch.inference_mode():
            out = model.generate(**enc, **kwargs)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        _, answer = split_reasoning(text)
        calls, content = extract_tool_calls(answer)
        delta = ({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}
                 if calls else {"content": content})

        def sse_once():
            base = {"object": "chat.completion.chunk", "created": created,
                    "model": MODEL_NAME.lower()}
            yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta,
                  "finish_reason": None}]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {},
                  "finish_reason": "tool_calls" if calls else "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_once(), media_type="text/event-stream")

    if body.get("stream"):
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        def run():
            with gpu_lock, torch.inference_mode():
                model.generate(**enc, streamer=streamer, **kwargs)

        threading.Thread(target=run, daemon=True).start()

        def chunk_of(piece):
            chunk = {"object": "chat.completion.chunk", "created": created,
                     "model": MODEL_NAME.lower(),
                     "choices": [{"index": 0, "delta": {"content": piece},
                                  "finish_reason": None}]}
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        def sse():
            buf, answering = "", show_reasoning
            for piece in streamer:
                if answering:
                    yield chunk_of(piece)
                    continue
                buf += piece
                if "</think>" in buf:  # 사고 종료 → 이후부터 답변 스트림
                    answering = True
                    rest = buf.split("</think>", 1)[1].lstrip("\n")
                    if rest:
                        yield chunk_of(rest)
                else:
                    yield ": thinking\n\n"  # SSE 주석 keepalive (내용 없음)
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    n_in = enc["input_ids"].shape[1]
    t0 = time.time()
    with gpu_lock, torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    dt = time.time() - t0
    n_new = out.shape[1] - n_in
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    reasoning, answer = split_reasoning(text)
    calls, content = extract_tool_calls(answer)
    msg = {"role": "assistant", "content": content or None if calls else answer}
    if calls:
        msg["tool_calls"] = calls
    if reasoning and show_reasoning:
        msg["reasoning_content"] = reasoning
    return {"object": "chat.completion", "created": created,
            "model": MODEL_NAME.lower(),
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_new,
                      "total_tokens": n_in + n_new,
                      "tokens_per_sec": round(n_new / dt, 2)}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
