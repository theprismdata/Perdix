# HF causal LM bf16 추론 스모크 테스트 (DGX Spark GB10)
# 학습 컨테이너와 메모리를 공유하므로 docker --memory 상한과 함께 실행할 것
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "/ws/model")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("tokenizer 로드")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

log("모델 로드 시작 (fp32 50GB -> bf16 캐스팅, 수 분 소요)")
t0 = time.time()
# GDA(grouped_ratio>1)는 SDPA/vanilla 경로에 assert가 있어 flash_attention_2 필수
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()
log(f"모델 로드 완료: {time.time()-t0:.0f}s")
log(f"GPU 메모리 할당: {torch.cuda.memory_allocated()/1e9:.1f}GB")

PROMPTS = [
    ("한국어 상식", "대한민국의 수도는 어디야? 그 도시의 역사도 두 문장으로 알려줘."),
    ("수학 (GSM8K풍)", "나탈리아는 4월에 친구 48명에게 클립을 팔았고, 5월에는 4월의 절반만큼 팔았습니다. 나탈리아가 4월과 5월에 판 클립은 모두 몇 개인가요?"),
    ("코딩", "공백과 대소문자를 무시하고 문자열이 팰린드롬인지 확인하는 파이썬 함수를 작성해줘."),
]

for name, q in PROMPTS:
    log(f"=== [{name}] 생성 시작 ===")
    msgs = [{"role": "user", "content": q}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    n_in = enc["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    new_tokens = out.shape[1] - n_in
    text = tok.decode(out[0][n_in:], skip_special_tokens=False)
    log(f"[{name}] {new_tokens} tokens / {dt:.0f}s = {new_tokens/dt:.1f} tok/s")
    print("-" * 60, flush=True)
    print(text, flush=True)
    print("=" * 60, flush=True)

log("전체 테스트 완료")
