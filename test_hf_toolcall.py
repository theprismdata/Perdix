# HF causal LM 툴콜 테스트: chat template에 tools를 넣고 <tool_call> JSON이 나오는지 확인
import json, os, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("MODEL_PATH", "/ws/model")
tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", attn_implementation="flash_attention_2").eval()

TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}},
 {"type":"function","function":{"name":"calculator","description":"Evaluate a math expression",
  "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
]
CASES = [
 ("날씨(한국어)", "서울의 현재 날씨를 알려줘.", "get_weather"),
 ("weather(en)", "What's the weather in Paris in celsius?", "get_weather"),
 ("계산기", "123456 * 789 를 계산기 도구로 계산해줘.", "calculator"),
 ("도구불필요", "안녕! 너는 누구니?", None),
]

def gen(msgs, n=3000):
    enc = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)

def parse(t):
    body = t.split("</think>")[-1]
    res = []
    for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.S):
        try: res.append(json.loads(m))
        except Exception as e: res.append({"_invalid": m})
    return res, ("<think>" in t), ("</think>" in t)

first = tok.apply_chat_template([{"role":"user","content":"hi"}], tools=TOOLS, add_generation_prompt=True, tokenize=False)
print("##### RENDERED PROMPT #####\n", first, "\n##########", flush=True)

for name, q, expect in CASES:
    t0 = time.time()
    raw = gen([{"role":"user","content":q}])
    calls, th, thc = parse(raw)
    ok = (calls and "_invalid" not in calls[0] and calls[0].get("name")==expect) if expect else (not calls)
    print(f"\n=== [{name}] expect={expect} -> {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s) think_open={th} think_closed={thc}")
    print("parsed:", calls)
    print("--- raw (tail 1500) ---\n", raw[-1500:], flush=True)

# 2턴: tool 결과 반영
msgs = [{"role":"user","content":"서울의 현재 날씨를 알려줘."},
        {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"서울"}}}]},
        {"role":"tool","content":json.dumps({"temp":21,"condition":"맑음"}, ensure_ascii=False)}]
raw = gen(msgs)
print("\n=== [tool 결과 후 최종답변] ===\n", raw[-1200:])
