# serve_hf_tools.py OpenAI 호환 API 툴콜 왕복 테스트 (stdlib만 사용)
import json, sys, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/v1/chat/completions"
TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}},
 {"type":"function","function":{"name":"calculator","description":"Evaluate a math expression",
  "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
]

def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=600).read().decode()
    if not stream:
        return json.loads(raw)
    return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]

def ask(msgs, **kw):
    r = post({"messages": msgs, "tools": TOOLS, "max_tokens": 2000, "show_reasoning": False, **kw})
    return r["choices"][0]

results = []
def check(name, ok, detail):
    results.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

# 1) 단일 툴콜 + 왕복
msgs = [{"role": "user", "content": "서울의 현재 날씨를 알려줘."}]
c = ask(msgs); m = c["message"]; tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and c["finish_reason"] == "tool_calls"
check("툴콜 생성", ok, json.dumps(tc, ensure_ascii=False) if tc else repr(m.get("content"))[:300])
if ok:
    json.loads(tc[0]["function"]["arguments"])  # arguments는 JSON 문자열이어야 함
    msgs += [m, {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_c": 21, "condition": "맑음"}, ensure_ascii=False)}]
    c2 = ask(msgs); m2 = c2["message"]
    check("툴 결과 반영 최종답변", not m2.get("tool_calls") and "21" in (m2.get("content") or ""), repr(m2.get("content"))[:300])

# 2) 계산기
c = ask([{"role": "user", "content": "123456 * 789 를 계산기 도구로 계산해줘."}]); tc = c["message"].get("tool_calls") or []
check("계산기 선택", bool(tc) and tc[0]["function"]["name"] == "calculator", json.dumps(tc, ensure_ascii=False)[:300])

# 3) 병렬 툴콜
c = ask([{"role": "user", "content": "서울과 부산의 날씨를 각각 알려줘."}]); tc = c["message"].get("tool_calls") or []
cities = [json.loads(t["function"]["arguments"]).get("city") for t in tc]
check("병렬 툴콜(2개)", len(tc) == 2, f"{cities}")

# 4) 도구 불필요
c = ask([{"role": "user", "content": "안녕! 너는 누구니?"}]); m = c["message"]
check("도구 불필요시 일반답변", not m.get("tool_calls") and c["finish_reason"] == "stop", repr(m.get("content"))[:150])

# 5) 기본 temperature(0.6) 반복 안정성
n = 5; hit = 0
for _ in range(n):
    tc = ask([{"role": "user", "content": "What's the weather in Paris?"}], temperature=0.6)["message"].get("tool_calls") or []
    hit += bool(tc) and tc[0]["function"]["name"] == "get_weather"
check(f"temp=0.6 반복 {n}회", hit == n, f"{hit}/{n}")

# 6) 스트리밍
chunks = post({"messages": [{"role": "user", "content": "부산 날씨 알려줘."}], "tools": TOOLS, "max_tokens": 2000, "stream": True}, stream=True)
d = chunks[0]["choices"][0]["delta"] if chunks else {}
check("스트리밍 툴콜", bool(d.get("tool_calls")) and chunks[-1]["choices"][0]["finish_reason"] == "tool_calls", json.dumps(d, ensure_ascii=False)[:300])

print(f"\n총 {sum(results)}/{len(results)} PASS")
