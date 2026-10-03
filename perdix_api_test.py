# api_server.py 배치 테스트. 질문 묶음을 보내 답변·토큰 수·속도·종료 사유를 확인한다. 표준 라이브러리만 사용.
#
#   python3 perdix_api_test.py                          # 기본 질문 묶음 (DGX 로컬 망 주소 사용)
#   python3 perdix_api_test.py http://host:8000         # 서버 주소 지정
#   python3 perdix_api_test.py -q "질문" -q "another"    # 직접 질문
#   python3 perdix_api_test.py --temp 0 --top-p 1       # greedy로 (결과가 매번 같아야 정상)
#
# 같은 망이 아니면 SSH 터널을 뚫고 http://localhost:8000 으로 접속한다:
#   ssh -N -L 8000:localhost:8000 prismdata.iptime.org
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("PERDIX_URL", "http://192.168.0.200:8000")   # DGX (gx10-0782) 로컬 망 주소
PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "김치가 무엇인지 두 문장으로 설명해 주세요.",
    "What is the capital of France?",
    "사과 3개와 배 5개가 있으면 과일은 모두 몇 개인가요?",
    "Write a short poem about the sea.",
    "안녕! 오늘 기분이 어때?",
]


def health(url):
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        return json.loads(r.read())


def chat(url, messages, **sampling):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps({"messages": messages, **sampling}, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("-q", "--question", action="append")
    ap.add_argument("--temp", type=float, help="0이면 greedy (서버 기본 0.7)")
    ap.add_argument("--top-k", type=int, help="0이면 끄기 (기본 50)")
    ap.add_argument("--top-p", type=float, help="1이면 끄기 (기본 0.9)")
    ap.add_argument("--rep", type=float, help="repetition_penalty (기본 1.1)")
    ap.add_argument("--max-tokens", type=int, help="기본 256")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    sampling = {k: v for k, v in {"temperature": args.temp, "top_k": args.top_k, "top_p": args.top_p,
                                  "repetition_penalty": args.rep, "max_tokens": args.max_tokens}.items()
                if v is not None}

    try:
        h = health(url)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"서버에 연결할 수 없습니다 ({url}): {e}\n"
                 f"같은 망이 아니면: ssh -N -L 8000:localhost:8000 prismdata.iptime.org 후 http://localhost:8000")
    print(f"서버: {h['model']} | 기본값 {h['defaults']} | GPU {h['gpu_mem_gb']}GB")

    questions = args.question or PROMPTS
    n_stop = 0
    for q in questions:
        r = chat(url, [{"role": "user", "content": q}], **sampling)
        if "error" in r:
            print(f"Q: {q}\n  서버 오류: {r['error']}")
            continue
        c, u = r["choices"][0], r["usage"]
        n_stop += c["finish_reason"] == "stop"
        print("-" * 60)
        print(f"Q: {q}")
        print(f"A: {c['message']['content']}")
        print(f"   ({u['completion_tokens']} tokens, {u['tokens_per_sec']} tok/s, finish={c['finish_reason']})")
    print("=" * 60)
    print(f"{len(questions)}개 중 {n_stop}개가 <|im_end|>로 정상 종료")


if __name__ == "__main__":
    main()
