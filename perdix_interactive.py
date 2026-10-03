# api_server.py에 붙어 터미널에서 Perdix와 대화한다. 멀티턴, 스트리밍. 표준 라이브러리만 사용.
#
#   python3 perdix_interactive.py                       # DGX 로컬 망 주소로 접속
#   python3 perdix_interactive.py http://host:8000      # 서버 주소 지정
#   python3 perdix_interactive.py --temp 0.5 --max-tokens 512
#
# 대화 중 명령:
#   /reset            대화 기록 비우기
#   /system <문장>    system 프롬프트 설정 (SFT 데이터 대부분이 system 없이 학습돼 효과는 제한적)
#   /temp <값>        temperature 변경 (0이면 greedy)
#   /quit             종료 (빈 줄, Ctrl-D, Ctrl-C도 종료)
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("PERDIX_URL", "http://192.168.0.200:8000")   # DGX (gx10-0782) 로컬 망 주소


def health(url):
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        return json.loads(r.read())


def stream(url, messages, **sampling):
    """답변 조각(str)을 순서대로 yield. 마지막에 finish_reason을 StopIteration 값으로 돌려준다."""
    body = {"messages": messages, "stream": True, **sampling}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    finish = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            ch = json.loads(data)["choices"][0]
            finish = ch.get("finish_reason") or finish
            piece = ch["delta"].get("content")
            if piece:
                yield piece
    return finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("--system", help="system 프롬프트")
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
    print(f"{h['model']} 에 연결됨. /reset /system /temp /quit, 빈 줄로 종료")

    system = args.system
    history = []
    while True:
        try:
            q = input("\n나> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q == "/quit":
            break
        if q == "/reset":
            history = []
            print("(대화 기록을 비웠습니다)")
            continue
        if q.startswith("/system"):
            system = q[len("/system"):].strip() or None
            history = []
            print(f"(system = {system!r}, 대화 기록 초기화)")
            continue
        if q.startswith("/temp"):
            try:
                sampling["temperature"] = float(q.split()[1])
                print(f"(temperature = {sampling['temperature']})")
            except (IndexError, ValueError):
                print("(사용법: /temp 0.7)")
            continue

        history.append({"role": "user", "content": q})
        messages = ([{"role": "system", "content": system}] if system else []) + history
        print("Perdix> ", end="", flush=True)
        answer, t0 = "", time.time()
        try:
            gen = stream(url, messages, **sampling)
            while True:
                try:
                    piece = next(gen)
                except StopIteration as done:
                    finish = done.value
                    break
                answer += piece
                print(piece, end="", flush=True)
        except (urllib.error.URLError, OSError) as e:
            print(f"\n(요청 실패: {e})")
            history.pop()
            continue
        dt = time.time() - t0
        note = " [max_tokens 도달]" if finish == "length" else ""
        print(f"\n   ({dt:.1f}s{note})")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
