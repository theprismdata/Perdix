"""SFT용 대화 데이터 준비 (DGX에서 실행).

1) tokenizer/tokenizer.json에 <|im_start|>, <|im_end|>를 추가해 tokenizer/tokenizer_chat.json 저장
2) sft_data/의 원본을 ChatML 형식으로 토큰화
     <|im_start|>user\n질문<|im_end|>\n<|im_start|>assistant\n답변<|im_end|>\n
3) 답변 구간(답변 본문 + <|im_end|>)만 loss 대상으로 표시해 sft_packed/{train,val}.npz 저장

실행: python3 sft_data.py
"""
import glob
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

MAX_LEN = 2048
VAL_SIZE = 1000
OUT_DIR = "sft_packed"
SEED = 1234

# (소스 이름, 최대 건수, 반복 횟수). None이면 전부 사용
SMOLTALK = [
    ("everyday-conversations", None, 3),   # 인사·일상 대화. 적어서 3번 반복
    ("smol-magpie-ultra", None, 1),
    ("smol-constraints", None, 1),
    ("smol-rewrite", 20000, 1),
    ("smol-summarize", 20000, 1),
    ("metamathqa-50k", 25000, 1),
    ("numina-cot-100k", 15000, 1),
]
KOREAN = [
    ("kullm-v2", None),
    ("koalpaca", 20000),
    ("OIG-smallchip2-ko", 30000),
    ("korquad-chat", 8000),
]
ROLE_TAG = {"<sys>": "system", "<usr>": "user", "<bot>": "assistant"}


def load_smoltalk(name, limit, rng):
    rows = []
    for f in sorted(glob.glob(f"sft_data/smoltalk/data/{name}/train-*.parquet")):
        rows += pq.read_table(f, columns=["messages"]).column("messages").to_pylist()
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def load_kullm():
    out = []
    with open("sft_data/kullm-v2/kullm-v2.jsonl") as f:
        for line in f:
            r = json.loads(line)
            q = r["instruction"].strip()
            if r.get("input", "").strip():
                q += "\n\n" + r["input"].strip()
            if q and r["output"].strip():
                out.append([{"role": "user", "content": q},
                            {"role": "assistant", "content": r["output"].strip()}])
    return out


def load_tagged(path, limit, rng):
    """'<usr> ...\\n<bot> ...' 형식(open-korean-instructions)을 메시지 목록으로."""
    out = []
    with open(path) as f:
        for line in f:
            parts = re.split(r"(<sys>|<usr>|<bot>)", json.loads(line)["text"])
            msgs = [{"role": ROLE_TAG[parts[i]], "content": parts[i + 1].strip()}
                    for i in range(1, len(parts) - 1, 2)]
            if msgs:
                out.append(msgs)
    rng.shuffle(out)
    return out[:limit] if limit else out


def valid(msgs):
    if not msgs or any(not (m.get("content") or "").strip() for m in msgs):
        return False
    roles = [m["role"] for m in msgs]
    body = roles[1:] if roles[0] == "system" else roles
    return (len(body) >= 2 and all(r == ("user" if i % 2 == 0 else "assistant")
                                   for i, r in enumerate(body)))


def encode_all(tok, convs):
    """대화 목록 -> (ids, loss mask) 목록. 2048토큰을 넘으면 들어가는 턴까지만 쓴다."""
    heads = {r: tok.encode(f"<|im_start|>{r}\n").ids for r in ("system", "user", "assistant")}
    end, nl = tok.encode("<|im_end|>").ids, tok.encode("\n").ids
    flat = [m["content"].strip() for c in convs for m in c]
    enc, B = [], 4096
    for i in range(0, len(flat), B):
        enc += [e.ids for e in tok.encode_batch(flat[i:i + B])]
        if (i // B) % 50 == 0:
            print(f"  토큰화 {i}/{len(flat)}", flush=True)
    out, k = [], 0
    for c in convs:
        ids, mask, cut = [], [], None
        for m in c:
            body = enc[k]; k += 1
            if cut is not None:
                continue
            is_a = m["role"] == "assistant"
            seg = heads[m["role"]] + body + end + nl
            seg_mask = [0] * len(heads[m["role"]]) + [int(is_a)] * (len(body) + len(end)) + [0] * len(nl)
            if len(ids) + len(seg) > MAX_LEN:
                cut = True
                continue
            ids += seg; mask += seg_mask
            if is_a:
                good = len(ids)          # 마지막으로 답변이 끝난 지점
        if any(mask):
            out.append((ids[:good], mask[:good]))
    return out


def save(path, items):
    lens = np.array([len(i) for i, _ in items], dtype=np.int64)
    np.savez(path,
             ids=np.concatenate([np.array(i, dtype=np.uint16) for i, _ in items]),
             mask=np.concatenate([np.array(m, dtype=np.uint8) for _, m in items]),
             offsets=np.concatenate([[0], np.cumsum(lens)]))
    print(f"{path}: {len(items)}건, {lens.sum()/1e6:.1f}M 토큰 "
          f"(평균 {lens.mean():.0f}, 최대 {lens.max()}), "
          f"loss 대상 {sum(sum(m) for _, m in items)/1e6:.1f}M 토큰", flush=True)


def main():
    rng = random.Random(SEED)
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    base_vocab = tok.get_vocab_size()
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tok.save("tokenizer/tokenizer_chat.json")
    print(f"vocab {base_vocab} -> {tok.get_vocab_size()} "
          f"(im_start={tok.token_to_id('<|im_start|>')}, im_end={tok.token_to_id('<|im_end|>')})")

    convs, stats = [], {}
    for name, limit, repeat in SMOLTALK:
        rows = [r for r in load_smoltalk(name, limit, rng) if valid(r)]
        stats[name] = len(rows) * repeat
        convs += rows * repeat
    for name, limit in KOREAN:
        rows = load_kullm() if name == "kullm-v2" else load_tagged(f"sft_data/oki/{name}.json", limit, rng)
        rows = [r for r in rows if valid(r)]
        stats[name] = len(rows)
        convs += rows
    print("소스별 건수:", json.dumps(stats, ensure_ascii=False), "| 합계", len(convs), flush=True)

    rng.shuffle(convs)
    items = encode_all(tok, convs)
    print(f"토큰화 후 {len(items)}건 (길이 초과로 제외 {len(convs) - len(items)}건)")
    os.makedirs(OUT_DIR, exist_ok=True)
    save(os.path.join(OUT_DIR, "val.npz"), items[:VAL_SIZE])
    save(os.path.join(OUT_DIR, "train.npz"), items[VAL_SIZE:])
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
