"""데이터셋 토큰화 + 패킹 (DGX에서 실행).

parquet의 text 컬럼을 tokenizer/tokenizer.json으로 인코딩하고
문서 끝마다 <|endoftext|>를 붙여 uint16 바이너리(.bin)로 저장한다.
파일 단위로 재개 가능(이미 있는 .bin은 건너뜀).

실행:
    python3 tokenize_pack.py [소스이름 ...]   # 인자 없으면 전체
결과:
    packed/<source>/<원본파일명>.bin
    packed/<source>/meta.jsonl  (파일별 토큰 수 기록)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

TOKENIZER_PATH = "tokenizer/tokenizer.json"
BATCH_DOCS = 2000  # encode_batch 단위 (rust 내부에서 전체 코어 병렬화)

SOURCES = {
    "korean": "data/fineweb2-kor/data/kor_Hang/train/*.parquet",
    "web_en": "data/dclm-baseline/filtered/**/*.parquet",
    "math":   "data/finemath-4plus/finemath-4plus/*.parquet",
}


def pack_file(tok, eot_id, src_path, out_path, label=""):
    tmp_path = out_path + ".tmp"
    total_tokens = 0
    t0 = time.time()
    pf = pq.ParquetFile(src_path)
    n_rg = pf.num_row_groups
    encode = getattr(tok, "encode_batch_fast", tok.encode_batch)
    with open(tmp_path, "wb") as out:
        for rg in range(n_rg):
            texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for i in range(0, len(texts), BATCH_DOCS):
                batch = [t for t in texts[i:i + BATCH_DOCS] if t]
                if not batch:
                    continue
                encs = encode(batch)
                flat = []
                for e in encs:
                    flat.extend(e.ids)
                    flat.append(eot_id)
                np.asarray(flat, dtype=np.uint16).tofile(out)
                total_tokens += len(flat)
            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            eta = (n_rg - rg - 1) * (elapsed / (rg + 1))
            print(f"  {label} rg {rg + 1}/{n_rg} | {total_tokens/1e6:7.1f}M tok "
                  f"| {rate/1e6:.2f}M tok/s | 남은시간 {eta/60:4.1f}분", flush=True)
    os.replace(tmp_path, out_path)
    return total_tokens


def main():
    only = set(sys.argv[1:]) or set(SOURCES)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None
    assert tok.get_vocab_size() <= 65536, "uint16 범위 초과"

    for name, pattern in SOURCES.items():
        if name not in only:
            continue
        files = sorted(glob.glob(pattern, recursive=True))
        out_dir = os.path.join("packed", name)
        os.makedirs(out_dir, exist_ok=True)
        # 같은 파일명이 여러 하위 폴더에 반복되면(DCLM의 global-shard 구조)
        # 경로 성분을 붙여 고유한 출력명을 만든다
        from collections import Counter
        base_counts = Counter(os.path.basename(p) for p in files)

        def out_name(path):
            base = os.path.basename(path)
            if base_counts[base] == 1:
                return base.replace(".parquet", ".bin")
            parts = path.split(os.sep)
            shard_parts = [p for p in parts if "shard" in p and not p.endswith(".parquet")]
            prefix = "__".join(shard_parts)
            return f"{prefix}__{base}".replace(".parquet", ".bin")
        meta_path = os.path.join(out_dir, "meta.jsonl")
        done = set()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                done = {json.loads(l)["file"] for l in f if l.strip()}
        print(f"[{name}] {len(files)} files ({len(done)} already packed)", flush=True)

        src_total = 0
        for n, path in enumerate(files, 1):
            base = out_name(path)
            if base in done:
                continue
            out_path = os.path.join(out_dir, base)
            ntok = pack_file(tok, eot_id, path, out_path,
                             label=f"[{name} {n}/{len(files)}]")
            src_total += ntok
            with open(meta_path, "a") as f:
                f.write(json.dumps({"file": base, "tokens": ntok}) + "\n")
            print(f"[{name}] {n}/{len(files)} {base}: {ntok/1e6:.1f}M tokens", flush=True)

        with open(meta_path) as f:
            grand = sum(json.loads(l)["tokens"] for l in f if l.strip())
        print(f"[{name}] TOTAL {grand/1e9:.2f}B tokens", flush=True)

    print("=== packing complete ===", flush=True)


if __name__ == "__main__":
    main()
