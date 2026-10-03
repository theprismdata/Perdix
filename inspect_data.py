"""학습 데이터 텍스트 뷰어.

외장하드의 원본 parquet에서 문서를 문자열로 출력한다.

사용 (화면 보기):
    python3 inspect_data.py korean              # 한국어 문서 3개
    python3 inspect_data.py web_en --n 5        # 영어 5개
    python3 inspect_data.py math --skip 100     # 100번째부터
    python3 inspect_data.py korean --chars 0    # 전문 출력 (자르지 않음)
    python3 inspect_data.py korean --url        # 출처 URL도 표시

사용 (파일로 추출):
    python3 inspect_data.py all --n 1000 --out samples/
        # 세 소스 각 1,000문서씩 → samples/korean.jsonl 등 3개 파일
    python3 inspect_data.py korean --n 0 --all-files --out dump/
        # 한국어 전체 추출 (주의: 원문 그대로라 수십~수백 GB)
"""
import argparse
import glob
import json
import os
import random

import pyarrow.parquet as pq

DATA_ROOT = "/Volumes/2TB/llm-smallversion/data"
SOURCES = {
    "korean": f"{DATA_ROOT}/fineweb2-kor/data/kor_Hang/train/*.parquet",
    "web_en": f"{DATA_ROOT}/dclm-baseline/filtered/**/*.parquet",
    "math":   f"{DATA_ROOT}/finemath-4plus/finemath-4plus/*.parquet",
}


def dump_one(pf, out, source, has_url, limit, written):
    """parquet 하나를 out에 기록. 기록한 문서 수를 더해 반환."""
    cols = ["text"] + (["url"] if has_url else [])
    for rg in range(pf.num_row_groups):
        if written >= limit:
            break
        tbl = pf.read_row_group(rg, columns=cols)
        texts = tbl.column("text").to_pylist()
        urls = tbl.column("url").to_pylist() if has_url else [None] * len(texts)
        for text, url in zip(texts, urls):
            if written >= limit:
                break
            rec = {"source": source, "text": text}
            if url:
                rec["url"] = url
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def extract(source, args):
    """한 소스를 jsonl로 추출.

    --all-files: parquet당 jsonl 하나씩 <out>/<source>/에 생성.
      .tmp에 쓰고 완료 시 이름 변경 → 재실행하면 완료된 파일은 건너뜀(재개 가능).
    기본(단일 파일 모드): <out>/<source>.jsonl 하나에 기록.
    """
    files = sorted(glob.glob(SOURCES[source], recursive=True))
    if not files:
        raise SystemExit("parquet 없음 — 외장하드(/Volumes/2TB) 연결 확인")
    limit = args.n if args.n > 0 else float("inf")

    if args.all_files:
        out_dir = os.path.join(args.out, source)
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        for i, path in enumerate(files, 1):
            stem = os.path.basename(path).replace(".parquet", "")
            out_path = os.path.join(out_dir, stem + ".jsonl")
            if os.path.exists(out_path):
                print(f"  [{source} {i}/{len(files)}] {stem} 이미 있음, 건너뜀", flush=True)
                continue
            pf = pq.ParquetFile(path)
            has_url = "url" in pf.schema_arrow.names
            with open(out_path + ".tmp", "w") as out:
                n = dump_one(pf, out, source, has_url, limit, 0)
            os.replace(out_path + ".tmp", out_path)
            total += n
            print(f"  [{source} {i}/{len(files)}] {stem}: {n:,}문서", flush=True)
        print(f"[{source}] 완료: {out_dir}/ — 총 {total:,}문서", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{source}.jsonl")
    written = 0
    with open(out_path, "w") as out:
        for path in files[args.file:args.file + 1] if not args.all_files else files:
            pf = pq.ParquetFile(path)
            has_url = "url" in pf.schema_arrow.names
            written = dump_one(pf, out, source, has_url, limit, written)
            if written >= limit:
                break
    print(f"[{source}] 완료: {out_path} — {written:,}문서, "
          f"{os.path.getsize(out_path)/2**20:.1f}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(SOURCES) + ["all"])
    ap.add_argument("--n", type=int, default=3,
                    help="문서 수 (기본 3, --out과 함께 0이면 전체)")
    ap.add_argument("--skip", type=int, default=0, help="앞에서 건너뛸 문서 수")
    ap.add_argument("--chars", type=int, default=1000,
                    help="문서당 최대 출력 문자 수 (0 = 전문)")
    ap.add_argument("--file", type=int, default=0, help="parquet 파일 인덱스")
    ap.add_argument("--random", action="store_true", help="무작위 파일에서 무작위 위치")
    ap.add_argument("--url", action="store_true", help="문서 출처 URL 표시")
    ap.add_argument("--out", help="화면 대신 이 디렉토리에 <source>.jsonl로 추출")
    ap.add_argument("--all-files", action="store_true",
                    help="--out 추출 시 모든 parquet 파일 순회 (기본: --file 하나)")
    args = ap.parse_args()

    if args.source == "all" or args.out:
        if not args.out:
            raise SystemExit("all은 --out과 함께 사용하세요 (예: --n 1000 --out samples/)")
        for src in (SOURCES if args.source == "all" else [args.source]):
            extract(src, args)
        return

    files = sorted(glob.glob(SOURCES[args.source], recursive=True))
    if not files:
        raise SystemExit(f"parquet 없음 — 외장하드(/Volumes/2TB) 연결 확인")
    fi = random.randrange(len(files)) if args.random else args.file
    pf = pq.ParquetFile(files[fi])
    cols = ["text"] + (["url"] if args.url and "url" in pf.schema_arrow.names else [])
    print(f"[{args.source}] 파일 {fi + 1}/{len(files)}: {files[fi].split('/')[-1]} "
          f"({pf.metadata.num_rows:,} 문서)\n")

    skip = args.skip
    if args.random:
        skip = random.randrange(max(pf.metadata.num_rows - args.n, 1))
    shown = 0
    for rg in range(pf.num_row_groups):
        if shown >= args.n:
            break
        tbl = pf.read_row_group(rg, columns=cols)
        texts = tbl.column("text").to_pylist()
        urls = tbl.column("url").to_pylist() if len(cols) > 1 else [None] * len(texts)
        for text, url in zip(texts, urls):
            if skip > 0:
                skip -= 1
                continue
            if shown >= args.n:
                break
            shown += 1
            head = f"── 문서 {shown} ({len(text):,}자)"
            if url:
                head += f" | {url}"
            print(head + " " + "─" * max(0, 60 - len(head)))
            print(text if args.chars == 0 else text[:args.chars])
            if args.chars and len(text) > args.chars:
                print(f"... (이하 {len(text) - args.chars:,}자 생략)")
            print()


if __name__ == "__main__":
    main()
