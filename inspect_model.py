"""모델 파일 구조 분석 도구.

가중치를 메모리에 로드하지 않고(또는 mmap으로) 구조만 빠르게 분석한다.

지원 형식:
  - .safetensors        : 헤더만 파싱 (50GB 파일도 즉시)
  - .pt / .pth          : torch 체크포인트 (pretrain.py 형식 자동 인식)
  - HF 모델 디렉토리     : config.json + *.safetensors 종합

사용:
    python3 inspect_model.py <HF 모델 디렉토리>
    python3 inspect_model.py ckpt/latest.pt
    python3 inspect_model.py path/to/model.safetensors --tensors   # 전체 텐서 나열
"""
import argparse
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.2f}T"


def read_safetensors_header(path):
    """(tensors: {name: (dtype, shape)}, metadata) — 헤더 8바이트+JSON만 읽음"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {})
    return {k: (v["dtype"], v["shape"]) for k, v in hdr.items()}, meta


def load_pt_tensors(path):
    """pretrain.py 체크포인트/일반 .pt에서 {name: (dtype, shape)} 추출 (mmap 시도)"""
    import torch
    try:
        obj = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    extra = {}
    if isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "items"):
        for k in ("step", "tokens_done", "config"):
            if k in obj:
                extra[k] = obj[k]
        obj = obj["model"]
    if not hasattr(obj, "items"):
        raise ValueError("state_dict 형태가 아닙니다")
    tensors = {k: (str(v.dtype).replace("torch.", "").upper()
                   .replace("FLOAT32", "F32").replace("BFLOAT16", "BF16")
                   .replace("FLOAT16", "F16"), list(v.shape))
               for k, v in obj.items() if hasattr(v, "shape")}
    return tensors, extra


LAYER_RE = re.compile(r"(?:^|\.)(?:layers?|blocks?|h)\.(\d+)\.")


def component_of(name):
    """텐서 이름 → 컴포넌트 분류"""
    n = name.lower()
    if "embed" in n or "wte" in n or "tok_emb" in n:
        return "embedding"
    if "lm_head" in n:
        return "lm_head"
    if any(k in n for k in ("attn", "attention", "wq", "wk", "wv", "wo",
                            "q_proj", "k_proj", "v_proj", "o_proj", "lambda", "subln")):
        return "attention"
    if any(k in n for k in ("mlp", "ffn", "feed", "gate_proj", "up_proj",
                            "down_proj", "up.", "down.", "act")):
        return "ffn"
    if "norm" in n or "ln" in n:
        return "norm"
    return "etc"


def analyze(tensors, title, extra=None):
    total = sum(math.prod(s) for _, s in tensors.values())
    total_bytes = sum(math.prod(s) * DTYPE_BYTES.get(d, 4) for d, s in tensors.values())
    dtypes = defaultdict(int)
    comps = defaultdict(int)
    layers = defaultdict(int)
    no_layer = defaultdict(int)
    for name, (dtype, shape) in tensors.items():
        n = math.prod(shape)
        dtypes[dtype] += n
        comps[component_of(name)] += n
        m = LAYER_RE.search(name)
        if m:
            layers[int(m.group(1))] += n
        else:
            no_layer[name] = n

    print(f"=== {title} ===")
    if extra:
        if "step" in extra:
            print(f"체크포인트: step {extra['step']}, "
                  f"{extra.get('tokens_done', 0)/1e9:.2f}B tokens 학습 시점")
        if "config" in extra:
            print(f"config: {extra['config']}")
    print(f"텐서 {len(tensors)}개 | 총 {total/1e9:.3f}B params | "
          f"저장 크기 {total_bytes/2**30:.1f}GiB")
    print(f"dtype: " + ", ".join(f"{d} {human(n)}" for d, n in
                                 sorted(dtypes.items(), key=lambda x: -x[1])))
    print(f"\n[컴포넌트별 파라미터]")
    for c, n in sorted(comps.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {human(n):>8}  ({n/total*100:5.1f}%)")
    if layers:
        n_layers = max(layers) + 1
        per = layers[0]
        uniform = all(v == per for v in layers.values())
        print(f"\n[레이어 구조] {n_layers}개 레이어 × {human(per)} params"
              + ("" if uniform else " (레이어별 크기 상이!)"))
        sample = [k for k in tensors if LAYER_RE.search(k)
                  and int(LAYER_RE.search(k).group(1)) == 0]
        print(f"[레이어 0 텐서 {len(sample)}개]")
        for name in sample:
            d, s = tensors[name]
            print(f"  {name:<60} {str(s):<20} {d}")
    if no_layer:
        print(f"\n[레이어 외 텐서]")
        for name, n in sorted(no_layer.items(), key=lambda x: -x[1])[:10]:
            d, s = tensors[name]
            print(f"  {name:<60} {str(s):<20} {d}")
    print(f"\n[추론 메모리 추정] fp32 {total*4/2**30:.1f}GiB | "
          f"bf16 {total*2/2**30:.1f}GiB | int8 {total/2**30:.1f}GiB | "
          f"int4 {total/2/2**30:.1f}GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true", help="전체 텐서 나열")
    args = ap.parse_args()
    p = args.path

    if os.path.isdir(p):
        cfg_path = os.path.join(p, "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
            keys = ["model_type", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "intermediate_size", "vocab_size", "max_position_embeddings",
                    "hidden_act", "tie_word_embeddings", "torch_dtype", "dtype"]
            print("[config.json]")
            for k in keys:
                if k in cfg:
                    print(f"  {k}: {cfg[k]}")
            print()
        sts = sorted(f for f in os.listdir(p) if f.endswith(".safetensors"))
        if not sts:
            sys.exit("디렉토리에 .safetensors가 없습니다")
        tensors = {}
        for f in sts:
            t, _ = read_safetensors_header(os.path.join(p, f))
            tensors.update(t)
        analyze(tensors, f"{p} ({len(sts)}개 셔드)")
    elif p.endswith(".safetensors"):
        tensors, _ = read_safetensors_header(p)
        analyze(tensors, p)
    else:
        tensors, extra = load_pt_tensors(p)
        analyze(tensors, p, extra)

    if args.tensors:
        print("\n[전체 텐서]")
        for name, (d, s) in tensors.items():
            print(f"  {name:<70} {str(s):<24} {d}")


if __name__ == "__main__":
    main()
