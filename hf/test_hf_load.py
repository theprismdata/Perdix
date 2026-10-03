"""변환된 폴더를 AutoModel로 불러와 원본 체크포인트와 logits가 같은지, 생성이 되는지 확인.

사용:
    python3 hf/test_hf_load.py Perdix-1.1B-Base ckpt/final.pt
    python3 hf/test_hf_load.py Perdix-1.1B-Instruct ckpt/sft_final.pt
토크나이저에 chat_template이 있으면(SFT) 대화 형식으로 생성해 <|im_end|>에서 멈추는지도 본다.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import PerdixConfig, PerdixSLM

hf_dir, ckpt_path = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(hf_dir)
hf = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ref = PerdixSLM(PerdixConfig(**ck["config"]))
ref.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
ref = ref.to(dev, dtype=torch.bfloat16).eval()

ids = tok("대한민국의 수도 서울은 한강을 끼고 있는 도시로", return_tensors="pt").input_ids.to(dev)
with torch.inference_mode():
    a = hf(input_ids=ids).logits.float()
    b = ref(ids)[0].float()
    loss = hf(input_ids=ids, labels=ids).loss
print("max |logit diff| =", (a - b).abs().max().item(), "| argmax 일치:", bool((a.argmax(-1) == b.argmax(-1)).all()), "| loss", round(loss.item(), 3))
print("eos:", tok.eos_token, tok.eos_token_id, "| generation eos:", hf.generation_config.eos_token_id)

torch.manual_seed(0)
if tok.chat_template:
    for q in ["대한민국의 수도는 어디인가요?", "What is the capital of France?", "김치가 무엇인지 두 문장으로 설명해 주세요."]:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")["input_ids"].to(dev)
        with torch.inference_mode():
            out = hf.generate(ids, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.9,
                              repetition_penalty=1.1)
        new = out[0, ids.shape[1]:]
        stopped = tok.eos_token_id in new.tolist()
        print("-" * 60); print("Q:", q); print("A:", tok.decode(new, skip_special_tokens=True))
        print(f"   ({len(new)} tokens, {'<|im_end|>에서 종료' if stopped else 'max_new_tokens 도달'})")
else:
    for p in ["대한민국의 수도 서울은", "김치는 한국의 전통 음식으로", "The capital city of France is", "Let x be a positive integer such that"]:
        enc = tok(p, return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = hf.generate(enc.input_ids, max_new_tokens=80, do_sample=True, temperature=0.8, top_k=50)
        print("-" * 60); print(tok.decode(out[0], skip_special_tokens=True))
