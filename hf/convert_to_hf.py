"""pretrain.py / sft_train.py 체크포인트(.pt)를 허깅 페이스 형식 폴더로 변환.

사용:
    python3 hf/convert_to_hf.py ckpt/final.pt Perdix-1.1B-Base            # 베이스
    python3 hf/convert_to_hf.py ckpt/sft_final.pt Perdix-1.1B-Instruct    # SFT
베이스/SFT는 체크포인트의 vocab 크기로 구분한다(SFT는 <|im_start|>, <|im_end|> 2개가 늘어남).
SFT면 tokenizer_chat.json과 ChatML chat_template을 넣고 eos를 <|im_end|>로 잡는다.
결과 폴더: config.json, model.safetensors, tokenizer 파일, 모델 코드, README.md(모델 카드)
"""
import json
import os
import shutil
import sys

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from hf.configuration_perdix import PerdixConfig
from hf.modeling_perdix import PerdixForCausalLM

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

ckpt_path, out_dir = sys.argv[1], sys.argv[2]
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}

base_tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
instruct = ck["config"]["vocab_size"] > base_tok.get_vocab_size()
tok_file = os.path.join(ROOT, "tokenizer", "tokenizer_chat.json" if instruct else "tokenizer.json")
raw_tok = Tokenizer.from_file(tok_file)
assert raw_tok.get_vocab_size() == ck["config"]["vocab_size"], "토크나이저와 체크포인트 vocab 불일치"
eos = "<|im_end|>" if instruct else "<|endoftext|>"
print("종류:", "SFT(instruct)" if instruct else "베이스", "| eos", eos, raw_tok.token_to_id(eos))

PerdixConfig.register_for_auto_class()
PerdixForCausalLM.register_for_auto_class("AutoModelForCausalLM")
cfg = PerdixConfig(**ck["config"], eos_token_id=raw_tok.token_to_id(eos),
                   bos_token_id=raw_tok.token_to_id("<|endoftext|>"))
model = PerdixForCausalLM(cfg)
print(model.load_state_dict(sd, strict=True))
n = sum(p.numel() for p in model.parameters())
print(f"{n/1e6:.1f}M params, dtype {next(model.parameters()).dtype}")
model.generation_config.use_cache = False
model.generation_config.eos_token_id = cfg.eos_token_id
model.save_pretrained(out_dir, safe_serialization=True)

tok = PreTrainedTokenizerFast(
    tokenizer_file=tok_file,
    bos_token="<|endoftext|>", eos_token=eos, pad_token="<|pad|>",
    additional_special_tokens=["<|im_start|>", "<|im_end|>"] if instruct else None,
    model_max_length=cfg.max_seq_len, clean_up_tokenization_spaces=False)
if instruct:
    tok.chat_template = CHAT_TEMPLATE
tok.save_pretrained(out_dir)

# transformers 5.x가 쓴 tokenizer_config.json(tokenizer_class=TokenizersBackend, backend, extra_special_tokens)은
# 4.x가 읽지 못한다. 양쪽 다 읽는 형태로 고쳐 쓴다.
tc_path = os.path.join(out_dir, "tokenizer_config.json")
with open(tc_path) as f:
    tc = json.load(f)
tc.pop("backend", None)
extra = tc.pop("extra_special_tokens", None) or tc.pop("additional_special_tokens", None)
if extra:
    tc["additional_special_tokens"] = list(extra)
tc["tokenizer_class"] = "PreTrainedTokenizerFast"
if instruct:
    tc["chat_template"] = CHAT_TEMPLATE      # 4.x는 chat_template.jinja 파일 대신 이 키를 본다
with open(tc_path, "w") as f:
    json.dump(tc, f, ensure_ascii=False, indent=2)

card = os.path.join(HERE, "MODEL_CARD_INSTRUCT.md" if instruct else "MODEL_CARD.md")
if os.path.exists(card):
    shutil.copy(card, os.path.join(out_dir, "README.md"))
print("saved:", sorted(os.listdir(out_dir)))
