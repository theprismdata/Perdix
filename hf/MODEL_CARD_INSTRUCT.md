---
license: apache-2.0
language:
- ko
- en
library_name: transformers
pipeline_tag: text-generation
base_model: prismdata/Perdix-1.1B-Base
tags:
- custom_code
- conversational
- instruct
- differential-attention
- polynorm
datasets:
- HuggingFaceTB/smoltalk
- nlpai-lab/kullm-v2
- heegyu/open-korean-instructions
---

# Perdix-1.1B-Instruct

[Perdix-1.1B-Base](https://huggingface.co/prismdata/Perdix-1.1B-Base)에 대화 데이터로 지시 미세조정(SFT)을 한 모델입니다. 이름은 그리스 신화의 페르딕스에서 따왔습니다. 다이달로스의 어린 제자였고 톱과 컴퍼스를 발명했는데, 탑에서 떨어지다 자고새가 되는 바람에 그 뒤로는 낮게만 날아다닌다고 합니다. 지금 이 모델 수준이 딱 그렇습니다.

A 1.1B-parameter Korean/English chat model, supervised-finetuned from Perdix-1.1B-Base on about 390K conversations. It answers simple questions and follows short instructions in Korean and English; it is small and often wrong on facts and arithmetic.

## 무엇을 할 수 있고 무엇을 못 하나

질문에 답하고 짧은 지시를 따릅니다. 한국어와 영어 모두 됩니다. 아래는 실제 출력입니다(temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.1).

> **대한민국의 수도는 어디인가요?**
> 대한민국의 수도는 서울입니다.

> **김치가 무엇인지 두 문장으로 설명해 주세요.**
> 김치는 한국의 전통 발효 식품으로, 일반적으로 절인배추에 다양한 양념을 넣고 버무린 후 김치를 담가 먹는 음식입니다.

> **What is 7 times 8?**
> To find 7 times 8, we multiply 7 by 8.
> The answer is 56.

> **안녕! 오늘 기분이 어때?**
> 기분이 좋습니다! 오늘은 무엇을 도와드릴까요?

못 하는 것도 분명합니다.

- **사실과 계산은 자주 틀립니다.** "1부터 10까지 더하면?"에 1+1=2, 2+1=3… 식으로 엉뚱한 풀이를 내놓습니다. 1.1B 크기의 한계입니다.
- **긴 추론, 코드 작성, 툴콜은 안 됩니다.** 그런 데이터로 학습하지 않았습니다.
- **대화 기록은 짧게만 유지됩니다.** 문맥 길이가 2,048토큰이고, 멀티턴은 몇 턴 정도만 자연스럽습니다.
- 학습 데이터가 웹과 공개 대화 데이터라 편향되거나 부적절한 내용이 나올 수 있습니다. 사실 확인이 필요한 용도에는 쓰면 안 됩니다.

## 사용법

모델 코드가 저장소에 들어 있어서 `trust_remote_code=True`가 필요합니다. 대화 형식은 ChatML이고 `chat_template`에 들어 있습니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "prismdata/Perdix-1.1B-Instruct"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

messages = [{"role": "user", "content": "김치가 무엇인지 두 문장으로 설명해 주세요."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_dict=True, return_tensors="pt")["input_ids"].to("cuda")
out = model.generate(ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.9,
                     repetition_penalty=1.1)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

알아 둘 점이 있습니다.

- 답변은 `<|im_end|>`에서 끝납니다. `eos_token`이 이 토큰으로 잡혀 있어 `generate`가 알아서 멈춥니다.
- system 프롬프트는 형식상 넣을 수 있지만, 학습 데이터 대부분이 system 없이 구성돼 효과는 제한적입니다.
- 문맥 길이는 2,048토큰입니다.
- KV 캐시를 구현하지 않아 생성할 때마다 전체 시퀀스를 다시 계산합니다. 길게 생성하면 느립니다.
- 패딩 마스크가 없습니다. `attention_mask`는 무시되므로, 길이가 다른 대화를 패딩해서 한 배치로 생성하면 결과가 틀어집니다. 한 건씩 넣으세요.

## 모델

| 항목 | 값 |
|---|---|
| 파라미터 | 1,107M |
| 구조 | decoder-only, pre-RMSNorm, RoPE, 입출력 임베딩 공유 |
| 층 / 차원 / 헤드 | 20 / 2048 / 16 |
| FFN 차원 | 8192 |
| 문맥 길이 | 2,048 |
| 어휘 | 49,154 (BPE 49,152 + `<|im_start|>`, `<|im_end|>`) |
| 가중치 형식 | float32 safetensors |

구조는 베이스와 같습니다. 일반 트랜스포머와 다른 점은 두 가지입니다.

- **Differential Attention.** 어텐션 맵을 두 개 만들어 하나에서 다른 하나를 뺍니다. 양쪽에 공통으로 끼는 잡음을 상쇄하려는 아이디어입니다.
- **PolyNorm.** 활성 함수 자리에 x, x², x³을 각각 정규화해서 학습되는 가중치로 섞어 씁니다.

둘 다 제가 고안한 게 아니고 Motif-2.6B 기술보고서([arXiv:2508.09148](https://arxiv.org/abs/2508.09148))와 Differential Transformer(Ye et al., 2024)를 읽고 직접 구현해 본 것입니다. 원 저자들과는 관계없는 개인 구현이라, 틀린 부분이 있다면 제 실수입니다.

## 학습

베이스 모델의 사전학습(30B 토큰)은 [Perdix-1.1B-Base](https://huggingface.co/prismdata/Perdix-1.1B-Base) 카드에 적었습니다. 그 위에 SFT를 했습니다.

- **데이터**: 약 390K 대화, 2 epoch
  - 영어: [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)의 everyday-conversations(3회 반복), smol-magpie-ultra, smol-constraints, smol-rewrite, smol-summarize, metamathqa-50k, numina-cot-100k 일부
  - 한국어: [kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2), [open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions)의 koalpaca, OIG-smallchip2-ko, korquad-chat 일부
  - 각 데이터셋의 라이선스와 출처는 원 저장소를 따릅니다.
- **형식**: ChatML. 토크나이저에 `<|im_start|>`, `<|im_end|>`를 추가하고 새 임베딩은 기존 임베딩 평균으로 초기화했습니다. loss는 답변 구간에만 걸었습니다.
- **학습률**: 최고 5e-5, 100스텝 워밍업 뒤 cosine으로 최고값의 10%까지
- **배치**: 스텝당 약 65K 토큰, 길이 구간별 패딩, bf16, 6,984스텝
- **결과**: 검증 loss 1.37 → 1.13
- **장비**: DGX Spark 한 대, 약 21시간

학습 코드는 [github.com/theprismdata/Perdix](https://github.com/theprismdata/Perdix)에 있습니다.

## 라이선스

모델 가중치와 코드는 Apache-2.0. 학습에 쓴 대화 데이터는 각각의 라이선스를 따르며 출처는 위에 적었습니다.
