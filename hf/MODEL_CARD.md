---
license: apache-2.0
language:
- ko
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- custom_code
- base-model
- differential-attention
- polynorm
datasets:
- mlfoundations/dclm-baseline-1.0
- HuggingFaceFW/fineweb-2
- HuggingFaceTB/finemath
---

# Perdix-1.1B-Base

작은 수준의 LLM으로 이름은 그리스 신화의 페르딕스에서 따왔습니다. 다이달로스의 어린 제자였고 톱과 컴퍼스를 발명했는데, 탑에서 떨어지다 자고새가 되는 바람에 그 뒤로는 낮게만 날아다닌다고 합니다. 지금 이 모델 수준이 딱 그렇습니다.

A 1.1B-parameter Korean/English base language model pretrained from scratch on a single machine. It only continues text; it has not been trained to chat or follow instructions.

## 무엇을 할 수 있고 무엇을 못 하나

**베이스 모델**입니다. 문장 앞부분을 주면 뒤를 잇는 것만 합니다. 질문에 답하거나 지시를 따르도록 학습시킨 적이 없어서, 대화용으로 쓰면 엉뚱한 글이 나옵니다.

이어 쓴 글은 문장으로는 그럴듯하지만 **내용은 자주 틀립니다.** 아래는 실제 출력입니다(temperature 0.8, top-k 50). 굵은 부분이 입력입니다.

> **대한민국의 수도 서울은** 세계 4대 문명 발상지이자 최대 도시이자 민주주의의 중심지로 알려져 있다. 그만큼 서울의 역사도 오래되었는데, 1905년 1월 4일 일본 제국이 서울에 신사·불각과 함께 …

> **The capital city of France is** Cannes, a city of the seas. And now, in Cannes, in this particular place, I'm going to speak about the culture of the city. …

> **김치는 한국의 전통 음식으로**, 다양한 재료를 사용해 김치를 담그는 문화와 식문화를 경험해 볼 수 있습니다. Q: 김치는 어떤 재료로 만들어지나요? A: 김치는 다양한 재료를 사용하여 만들어집니다. 주로 고춧가루, 마늘, 생강 등이 사용되며 …

사실 확인이 필요한 용도에는 쓰면 안 됩니다. 학습 데이터가 웹 문서라 편향되거나 부적절한 내용이 나올 수 있습니다.

## 사용법

모델 코드가 저장소에 들어 있어서 `trust_remote_code=True`가 필요합니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "prismdata/Perdix-1.1B-Base"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

ids = tok("대한민국의 수도 서울은", return_tensors="pt").input_ids.to("cuda")
out = model.generate(ids, max_new_tokens=100, do_sample=True, temperature=0.8, top_k=50)
print(tok.decode(out[0], skip_special_tokens=True))
```

알아 둘 점이 있습니다.

- 문맥 길이는 2,048토큰입니다.
- KV 캐시를 구현하지 않아 생성할 때마다 전체 시퀀스를 다시 계산합니다. 길게 생성하면 느립니다.
- 패딩 마스크가 없습니다. `attention_mask`는 무시되므로, 길이가 다른 문장을 패딩해서 한 배치로 생성하면 결과가 틀어집니다. 한 문장씩 넣으세요.

## 모델

| 항목 | 값 |
|---|---|
| 파라미터 | 1,107M |
| 구조 | decoder-only, pre-RMSNorm, RoPE, 입출력 임베딩 공유 |
| 층 / 차원 / 헤드 | 20 / 2048 / 16 |
| FFN 차원 | 8192 |
| 문맥 길이 | 2,048 |
| 어휘 | 49,152 (BPE) |
| 가중치 형식 | float32 safetensors |

일반 트랜스포머와 다른 점은 두 가지입니다.

- **Differential Attention.** 어텐션 맵을 두 개 만들어 하나에서 다른 하나를 뺍니다. 양쪽에 공통으로 끼는 잡음을 상쇄하려는 아이디어입니다.
- **PolyNorm.** 활성 함수 자리에 x, x², x³을 각각 정규화해서 학습되는 가중치로 섞어 씁니다.

둘 다 제가 고안한 게 아니고 Motif-2.6B 기술보고서([arXiv:2508.09148](https://arxiv.org/abs/2508.09148))와 Differential Transformer(Ye et al., 2024)를 읽고 직접 구현해 본 것입니다. 원 저자들과는 관계없는 개인 구현이라, 틀린 부분이 있다면 제 실수입니다.

## 학습

- **데이터**: 30B 토큰. 영어 웹([DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), CC-BY-4.0), 한국어 웹([FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) kor_Hang, ODC-By), 수학([FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) 4+, ODC-By)
- **믹싱**: 영어 65% / 한국어 30% / 수학 5%에서 시작해 25% / 50% / 25%로 서서히 바꿨습니다.
- **학습률**: 최고 3e-4. 1B 토큰 워밍업 뒤 유지하다 마지막 20% 구간에서 최고값의 25%까지 내렸습니다.
- **배치**: 스텝당 약 1M 토큰, 시퀀스 길이 2,048, bf16
- **결과**: loss 10.7 → 2.3 근처
- **장비**: DGX Spark 한 대

학습 코드는 [github.com/theprismdata/Perdix](https://github.com/theprismdata/Perdix)에 있습니다.

## 라이선스

Apache-2.0. 학습 데이터의 출처와 라이선스는 위에 적었습니다.
