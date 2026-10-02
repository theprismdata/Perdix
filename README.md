# Perdix

밑바닥부터 사전학습해 보는 작은 언어 모델(SLM)입니다.

> 페르딕스(Perdix)는 명장 다이달로스의 어린 제자입니다. 톱과 컴퍼스를 발명했지만,
> 탑에서 떨어지다 자고새가 되어 그 뒤로는 낮게만 납니다.
> 이 모델도 아직은 낮게 납니다.

## 현재 상태

- **베이스 모델**입니다. 이어쓰기만 하고, 대화·지시 수행·툴콜은 학습하지 않았습니다.
- 약 1.1B 파라미터 (dim 2048, 20층, 16헤드, FFN 8192, 문맥 2048 토큰, vocab 49,152)
- 30B 토큰 사전학습: 영어 웹(DCLM-baseline), 한국어 웹(FineWeb2 kor), 수학(FineMath 4+)
- 가중치는 이 저장소에 포함하지 않습니다.

## 구조

decoder-only, pre-RMSNorm, RoPE, tied embedding 위에 두 가지를 구현했습니다.

- **Differential Attention**: `[softmax(Q1K1ᵀ) − λ·softmax(Q2K2ᵀ)] V`
- **PolyNorm**: x, x², x³를 각각 RMS 정규화한 뒤 학습 가중치로 합성하는 활성 함수

## 파일

| 파일 | 내용 |
|---|---|
| `model.py` | `PerdixConfig`, `PerdixSLM` |
| `tokenize_pack.py` | parquet → uint16 토큰 바이너리(`packed/`) |
| `train.py` | 사전학습 (선형 데이터 믹싱, WSD 학습률, bf16, 재개 지원) |
| `test_slm_infer.py`, `serve_slm.py` | 체크포인트 이어쓰기 테스트 / 임시 서빙 |
| `inspect_model.py`, `view_data.py` | 모델 파일 구조 분석 / 학습 데이터 뷰어 |
| `serve_hf.py`, `serve_hf_tools.py` | HF 모델용 OpenAI 호환 서버 (`MODEL_PATH` 환경변수로 지정). `_tools`는 툴콜 지원 |
| `test_hf_infer.py`, `test_hf_toolcall.py`, `test_api_toolcall.py` | HF 모델 추론 / 툴콜 / API 왕복 테스트 |

## 사용

```bash
python3 tokenize_pack.py            # 데이터 토큰화
python3 train.py --smoke            # 초소형 스모크 테스트
python3 train.py --compile          # 본 학습
python3 train.py --resume ckpt/latest.pt
```

## 참고한 연구

- Motif-2.6B 기술보고서 ([arXiv:2508.09148](https://arxiv.org/abs/2508.09148)): Differential Attention + PolyNorm 구조, 데이터 믹싱과 학습률 감쇠 방식. Perdix는 Motif Technologies와 무관한 독립 구현입니다.
- Ye et al. 2024, Differential Transformer: λ 재파라미터화

## 라이선스

Apache-2.0
