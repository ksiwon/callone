# callone 사용 가이드

상황에 맞는 문서를 보면 된다.

| 하고 싶은 것 | 문서 |
|---|---|
| 노트북에서 녹음 파일 처리(데이터셋·방언프로필) | [1. 로컬에서 학습](1_로컬에서_학습.md) |
| GPU 서버에서 1000+개로 화자분리·전사 + LoRA(말투) 학습 | [2. GPU에서 학습](2_GPU에서_학습.md) |
| 노트북에서 clone과 **실시간 통화** (Qwen3.5+LoRA, llama.cpp, Piper) | [3. 노트북에서 통화](3_노트북에서_통화.md) |
| 휴대폰에서 통화 | [4. 휴대폰에서 통화](4_휴대폰에서_통화.md) |
| 그 사람 **목소리** 학습 (Piper TTS, GPU) | [5. 화자 A 목소리 학습](5_화자A_목소리_학습.md) |

---

## 새 데이터셋으로 처음부터 (재현 순서)

녹음(m4a, 두 화자 혼합) 한 묶음으로 시작 → clone 통화까지. `A`=상대, `B`=본인.

1. **데이터** — m4a 전부 `data/raw/` 에. → [2. GPU에서 학습](2_GPU에서_학습.md) A·B 단계.
   `setup_server.sh full` 이 화자분리·전사·방언프로필·TTS셋·대화셋·페르소나카드 생성
   → 산출물: `data/speakers/{A,B}/`, `data/datasets/{A,B}/`
2. **말투(LLM LoRA)** — GPU에서 `callone-llm-train` (Qwen3.5-4B QLoRA)
   → 산출물: `models/llm_phone/{A,B}/checkpoint-*/` (LoRA 어댑터)
3. **목소리(TTS)** — [5. 화자 A 목소리 학습](5_화자A_목소리_학습.md): `prep_piper.py` → GPU Piper 학습 → `A.onnx`
4. **노트북 배포·통화** — [3. 노트북에서 통화](3_노트북에서_통화.md):
   `make_gguf.py`(LoRA→GGUF) → llama-server → `models/tts_piper/A.onnx` 배치 → `call_mic.py`

노트북으로 가져올 산출물: `models/llm_phone/*/checkpoint-*`, `data/speakers/*`,
`models/tts_piper/*.onnx`. (중간물 base/merged/f16 는 노트북에서 재생성 후 삭제 가능.)

> 기술 명세·모델 결정 근거는 최상위 [`README.md`](../README.md), 원 스펙은 [`callone_spec.md`](../callone_spec.md).
