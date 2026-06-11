# 2. GPU 서버(엘리스 H100)에서 학습하기

> 1000+개 녹음으로 **진짜** 화자분리·전사 + **목소리/말투 복제 학습**을 한다.

## A. 로컬에서 업로드 (Windows)
```powershell
# 녹음 m4a 전부 data\raw\ 에 넣은 뒤, 한 줄:
.\scripts\upload_to_elice.ps1 -Pem "$env:USERPROFILE\.ssh\elice-xxxx.pem" -Port [포트]
```
- `.pem`(개인키)·`[포트]`는 엘리스 Run Box "연결 정보"에서 확인.
- 불필요한 것 빼고 `tar`로 묶어 자동 업로드. 끝나면 다음 명령을 출력해줌.

## B. 서버에서 설치 + 데이터 처리 (한 줄)
```bash
# 엘리스 접속:
ssh -i ~/.ssh/elice-xxxx.pem elicer@central-01.tcp.tunnel.elice.io -p [포트]

# 압축 풀고 설치 + 전량 파이프라인:
tar -xf ~/callone.tar && cd callone && bash scripts/setup_server.sh full
```
`setup_server.sh`가 알아서: venv → torch(CUDA) → 의존성 → `DEVICE=cuda` →
1000+개 분리·전사·데이터셋 생성(A·B).
(`full` 대신 `pilot` = 50통만 먼저 확인. 인자 없으면 설치만.)

> HF 토큰: `.env`에 같이 따라온다. 없으면 서버에서 `nano .env`로 넣고,
> pyannote `community-1`/`segmentation-3.0` 게이트 동의(무료).

## C. 모델 학습 (핵심)
데이터셋이 만들어지면:
```bash
# (1) ASR 방언 적응 — 정확도 ↑ (권장)
callone-correct --hours 3
#   → data/datasets/asr_correction/to_correct.csv 의 corrected_text 칸을 사람이 교정 후:
callone-asr-train

# (2) 목소리 복제
callone-tts-train  --speakers A B      # 서버 고품질 (Qwen3-TTS / VoxCPM2)
callone-tts-phone  --speakers A B      # 폰용 경량 (Piper / MeloTTS)

# (3) 말투·성격 복제
callone-llm-train  --config llm_server --speakers A B   # Gemma 4 12B
callone-llm-train  --config llm_phone  --speakers A B   # 폰용 Gemma 4 E4B
```

## 결과물 (학습된 모델)
- `models/asr_dialect/` — 방언 적응 ASR
- `models/tts_server/{A,B}/`, `models/tts_phone/{A,B}/` — 목소리
- `models/llm_12b/{A,B}/`, `models/llm_e4b/{A,B}/` — 말투

이 `models/` 폴더를 노트북/폰으로 가져가면 통화 가능 → [3번](3_노트북에서_통화.md) · [4번](4_휴대폰에서_통화.md).

## 통화까지 서버에서 바로
```bash
callone-serve        # GPU 서버에서 실시간 통화 서버 (Gemma 12B + Qwen3-TTS)
```
