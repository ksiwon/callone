# 영상통화(토킹헤드) — Ditto 구조·문제해결 노트

> **설치/실행은 [FRESH_SETUP.md](FRESH_SETUP.md) 가 정본**(setup_avatar_gpu.sh 가 Ditto+TensorRT+의존성 자동).
> 이 문서는 구조·디버깅 참고용. 우선순위: 목소리 > 한국어 > **얼굴** > 속도.
> 안전철학: Ditto 안 떠도 **정지사진 폴백** → 음성 통화는 항상 정상.

전체 그림(별 프로세스 3개, llama-server 패턴):
```
.venv-serve  callone-serve(:8000, WS) ──오디오──▶ 스피커/UI
                     │  └─사진,오디오청크(HTTP/WS)─▶ avatar-server(:8091, .venv-avatar) ─JPEG프레임─▶ UI
                     └─프롬프트(HTTP)─▶ llama-server(:8090)
```

---

## 1. 음성 스택 (이미 됨)
```bash
PORT=8090 bash scripts/bootstrap_gpu.sh        # venv·llama-server·모델 (FRESH_SETUP.md 참고)
# 화자 ref(목소리) + 증명사진 둘 다 data/speakers/<화자>/ 에:
#   ref_24k.wav + ref_text.txt  (목소리)   /   portrait.jpg|png  (얼굴)
```
> 증명사진: 정면·얼굴 또렷·정상 조명(선글라스·가림 X). 얼굴 검출·정렬은 Ditto 내부가 처리.

## 2. avatar-server 설치 (별 venv, GPU)
```bash
bash scripts/setup_avatar_gpu.sh               # .venv-avatar + Ditto repo+checkpoints(PyTorch) + 드라이버호환 torch
```
✅ 끝에 `torch ... cuda ... OK` + DITTO_DATA_ROOT/CFG_PKL 경로 출력.
> **얼굴 움직임 = TRT 경로.** Ampere(A100/3090/3090Ti)는 프리빌트 엔진(검증됨). 4090(Ada)은 setup 이
> `cvt_onnx_to_trt` 로 **custom TRT 엔진 자동 빌드**(~수십분) → 성공 시 움직임. 빌드 깨지면(TRT↔드라이버)
> PyTorch는 0프레임이라 `AVATAR_BACKEND=static`(음성+정지사진)으로. 확실한 건 Ampere GPU.

## 3. avatar-server 기동 (별 터미널, 계속 켜둠)
```bash
source .venv-avatar/bin/activate
AVATAR_BACKEND=ditto python -m avatar_server --port 8091 &
sleep 5; curl -s http://127.0.0.1:8091/health ; echo      # {"status":"ok","backend":"ditto"}
```
- `backend":"static"` 으로 뜨면 → Ditto 로드 실패(폴백). 로그 보고 [V] 검증지점(DittoModel) 맞춰라. 음성은 정상.
- ⚠️ **첫 검증은 static 으로**: `AVATAR_BACKEND=static python -m avatar_server` → 정지사진이라도 화면 뜨는지(파이프라인) 먼저 확인 → 그담 ditto.

## 4. callone 에서 avatar 켜기
`configs/serve.yaml`:
```yaml
avatar:
  enabled: true                 # ← false→true
  backend: auto                 # auto = avatar-server 있으면 Ditto, 없으면 static 폴백
  base_url: "http://127.0.0.1:8091"
  # image 비우면 data/speakers/<화자>/portrait.jpg|png 자동탐색
  fps: 25
  resolution: 256
```
```bash
sed -i 's/^  enabled: false/  enabled: true/' configs/serve.yaml      # avatar 블록(주의: 다른 enabled 없으면)
```

## 5. 실시간 영상통화 (callone-serve + UI)
스튜디오(턴제) 말고 **실시간 WS** 경로 = 진짜 통화 느낌(음성 도착하는대로 + barge-in + 얼굴):
```bash
# 터미널 A — 백엔드(WS)
source .venv-serve/bin/activate && callone-serve            # :8000

# 터미널 B — UI
cd ui && npm install && npm run dev                         # :5173 (vite 프록시로 :8000)
```
→ 브라우저 `localhost:5173`(원격이면 포트 노출/터널) → 화자 선택 → 통화.
**CallScreen 이 음성+얼굴 프레임을 같이 재생**(avatar 꺼져 있으면 파형만). 말끝 버튼/말하면 barge-in.

---

## 체크 순서 (싼것 먼저, 비싼 디버깅 최소)
1. avatar-server **static** /health → 파이프라인 OK?
2. static 으로 callone-serve+UI 통화 → 정지사진이 화면에 뜨고 음성 정상?
3. avatar-server **ditto** /health backend=ditto?
4. ditto 로 통화 → 입·표정·고개 움직이나? FFD(첫프레임)·fps·VRAM 로그 확인.
5. 동시구동(LLM+TTS+ASR+Ditto) 시 첫음성 지연 변화 측정.

## 막힐 때
| 증상 | 처리 |
|---|---|
| /health backend=static (ditto 의도했는데) | DITTO_* env 확인, Ditto import/checkpoint 실패 로그. PyTorch checkpoint 경로 맞나 |
| 얼굴 안 움직이고 정지 | static 폴백 중. ditto 백엔드+env 확인 |
| 프레임 안 옴(음성만) | serve.yaml avatar.enabled=true? portrait 있나? avatar-server 떴나(8091)? |
| torch GPU 실패(avatar venv) | 드라이버보다 최신 CUDA — 서빙과 동일 이슈. torch 를 드라이버 호환 버전으로 |
| Ditto frames [V] 안 맞음 | DittoModel.frames 의 setup_Nd/chunksize/writer attr 를 repo stream_pipeline_online.py 로 정합 |
| VRAM 부족(4090 24GB) | resolution 256 유지, LLM 양자화 유지. 최후 avatar.enabled=false(음성만) |
