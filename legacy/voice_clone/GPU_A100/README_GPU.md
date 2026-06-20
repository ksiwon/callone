# GPU 버전 (A100 / Elice) — 처음부터

## 전제
- 리눅스 + NVIDIA GPU(A100 등) + CUDA + conda 가 있는 인스턴스 (Elice가 여기에 해당).
- 인터넷 연결(클론·패키지·모델 다운로드용).

## 1. 이 폴더를 인스턴스에 올리기
Elice 파일 업로드 기능이나 git/scp 로 `GPU_A100` 폴더(또는 이 zip)를 인스턴스에 올립니다.
업로드한 폴더로 이동:
```bash
cd GPU_A100
```

## 2. 설치 (한 번, 수십 분)
```bash
bash setup.sh
```
- CosyVoice 클론 → 환경 → 의존성 → **9.75GB 모델 다운로드** → 앱 배치까지 자동입니다.
- (선택) HF 다운로드가 느리면 토큰을 먼저 주면 빨라집니다:
  `export HF_TOKEN='hf_...'` 후 `bash setup.sh`.

## 3. 실행
```bash
cd ~/CosyVoice
bash run.sh
```
콘솔에 `Running on local URL: http://0.0.0.0:50000` 이 뜨면 됩니다.

## 4. 접속
- 인스턴스에 직접 접속 가능한 포트(50000)가 열려 있으면 브라우저에서 그 주소로 접속.
- 포트가 막혀 있으면(흔함), `app_studio.py` 맨 아래의
  `demo.launch(server_name="0.0.0.0", server_port=port, share=False)` 에서
  `share=False` → `share=True` 로 바꾸세요. 임시 공개 링크(`https://....gradio.live`)가 생깁니다.
  (Elice가 특정 포트 매핑/프록시를 제공하면 그쪽을 쓰는 게 더 좋습니다.)

## 5. 사용
- ① 만들기: WAV 업로드 → 🔎 자동 전사(틀린 글자 수정) → 🧪 후보 생성 →
  가장 본인 같은 후보 선택 → 이름 적고 💾 저장.
- ② 사용하기: 저장한 목소리 선택 → 텍스트 입력 → ▶️ 합성 → 결과 ⤓ 다운로드.

## 메모
- GPU에서는 dtype(bf16) 이 그대로 동작하므로 CPU 버전과 달리 config 수정이 없습니다.
- 토크나이저 onnx 는 CPU onnxruntime 으로 돌립니다(작아서 무관, CUDA 버전 충돌 회피).
- 더 빠르게: `app_studio.py` 의 `AutoModel(..., fp16=False)` 를 `fp16=True` 로 바꾸면 빨라지지만,
  유사도 우선이면 그대로 두세요.
