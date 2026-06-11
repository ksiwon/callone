# 화자 A 음색 Piper TTS 학습 → 노트북 실시간 (onnx)

목표: 화자 A TTS 데이터셋(73.8분)으로 **화자 A 목소리** Piper 음성 모델 학습 → onnx export →
노트북에서 `serve/tts_piper.py` 가 CPU 실시간으로 화자 A 목소리 합성.

> ⚠️ **학습엔 CUDA GPU 필요.** Piper(VITS)는 노트북 Arc/CPU로는 비현실적(며칠).
> LLM 학습 때 쓴 Elice 같은 GPU 인스턴스 다시 띄워서 거기서 학습 → onnx 만 노트북으로 복사.
> **추론(서빙)은 노트북 CPU로 실시간** = onnx + piper-phonemize, torch 불필요.

---

## 0) (노트북) 데이터셋 변환 — GPU 불필요, 먼저 해둠
```powershell
pip install soundfile soxr scipy
python scripts\prep_piper.py --speaker A          # → piper_ds\A\{wav, metadata.csv}
```
결과: `piper_ds\A\metadata.csv`(`id|text`) + `piper_ds\A\wav\*.wav`(22050Hz). 이 폴더를 GPU 서버로 업로드.

---

## 1) (GPU 서버) 학습 환경
```bash
python -m venv .venv && source .venv/bin/activate
pip install piper-tts                      # 추론용
# 학습 코드(rhasspy/piper): 학습 모듈 + 의존성
git clone https://github.com/rhasspy/piper
cd piper/src/python
pip install -e .
pip install -r requirements_train.txt
sudo apt-get install -y espeak-ng          # 한국어 음소(ko)
bash build_monotonic_align.sh              # MAS 빌드
```

## 2) preprocess (음소화: espeak-ng 한국어 'ko')
```bash
python -m piper_train.preprocess \
  --language ko \
  --input-dir /path/piper_ds/A \
  --output-dir /path/piper_train/A \
  --dataset-format ljspeech --single-speaker \
  --sample-rate 22050
```

## 3) train
**파인튜닝(권장, 빠름·음질↑)** — medium 체크포인트(타 언어라도 음향디코더 전이됨)에서 시작:
```bash
# 예: lessac-medium .ckpt 받아서 --resume_from_checkpoint 로 지정
python -m piper_train \
  --dataset-dir /path/piper_train/A \
  --accelerator gpu --devices 1 --batch-size 24 --precision 16 \
  --quality medium \
  --resume_from_checkpoint /path/base-medium.ckpt \
  --max_epochs 3000 --checkpoint-epochs 50 \
  --validation-split 0.0 --num-test-examples 0
```
- 73분 데이터 파인튜닝: base 기준 **+~1000 epoch** 면 쓸만, 더 돌리면 더 좋음.
- 스크래치(base 없이): `--resume_from_checkpoint` 빼고 `--max_epochs 6000`(오래 걸림).
- 중간 `lightning_logs/version_0/checkpoints/*.ckpt` 로 진행 확인.

## 4) export onnx
```bash
python -m piper_train.export_onnx \
  lightning_logs/version_0/checkpoints/last.ckpt \
  A.onnx
cp /path/piper_train/A/config.json A.onnx.json
```

## 5) (노트북) 배치 + 서빙
```powershell
pip install piper-tts                       # 추론(onnxruntime + 음소화)
mkdir models\tts_piper
# 서버에서 받은 두 파일을 여기로:
#   models\tts_piper\A.onnx
#   models\tts_piper\A.onnx.json
```
`serve/tts_piper.py` 가 `models/tts_piper/A.onnx` 를 자동 인식(orchestrator `_pick_tts` 1순위).
없으면 Kokoro→placeholder 로 폴백하니, 학습 끝나기 전에도 통화 파이프라인은 돌아간다.

---

## 빠른 점검(노트북, 학습본 받은 뒤)
```powershell
python -c "from callone.serve.tts_piper import PiperTTS; import soundfile as sf; t=PiperTTS('A'); y,sr=t.synth('화자 A 밥 먹었나'); sf.write('test_mom.wav', y, sr); print('OK', len(y)/sr, '초')"
```
`test_mom.wav` 들어보면 화자 A 목소리. B 화자(화자 B/본인)도 `--speaker B` 로 동일.
