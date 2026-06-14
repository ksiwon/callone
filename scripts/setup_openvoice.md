# OpenVoice V2 + MeloTTS 한국어 음성복제 — 설치 + 품질검증

> 목표: MeloTTS(한국어 네이티브) + OpenVoice(톤컬러 변환=A 음색) → 빠른 로컬 한국어 복제.
> 1단계 torch-CPU 로 **품질 확인** → 좋으면 2단계 OpenVINO 로 Arc 가속.
> 학습 불필요(제로샷, A 레퍼런스 클립만).

## 1) 설치 (Anaconda Prompt 권장 — MeloTTS 한국어 의존성 때문)
```bat
conda create -n openvoice -y python=3.10
conda activate openvoice
:: OpenVoice
git clone https://github.com/myshell-ai/OpenVoice C:\tools\OpenVoice
cd C:\tools\OpenVoice
pip install -e .
:: MeloTTS (한국어 g2p 포함)
pip install git+https://github.com/myshell-ai/MeloTTS.git
pip install g2pkk
python -m unidic download
```

## 2) OpenVoice V2 체크포인트 다운로드
```bat
:: checkpoints_v2 (톤컬러 변환기 + 화자 SE)
curl -L -o cv2.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
tar -xf cv2.zip
:: → C:\tools\OpenVoice\checkpoints_v2\ (converter/, base_speakers/ses/kr.pth 등)
dir checkpoints_v2\base_speakers\ses
```

## 3) 품질 검증 (A 목소리로 한국어, torch-CPU)
```bat
copy C:\Users\pjo12\Downloads\coding\callone\scripts\openvoice_clone_test.py .
python openvoice_clone_test.py ^
  --ref "C:\Users\pjo12\Downloads\coding\callone\cosyvoice_ref\A_ref1.wav" ^
  --text "내 왔다 아이가 밥은 묵었나" --ckpt checkpoints_v2
```
→ `A_clone.wav` + 실시간배율 출력. **들어보고 판단**:
- ① 진짜 한국어? (외계어/외국인억양 아님 — MeloTTS 네이티브라 OK일 것)
- ② A 음색 닮았나? (OpenVoice 톤컬러 변환)
- ③ 속도 (feed-forward라 CosyVoice보다 훨 빠를 것)

## 4) (품질 OK면) OpenVINO 로 Arc 가속
[OpenVINO 공식 노트북](https://docs.openvino.ai/2024/notebooks/openvoice-with-output.html) 로 MeloTTS+OpenVoice 를 IR 변환 → `device="GPU"`(Arc) 또는 `"NPU"`. CPU보다 빠름. (이 단계는 품질 확인 후.)

---

## 흔한 막힘
- MeloTTS 한국어 `g2pkk`/`mecab` 에러 → `pip install g2pkk eunjeon` 추가 시도. 안 되면 에러 붙여줘.
- `pip install -e .`(OpenVoice) 의존성 충돌 → 보통 무시 가능, 안 되면 개별 설치.
- 체크포인트 zip 안 받아지면 → HF `myshell-ai/OpenVoiceV2` 에서 받기.

품질만 확인되면(③ 셋 다 OK) → OpenVINO/Arc 가속 + callone 통합. 안 되는 단계 에러 붙여줘.
