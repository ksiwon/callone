# 설치 시간 줄이기 — 어디가 오래 걸리고, 어떻게 없애나

실측 기준 설치 시간의 지배 요인(감사 2026-07-11):

| 요인 | 시간 | 현재 상태 |
|---|---|---|
| LLM GGUF 다운로드 (7.8B Q6_K ~6.5GB / 32B ~26GB) | 수 분~수십 분 | `HF_HUB_ENABLE_HF_TRANSFER` 적용됨(bootstrap) + 존재 시 스킵 |
| CosyVoice 셋업 (conda env + 서브모듈 + torch 재설치 + 모델 9.75GB) | **최대 병목, ~20-40분** | 단계별 멱등(재실행 시 스킵) |
| llama.cpp CUDA 컴파일 (40~60분) | 프리빌트 tgz 로 **기본 회피**(드라이버 불일치 시만 재빌드) | 적용됨 |
| Ditto TRT 엔진 빌드 (비-Ampere 카드) | 수십 분 | Ampere(3090/Ti)는 해당 없음, 산출물 존재 시 스킵 |
| pip 대형 휠(torch 등) x venv 4개 | 수 분씩 | venv 존재 시 스킵 |

## 핵심 전략: "두 번째 부팅부터 0원" — RunPod Network Volume

스크립트 전부가 **존재 시 스킵(멱등)** 설계라, 산출물이 살아남으면 재설치가 통째로 사라진다.

1. RunPod 에서 **Network Volume** 생성 → 포드 만들 때 `/workspace` 에 마운트.
2. repo 를 `/workspace/callone` 에 클론(→ `.venv-*` 들도 볼륨에 남음). `CALLONE_HOME=/workspace` 는 자동 감지.
3. 최초 1회만 풀 설치. **이후 포드는 생성→`run_all.sh` 까지 ~2-5분**(모델·venv·바이너리 전부 재사용).
   - 포드 삭제/교체 자유. GPU 종류 바꿔도 llama 프리빌트가 안 맞으면 자동 재빌드만 발생.

## 보조 전략

- **포드 템플릿 프리베이크**: 다 세팅된 포드를 RunPod "Save as Template"(이미지化) → 새 포드가 부팅부터 완제품. 볼륨 방식보다 GPU 종류 바뀔 때 취약(TRT/컴파일 산출물이 카드 종속)하니 볼륨 방식 우선.
- **uv 사용(선택)**: `pip install uv` 후 setup 스크립트의 pip 를 `uv pip` 로 — 해석/다운로드 5~10배. 최초 설치에만 효과.
- **필요한 것만 설치**: 음성만 쓸 땐 `SKIP_AVATAR=1`(install.sh), Qwen A/B 안 할 땐 setup_qwen_tts 생략 — 각 venv 는 독립이라 안 깔면 그만큼 단축(폴백이 알아서 동작).

## "튜닝 시간"에 대해

- **목소리: 튜닝 자체가 없다.** 제로샷 클론(5~10초 클립)이 정본 — Piper 학습 경로(수 시간)는 폐기됨.
- **말투(LoRA, 선택)**: `callone-llm-train`(trl). 오래 걸리면 ① epochs 3→1-2 ② QLoRA(`load_in_4bit: true`) ③ unsloth 도입(2~3배) 순으로. 단 **캐릭터 카드 프롬프트(기본 경로)만으로 말투 재현이 충분한 경우가 많아**, LoRA 는 A/B 로 이득 확인 후에만 돌릴 것.
- **기억(RAG): 학습이 아님** — 전사→`extract_memories.py`(1회, 분 단위)→`use_rag: auto` 로 즉시 반영.
