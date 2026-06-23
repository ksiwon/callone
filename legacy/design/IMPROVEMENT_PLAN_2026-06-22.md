# callone 개선 설계안 — 한국어 품질 · TTS속도 · 영상속도 (2026-06-22)

> 상태 = **코드 구현 완료(2026-06-22), GPU 박스 검증 대기.** 사용자 요청: 한국어 답변 어색·맥락약함·
> 반복·되묻기 개선 + TTS/영상 생성 속도 개선. 시행착오 로그(`legacy/design/`, `docs/`) 반영.
> 우선순위 원칙 불변: **①목소리 ②한국어 ③얼굴 ④속도.**

## 구현 상태 (2026-06-22, pytest 53 pass · ui tsc EXIT=0)
- **작업2(한국어, 무해):** `llama_llm._system()` 충돌 규칙 통합 + `_payload()` DRY/min_p, `serve.yaml` 노브. **temp 0.4 유지**(검토: 0.6 미검증·비문위험). ✅ 적용(기본 동작 변경 = 반복억제만).
- **작업1(TTS 스트리밍):** `cosyvoice_server` `/synth_stream` + `tts_cosyvoice` 토글. **`tts.stream: false` 기본**(A/B 전 음색 불변). ✅ 코드, ⏸ 라이브는 박스 A/B 후.
- **작업4-A(아바타 워밍업):** `orchestrator._warmup_avatar()` — 세션 시작 단발(라이프사이클 불변). ✅ 적용.
- **작업3(모델):** A100/H100 기본을 **EXAONE-4.0-32B-abliterated Q6_K로 확정**. 24GB GPU는 VRAM 한계로 7.8B 유지. ✅
- **작업2-D(UI):** `CallScreen.tsx` 예시 캐릭터 프리셋 칩 4개(원클릭→example_dialogue 채움). ✅ 적용.
- 회귀 테스트: `tests/test_improve_round.py` 9개(샘플러·thinking 게이트·TTS 프레이밍·아바타 워밍업).
- **GPU 박스 전용(남음):** ① TTS `/synth_stream` 저장wav 음색 A/B→통과 시 `tts.stream:true` ② 32B 실통화 한국어·속도 확인 ③ `callone-bench` 첫음성 ms·아바타 첫프레임 ④ llama-server `dry_*`·`min_p` 지원 확인. (상세 = 각 작업 '검증' 절 + 맨 아래 체크리스트.)

## 시행착오에서 가져온 제약 (이 설계가 반드시 지킴)
- **한국어 모델:** Gemma4(약함)→Qwen3.5-9B(**한국어 깨짐**)→EXAONE 채택. → 교체 후보는 **한국어 특화**만(Qwen 계열 제외).
- **목소리:** Qwen3-TTS(음색 튐)→CosyVoice3. chunk_size 25→10 A/B까지 함. → TTS 변경은 **저장 wav A/B 통과 전 라이브 금지**. config 토글로 즉시 롤백 가능하게.
- **영상:** 발화당 `setup→run→close`는 OOM/thrashing 픽스 결과(커밋 f8d3261). persistent 세션 재도입 = 그 버그 재발. → **라이프사이클 안 건드림.**
- 이미 튜닝됨: temp/penalty/seed/warmup/max_tokens. → 안 해본 레버만 쓴다: **DRY 샘플러·min_p·CosyVoice 네이티브 stream=True·ref 임베딩 캐싱·아바타 워밍업.**

---

## 작업 1 — TTS 스트리밍 (CosyVoice 네이티브 `stream=True`)  ★효과 최대

### 문제
- `cosyvoice_server/app.py:92` 가 `inference_zero_shot(..., stream=False)` → **응답 전체 합성 완료 후** 통짜 PCM 반환.
- `synth_mode: full`([serve.yaml:66](../configs/serve.yaml#L66))이라 첫 음성 지연 = LLM전체 + 응답전체 TTS합성. chunk_size는 "받은 통짜를 재생 시 쪼개는 크기"라 합성 자체는 안 빨라짐.

### 핵심 사실 (오해 정정)
CosyVoice2/3 `stream=True`는 **문장 쪼개기가 아님** — 단일 생성 내부에서 토큰→오디오를 흘리는 bi-streaming. 운율/음색 연속 = 통짜와 동일, 첫 패키지 ~150ms. (과거 "톤 튐"은 Qwen 시절 문장 분할이지 네이티브 스트리밍이 아님.)

### 변경점
**A. `cosyvoice_server/app.py` — `/synth_stream` 신설 (기존 `/synth` 유지 = A/B·폴백)**
```python
# 신설 엔드포인트. 기존 synth() 는 그대로 둠(stream=False, A/B 기준선).
from fastapi.responses import StreamingResponse
import struct

@app.post("/synth_stream")
def synth_stream(req: SynthReq):
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)
    ref = np.frombuffer(base64.b64decode(req.ref_audio_b64), dtype=np.float32)
    ref16 = ref if req.ref_sr == 16000 else _resample(ref, req.ref_sr, 16000)
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    fd, ref_path = tempfile.mkstemp(suffix=".wav", prefix="cosy_ref_", dir=base); os.close(fd)
    sf.write(ref_path, ref16.astype(np.float32), 16000)

    def gen():
        try:
            if SEED >= 0:
                torch.manual_seed(SEED)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
            pt = (req.prompt_text or "").strip()
            if NEEDS_SYS:
                it = _cosy.inference_zero_shot(req.text.strip(), SYS_PROMPT + pt, ref_path, stream=True)
            elif pt:
                it = _cosy.inference_zero_shot(req.text.strip(), pt, ref_path, stream=True)
            else:
                it = _cosy.inference_cross_lingual(req.text.strip(), ref_path, stream=True)
            for j in it:                       # 청크가 나오는 대로
                a = j["tts_speech"].squeeze(0).detach().cpu().numpy().astype(np.float32)
                a = np.clip(_resample(a, SR, OUT_SR), -1.0, 1.0).astype(np.float32)
                b = a.tobytes()
                yield struct.pack("<I", len(b)) + b   # [4바이트 길이][f32 PCM] 프레이밍
        finally:
            try: os.remove(ref_path)
            except OSError: pass
    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(OUT_SR)})
```
> 프레이밍 이유: f32 청크 경계가 4바이트로 안 맞을 수 있어 길이 프리픽스로 명확히. 청크별 `_resample`은 경계 클릭 가능성 → 검증 항목(아래).

**B. `callone/serve/tts_cosyvoice.py` — stream 토글 + 프레임 파서**
```python
# __init__ 에 추가
self.stream = bool(scfg.get("stream", True))   # serve.yaml tts.stream

# synth_stream 교체: stream True 면 /synth_stream 을 프레임 단위로 읽어 즉시 yield
def synth_stream(self, text, chunk_ms=200, emotion=None):
    if not text.strip() or not self._ref_b64: return
    payload = json.dumps({...동일...}).encode()
    if not self.stream:
        yield from self._synth_whole(text)      # 기존 /synth 경로(통짜) = A/B 기준선
        return
    req = urllib.request.Request(f"{self.base_url}/synth_stream", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=self.timeout) as r:
        out_sr = int(r.headers.get("X-Sample-Rate", self.sr))
        while True:
            head = _read_exactly(r, 4)
            if not head: break
            (n,) = struct.unpack("<I", head)
            buf = _read_exactly(r, n)
            audio = np.frombuffer(buf, dtype=np.float32)
            if out_sr != self.sr: audio = self._resample(audio, out_sr, self.sr)
            yield audio                          # 첫 청크가 ~150ms 안에 도착
```
> `_read_exactly(resp, n)` = HTTP 본문에서 정확히 n바이트 모아 읽는 헬퍼(소켓 부분수신 대비). 기존 통짜 경로는 `_synth_whole` 로 이름만 옮겨 보존.

**C. `configs/serve.yaml` tts 블록**
```yaml
tts:
  stream: true        # ← 이미 있으나 cosyvoice 경로에서 미사용이었음. 이제 실제 토글.
                      #    false = 통짜 합성(현재 동작, A/B 기준선·즉시 롤백)
```

**D. (선택, 2단계) ref 임베딩 캐싱 — 턴당 전처리 제거**
- 지금은 매 합성마다 ref wav를 tmpfs에 쓰고 화자 임베딩 재추출([app.py:78-96](../cosyvoice_server/app.py#L78-L96)).
- CosyVoice `add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_id)` + `save_spkinfo()` 로 **통화 시작 1회 등록** → 턴마다 `zero_shot_spk_id`만 넘김. 턴당 전처리 0 + 음색 더 안정.
- 세션 개념 필요(서버가 현재 stateless) → `/session/register` + `/synth_stream(spk_id)`. 1단계(A~C) 검증 뒤 착수.

### 검증 (목소리 우선 — 라이브 전 필수)
1. **저장 wav A/B:** 같은 텍스트·ref 로 `/synth`(통짜) vs `/synth_stream` 합성본 저장 → 파형/스펙트로그램 + 귀로 음색 동일 확인. **다르면 라이브 금지.**
2. 청크 경계 클릭/팝 노이즈 없는지(청크별 resample이 의심되면 서버에서 누적 후 한 번에 resample로 변경).
3. `callone-bench` 로 `tts_first_ms` 통짜 대비 감소 측정(기대 ~1.3s→~0.3-0.5s 첫음성).
4. 라이브 통화에서 끊김 없는지(과거 chunk_size=8 끊김 전례 — 스트리밍은 네트워크 버퍼도 변수).

### 리스크 / 롤백
- 음색 변质 → `tts.stream: false` 한 줄로 즉시 원복.

---

## 작업 2 — 한국어 프롬프트 다이어트 + 샘플러  ★무해·즉시

### 문제 (코드 근거)
- [llama_llm.py:133-148](../callone/serve/llama_llm.py#L133-L148): 부정 규칙("절대 ~마라") 10개+ 과적재. 7.8B에서 지시 충돌 → 퇴행(반복/되묻기).
- **핵심 충돌:** `"1~2문장 짧게"` + `"질문으로 끝내지 마라"` + `"같은 질문 반복 금지"` 동시 강제 → 모델이 안전한 짧은 패턴(되묻기/맞장구 반복)으로 붕괴. **지금 증상과 정확히 일치.**
- [serve.yaml:38](../configs/serve.yaml#L38) `temperature: 0.4` 낮아 단조·반복. 0.7은 비문(과거 실측) → 온도로는 해결 불가, **DRY 샘플러로 우회.**

### 변경점
**A. `_system()` 규칙 블록 슬림화 (충돌 제거 + 긍정 표현)**
```text
[말하는 방식]
- 너는 전화기 너머 진짜 사람이다. 끝까지 그 사람으로 연기하고 AI라 밝히지 마라.
- 한 번에 1~2문장으로 짧게 말하고 상대 차례를 줘라.
- 네 생각·감상·반응을 먼저 말해라. 질문은 꼭 필요할 때만(서너 번에 한 번꼴), 매 턴 되묻지 마라.   ← 충돌 3개를 1개 긍정 규칙으로 통합
- 앞 대화를 기억해 구체적으로 이어가고, 이미 한 말·질문은 반복하지 마라.
- 정해진 말투(반말/존댓말)를 처음부터 끝까지 일관되게.
- 순 한국어(한글)만. 한자·외국어·이모지·괄호 속 행동/혼잣말 금지.
- 비서 말투("도와드릴게요","~할 수 있습니다", 목록·번호) 금지.
```
> 기존 7~8줄 부정나열 → 7줄 긍정/통합. `_strip_unspoken`·`_parse_emotion`(이모지/한자 사후정제)은 안전망으로 유지.

**B. `_payload()` 샘플러 교체 ([llama_llm.py:187-202](../callone/serve/llama_llm.py#L187-L202))**
```python
return {
    "messages": self._messages(user_text, history),
    "max_tokens": self.max_new_tokens,
    "temperature": self.temperature,          # serve.yaml=0.4 유지(검토: 0.6 은 '0.7=비문' 실측에 위배·미검증)
    "min_p": self.min_p,                       # ← 신규(top_k/top_p 는 _payload 가 안 보냄, min_p 로 truncation)
    "stream": stream,
    # DRY 샘플러 — n-gram 반복만 정확 억제, 조사·어미 문법은 안 건드림(repeat_penalty 부작용 회피)
    "dry_multiplier": self.dry_multiplier,     # 0.8
    "dry_base": self.dry_base,                 # 1.75
    "dry_allowed_length": self.dry_allowed_length,  # 2
    "dry_penalty_last_n": -1,                  # 전체 컨텍스트
    # 보조 페널티는 약하게(DRY 가 주력). repeat_last_n=64(settled) 유지 — 넓히면 조사 반복도 눌러 위험.
    "repeat_penalty": 1.05,
    "repeat_last_n": 64,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.3,
    "chat_template_kwargs": {"enable_thinking": False},
}
```
> llama-server 지원 확인 필요 파라미터: `dry_multiplier/dry_base/dry_allowed_length/dry_penalty_last_n`, `min_p` (현 빌드 버전에서 노출되는지 `/props` 또는 단발 테스트로 검증).

**C. `configs/serve.yaml` llm 블록 — 노브 노출**
```yaml
llm:
  temperature: 0.4          # 유지(반복은 DRY+프롬프트가 담당). 더 생생하게는 0.5~0.6 A/B(비문 trade)
  min_p: 0.05               # 신규
  dry_multiplier: 0.8       # 신규
  dry_base: 1.75
  dry_allowed_length: 2
```

**D. example_dialogue 강화 (말투 충실도 = 최대 레버)**
- 캐릭터챗 업계(Character.AI/제타) 공통: 규칙보다 **few-shot 예시 대화**가 말투를 잡음.
- [llama_llm.py:121-124](../callone/serve/llama_llm.py#L121-L124) 의 `example_dialogue` 는 현재 선택값 → UI 설정화면에서 **3~4 교환 권장 안내** + 비면 기본 톤 예시 1개라도 주입.

### 검증
- 고정 한국어 8턴 시나리오로 변경 전/후 응답 비교(반복률·되묻기 빈도·비문 수 수기 카운트).
- DRY on/off A/B: 반복 사라지되 조사·어미 안 깨지는지(과거 페널티 부작용 "그치마니/하렴 야호" 재발 여부).

### 리스크
- 거의 없음(모델/음색 0영향). config 원복 즉시.

---

## 작업 3 — 한국어특화 무검열 LLM 모델 교체 A/B (VRAM 티어별)

### 결정 (웹확인 2026-06-22 + 사용자 확정)
- 현 모델 = **AetherArchitectural/EXAONE-3.5-7.8B-Instruct-abliterated**(imatrix GGUF) = **이미 무검열(abliterated)**, 검증됨.
- **EXAONE 4.0 사이즈 = 1.2B / 32B 두 개뿐**(2025-07-15). 7.8B 같은 중간 없음. 무검열본 둘 다 존재:
  `mradermacher/Huihui-EXAONE-4.0-1.2B-abliterated-GGUF`, `huihui-ai/Huihui-EXAONE-4.0-32B-abliterated`(GGUF 있음).
  ⚠️ huihui 32B = "crude proof-of-concept" → abliterate가 한국어 coherence 깎을 수 있음(벤치로 확인).
  ⚠️ 라이선스 = EXAONE AI Model License = 비상업/연구 제한. **로컬 개인용 OK, 상업화 금지.**
- 1.2B = 현 7.8B보다 약함 = 역행 → **후보 탈락.** Qwen 계열 = 과거 한국어 깨짐 → 제외.

### 모델 선택 = VRAM 티어 자동 (사용자 확정: 두 박스 다 씀)
| GPU | VRAM | LLM | 근거 |
|---|---|---|---|
| **A100/H100** | 80GB | **EXAONE-4.0-32B-abliterated** (Q5~Q6) | 동시구동 여유. 무검열 최신·최대 |
| **3090/4090** | 24GB | **EXAONE-3.5-7.8B-abliterated** (현행 유지) | 32B는 TTS+ASR+Ditto 동시구동 시 24GB 초과 위험 → 제외 |
> 사용자 확정(2026-06-23): A100/H100 기본은 32B Q6_K. 24GB GPU만 7.8B를 유지한다.

### 변경점
**A. `common/hardware` VRAM 감지 + 모델 자동선택**
- 현 `detect_tier()` 는 server_gpu/laptop_cpu만 구분(24 vs 80 구분 안 함).
- VRAM 임계(>=40GB)로 `LLM_REPO`/`LLM_GGUF` 기본값 분기. `bootstrap_gpu.sh` 가 읽음.

**B. `scripts/bootstrap_gpu.sh` — VRAM별 확정 모델 자동선택**
- A100/H100(VRAM>=40GB)은 `mradermacher/Huihui-EXAONE-4.0-32B-abliterated-GGUF` Q6_K를 자동 선택한다. 24GB GPU는 기존 7.8B Q6_K를 유지한다.
```bash
VRAM_GB=... # nvidia-smi
# >=40GB: EXAONE-4.0-32B-abliterated Q6_K
# <40GB:  EXAONE-3.5-7.8B-abliterated Q6_K
```
**C. thinking 처리 per-model (중요)**
- 현 `configs/qwen3_nothink.jinja`·`/no_think` 는 **Qwen3.5 thinking 버그 대응**(llama.cpp #20182). **EXAONE엔 불필요·해로울 수 있음.**
- → `llm.thinking_workaround: qwen|none` 설정. EXAONE(3.5/4.0)=`none`: jinja 미주입 + `_system()` `/no_think` 미부착 + `chat_template_kwargs.enable_thinking:false`만(EXAONE4 reasoning off).
- 착수 시 검증: EXAONE4-32B 가 system+history 있을 때 content 정상 반환(Qwen 빈응답 버그 무관 확인).
**D. 벤치 하니스 신설 `scripts/bench_llm_korean.py`**
```text
입력: --base-url(여러 개), 고정 한국어 멀티턴 시나리오(8턴, 페르소나+상황, 더미데이터)
출력: 모델별 응답 덤프(txt) + 토큰속도. 사람이 A/B 채점(반복·맥락·자연스러움·되묻기).
용도: A100에서 EXAONE-4.0-32B-abliterated vs EXAONE-3.5-7.8B-abliterated → 한국어 승자 확정.
프라이버시: 더미 시나리오만(실데이터 X).
```

### 검증
- 같은 시나리오·같은 프롬프트(작업2 적용본)로 EXAONE3.5-7.8B vs 후보들 응답 나란히 → 사람 채점.
- 동시구동 VRAM·첫토큰 지연 측정(LLM 속도는 빠르다 했으나 14B+avatar 동시 여유 확인).

### 리스크
- 음색 0영향. VRAM 초과 가능 → Q4/사이즈 조정 또는 avatar resolution 낮춤. 한국어가 EXAONE보다 나쁘면 즉시 원복(env 한 줄).

---

## 작업 4 — 영상 파이프라이닝 (라이프사이클 불변)

### 문제
- [orchestrator.py:557-565](../callone/serve/orchestrator.py#L557-L565): 세그먼트 **전체 오디오 모은 뒤**(`np.concatenate`) `frames_for` 1회. TTS가 통짜라 영상도 통짜 = 직렬.
- 첫 프레임 콜드 ~30s(TRT 첫 추론) 그대로 노출. TTS/LLM은 워밍업하나([orchestrator.py:243](../callone/serve/orchestrator.py#L243)) **아바타만 워밍업 빠짐.**

### 변경점 (안전 순)
**A. 아바타 세션 워밍업 — 30s 콜드 은닉 (저위험, 1순위)**
- `init_session`([orchestrator.py:370-385](../callone/serve/orchestrator.py#L370-L385)) 에서 사진 등록 직후, "연결 중" 동안 **더미 1s 무음 오디오로 frames_for 1회**(출력 폐기) → TRT 첫 추론을 통화 시작 구간으로 이동.
- 과거 워밍업 제거 이유 = "드레인 경쟁/타임아웃"인데 그건 **턴 중 워밍업**이었음. **세션 시작 단발**은 경쟁 없음(턴 시작 전).
- 발화 단위 `setup→run→close` 라이프사이클은 그대로(누적 버그 안 건드림). 워밍업도 그 사이클 1회 통과.
```python
# init_session 내 avatar 구성 직후
if self.avatar is not None:
    try:
        import numpy as _np
        for _ in self.avatar.frames_for(_np.zeros(int(1.0*sr_out), dtype=_np.float32), sr_out):
            pass   # TRT 첫 추론 예열, 프레임 폐기
    except Exception as e:
        log.warning("아바타 워밍업 스킵(%s)", e)
```

**B. TTS 스트리밍과 재생 오버랩 확인 (작업1 의존)**
- 작업1로 오디오가 스트리밍되면 프론트가 음성을 먼저 재생 시작, 영상 프레임은 도착하는 대로 동기(이미 A/V 동기 로직 존재 — `AV_LEAD_S` 등 커밋 afe23e1).
- orchestrator는 여전히 발화 전체 오디오로 `frames_for`(Ditto `setup_Nd` 가 총 길이 요구) → **영상 첫 프레임은 TTS 끝난 뒤** 시작. 음성은 먼저 흐르므로 체감 개선, 단 영상-음성 갭 존재.

**C. (2단계, 리스크 큼) 진짜 청크 파이프라인 — 보류·연구 스파이크**
- Ditto `setup_Nd(N_d)` 는 총 프레임수 선지정 필요 → 청크 단위로 못 흘림(현 구조). 텍스트 길이로 N_d 추정 + 순수 online 모드 = 라이프사이클 재설계 = **OOM 버그 영역**. → 1단계(A,B) 효과 측정 후에만.
- 대안 모델(중기, 웹조사): **FlowTalk**(100+FPS, flow-matching, diffusion 5배, TRT), **REST**(스트리밍 e2e 최저지연), **OmniTalker**(25fps). Ditto가 이미 RTF<1이라 **모델 교체보다 A·B 먼저.**

### 검증
- 워밍업 후 첫 턴 영상 첫 프레임 지연 측정(30s→? 기대 콜드가 통화시작 구간으로 이동).
- VRAM(워밍업 1회분), 멀티턴 누적 0 유지(close 사이클 그대로인지 로그 `아바타 N프레임` 확인).

### 리스크
- A는 저위험(세션 시작 단발). C는 보류(버그 영역).

---

## 권장 실행 순서 (리스크 오름차순)
1. **작업2** (프롬프트+샘플러) — 무해, 즉시, 반복/되묻기 바로 완화. **먼저.**
2. **작업4-A** (아바타 워밍업) — 저위험, 30s 콜드 은닉.
3. **작업1 A~C** (TTS 스트리밍) — 효과 최대, 단 저장 wav A/B 통과 후 라이브. config 롤백.
4. **작업3** (모델 교체 A/B) — 벤치 하니스로 후보 채점 후 결정.
5. **작업1-D**(ref 캐싱), **작업4-C**(청크 파이프라인/모델교체) — 2단계, 1차 측정 후.

## 공통 검증 도구
- `callone-bench`(단계별 ms), `~/serve.log` `grep "ASR 완료|LLM 완료|아바타"`, 저장 wav 스펙트로그램 A/B, 고정 한국어 8턴 시나리오.

## 미확정·착수 시 웹 재확인 (§3 의무)
- [x] EXAONE4.0 무검열 확인(2026-06-22): 1.2B/32B만, abliterated 둘 다 존재. 32B=`huihui-ai/Huihui-EXAONE-4.0-32B-abliterated`. 1.2B 탈락(역행). 현 7.8B-abliterated 이미 무검열.
- [x] 32B GGUF 저장소·Q레벨 확정(2026-06-23): `mradermacher/Huihui-EXAONE-4.0-32B-abliterated-GGUF`, Q6_K(약 26.4GB).
- [ ] EXAONE AI Model License 상업화 조건(로컬 개인용은 OK).
- [ ] 현 llama-server 빌드가 `dry_*`·`min_p` 파라미터 노출하는지.
- [ ] CosyVoice3(Fun-CosyVoice3-0.5B) `inference_zero_shot(stream=True)` 청크 yield 형태·경계 노이즈.
- [ ] EXAONE/A.X thinking 동작(Qwen 버그 무관한지) → thinking_workaround=none 검증.
