# callone UI

미스보이스 화면 골격 차용 + **외부 유료 API 제거** → 로컬 백엔드(`calloneClient.ts`).

## 화면
- `/` ContactList — A/B 를 이름·관계로 표시, 통화 시작
- `/call/:id` CallScreen — 타이머·파형·음소거·종료 + 자막, 마이크↔WS↔음성
- `/editor/:id` SpeakerCardEditor — 라벨링 편집기(auto 초안 + user 확정)
- `/processing` ProcessingView — 파이프라인 진행 표시

## 실행
```bash
npm install
npm run dev        # :5173, /api·/ws → localhost:8000(callone-serve) 프록시
```

백엔드를 먼저 `callone-serve` 로 띄워야 화자 목록/통화가 동작한다.
