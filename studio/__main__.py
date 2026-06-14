#!/usr/bin/env python3
"""callone studio 런처 — `python -m studio` 로 통합 앱 실행.

(coding/callone/ 에서 실행. 헤더에서 [환경] [목적] [데이터모드] 고르면
 알아서 맞는 파이프라인으로 라우팅한다.)
"""
import sys
from pathlib import Path

# coding/callone 을 path 에 → `import studio` 가능(cwd 무관)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.app import main  # noqa: E402

if __name__ == "__main__":
    main()
