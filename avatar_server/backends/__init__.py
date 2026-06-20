"""avatar-server GPU 백엔드(Ditto/MuseTalk). 별 venv(.venv-avatar)서만 import 가능.

callone 서빙 venv 엔 들어오지 않는다(transformers/diffusers/mmcv 충돌 회피). app._pick_model 이
AVATAR_BACKEND 에 따라 여기서 골라 로드하고, 실패하면 StaticModel(CPU) 로 폴백.
"""
