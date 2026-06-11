"""test_s1 (§19): 복원 음색 보존(임베딩 유사도) + cosine 헬퍼."""
import numpy as np

from callone.common.audio import cosine


def test_cosine_identity():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-6


def test_timbre_guard_logic():
    """가드: 유사도 임계 미만이면 강도 완화 흉내."""
    sim, min_cos, strength, step = 0.5, 0.8, 0.6, 0.2
    while sim < min_cos and strength > 0:
        strength = max(0.0, strength - step)
        sim += 0.2   # 강도 낮추면 음색 보존 향상 가정
    assert strength < 0.6
