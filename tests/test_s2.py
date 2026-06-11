"""test_s2 (§19): 정확히 2개 주군집(A/B) + 제3자 이상치 분리."""
import numpy as np

from callone.diarize.filter_thirdparty import assign_AB, two_main_centroids


def test_two_clusters_and_thirdparty():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (30, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    b = rng.normal(0, 0.05, (30, 8)) + np.array([0, 1, 0, 0, 0, 0, 0, 0])
    third = np.array([[0, 0, 1, 0, 0, 0, 0, 0]])
    emb = np.vstack([a, b, third])
    cA, cB, labels = two_main_centroids(emb, method="hdbscan")
    # 두 센트로이드는 서로 달라야
    assert not np.allclose(cA, cB)
    # 제3자는 두 센트로이드 모두에 유사도 낮음 → thirdparty
    res = assign_AB(third[0], cA, cB, sim_threshold=0.55)
    assert res["is_thirdparty"]


def test_assign_ab_picks_closer():
    cA = np.array([1.0, 0.0])
    cB = np.array([0.0, 1.0])
    res = assign_AB(np.array([0.9, 0.1]), cA, cB, 0.3)
    assert res["global_speaker"] == "A"
