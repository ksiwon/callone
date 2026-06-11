"""S2 (d) 제3자/잡음 제거 (§10d).

전체 세그먼트 임베딩을 HDBSCAN(또는 임계값) 클러스터링.
2개 주군집 = A·B, 나머지 소군집/이상치 = 제3자 → 제거.
두 센트로이드 모두에 유사도 낮은 세그먼트도 drop.
"""
from __future__ import annotations

import numpy as np

from ..common.audio import cosine
from ..common.logging import get_logger

log = get_logger("filter3p")


def _kmeans2(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans  # type: ignore

    return KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(embeddings)


def two_main_centroids(embeddings: np.ndarray, method: str = "kmeans") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """임베딩 → 2개 주군집 센트로이드 + 각 세그먼트 라벨(0/1/-1=이상치).

    기본 KMeans k=2 (2화자 통화에 신뢰성 높음, 항상 2군집).
    HDBSCAN 은 이상치(제3자) 탐지에 쓰되, 2군집 미만이 나오면 KMeans 로 폴백.
    반환: (centroid0, centroid1, labels)
    """
    if len(embeddings) < 2:
        c = embeddings.mean(axis=0) if len(embeddings) else np.zeros(1)
        return c, c, np.zeros(len(embeddings), dtype=int)

    labels = None
    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            clu = hdbscan.HDBSCAN(min_cluster_size=max(5, len(embeddings) // 50))
            labels = clu.fit_predict(embeddings)
            n_clusters = len(np.unique(labels[labels >= 0]))
            if n_clusters < 2:
                log.warning("HDBSCAN 군집 %d개(<2) — KMeans 폴백", n_clusters)
                labels = None
        except Exception as e:  # noqa: BLE001
            log.warning("HDBSCAN 불가(%s) — KMeans 폴백", e)

    if labels is None:
        try:
            labels = _kmeans2(embeddings)
        except Exception as e:  # noqa: BLE001
            log.warning("KMeans 불가(%s) — 단일군집 폴백", e)
            labels = np.zeros(len(embeddings), dtype=int)

    # 가장 큰 2개 군집 = A,B
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(uniq) == 0:
        c = embeddings.mean(axis=0)
        return c, c, labels
    order = uniq[np.argsort(-counts)]
    top2 = order[:2]
    c0 = embeddings[labels == top2[0]].mean(axis=0)
    c1 = embeddings[labels == top2[1]].mean(axis=0) if len(top2) > 1 else c0

    # 라벨 재매핑: 주군집0→0, 주군집1→1, 나머지→-1(제3자)
    remapped = np.full(len(labels), -1, dtype=int)
    remapped[labels == top2[0]] = 0
    if len(top2) > 1:
        remapped[labels == top2[1]] = 1
    return c0, c1, remapped


def assign_AB(emb: np.ndarray, cA: np.ndarray, cB: np.ndarray,
              sim_threshold: float) -> dict:
    """단일 세그먼트 임베딩 → A/B 귀속 + 제3자 판정."""
    sa, sb = cosine(emb, cA), cosine(emb, cB)
    if sa < sim_threshold and sb < sim_threshold:
        return {"global_speaker": "UNK", "sim_A": sa, "sim_B": sb, "is_thirdparty": True}
    gs = "A" if sa >= sb else "B"
    return {"global_speaker": gs, "sim_A": sa, "sim_B": sb, "is_thirdparty": False}
