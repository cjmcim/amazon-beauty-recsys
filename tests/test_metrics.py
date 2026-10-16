"""共享评估指标单元测试（纯 pandas/numpy，无需 Spark）。

手工构造两组用户数据，期望值按 NDCG/Recall 定义显式计算，避免「用实现验证实现」。
"""

import numpy as np
import pandas as pd
import pytest

from src.recsys.evaluate.metrics import calc_diversity, ranking_metrics, roc_auc


def dcg(rels):
    idx = np.arange(1, len(rels) + 1)
    return np.sum((2 ** rels - 1) / np.log2(idx + 1))


def ndcg(labels, k):
    k_labs = labels[:k]
    ideal = np.sort(labels)[::-1][:k]
    idcg = dcg(ideal)
    return 0.0 if idcg == 0 else dcg(k_labs) / idcg


def _make_df():
    # u1: 2 个正样本；u2: 1 个正样本
    return pd.DataFrame({
        "user_id": ["u1"] * 4 + ["u2"] * 2,
        "item_id": ["a", "b", "c", "d", "e", "f"],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "label": [1, 0, 1, 0, 0, 1],
    })


def test_ranking_metrics_sort_by_score():
    pdf = _make_df()
    res = ranking_metrics(pdf, ks=[1, 2, 4], score_col="score", sort=True)

    # u1 按 score 排序: [1,0,1,0]; u2: [0,1]
    labels_u1 = np.array([1, 0, 1, 0])
    labels_u2 = np.array([0, 1])

    assert res["Recall@1"] == pytest.approx((labels_u1[:1].sum() / labels_u1.sum()
                                             + labels_u2[:1].sum() / labels_u2.sum()) / 2)
    assert res["Recall@2"] == pytest.approx((labels_u1[:2].sum() / labels_u1.sum()
                                             + labels_u2[:2].sum() / labels_u2.sum()) / 2)
    assert res["Recall@4"] == pytest.approx((labels_u1[:4].sum() / labels_u1.sum()
                                             + labels_u2[:4].sum() / labels_u2.sum()) / 2)
    assert res["NDCG@4"] == pytest.approx((ndcg(labels_u1, 4) + ndcg(labels_u2, 4)) / 2)


def test_ranking_metrics_row_order():
    """重排场景：sort=False 时按当前行顺序评估。"""
    pdf = _make_df()
    res = ranking_metrics(pdf, ks=[1], sort=False)
    # u1 行顺序首条 label=1 → Recall@1=0.5；u2 行顺序首条 label=0 → 0
    assert res["Recall@1"] == pytest.approx((0.5 + 0.0) / 2)


def test_ranking_metrics_all_negative_user_skipped():
    """label 全为 0 的用户不参与均值（与 legacy 一致）。"""
    pdf = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "score": [0.9, 0.5, 0.8, 0.4],
        "label": [1, 0, 0, 0],
    })
    res = ranking_metrics(pdf, ks=[1, 2], score_col="score")
    # 仅 u1 参与: [1,0] → Recall@1 = 1/1 = 1, Recall@2 = 1/1 = 1
    assert res["Recall@1"] == pytest.approx(1.0)
    assert res["Recall@2"] == pytest.approx(1.0)


def test_calc_diversity():
    pdf = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2"],
        "main_category": ["A", "B", "A", "C", "C"],
    })
    # u1: {A,B}=2; u2: {C}=1 → 均值 1.5
    assert calc_diversity(pdf) == pytest.approx(1.5)


def test_roc_auc():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    auc = roc_auc(y_true, y_score)
    assert 0.0 <= auc <= 1.0
    # 全对序时 AUC=1
    assert roc_auc([0, 1], [0.1, 0.9]) == pytest.approx(1.0)
