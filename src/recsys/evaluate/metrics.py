"""共享评估指标（纯 pandas/numpy 实现，精排与重排复用）。

- ranking_metrics: 每用户分组算 Recall@K / NDCG@K，取均值
- calc_diversity: 每用户结果中不同类目数均值
- roc_auc: 候选集 AUC

与原 legacy 的两个重复实现相比，统一成一份，行为保持等价：
- 精排场景: sort=True，按 score 降序作为排序；
- 重排场景: sort=False，直接按 DataFrame 当前行顺序视为最终排序。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _dcg(rels: np.ndarray) -> float:
    idx = np.arange(1, len(rels) + 1)
    return float(np.sum((2 ** rels - 1) / np.log2(idx + 1)))


def ranking_metrics(pdf: pd.DataFrame, ks: Sequence[int],
                    score_col: str | None = None, label_col: str = "label",
                    sort: bool = True) -> Dict[str, float]:
    """按 user_id 分组的排序指标。

    Args:
        pdf: 至少包含 user_id / label（和可选 score）的 DataFrame
        ks: 评估的 K 列表
        score_col: 排序依据列；sort=False 时忽略
        sort: True → 组内按 score 降序；False → 保持当前行顺序

    Returns:
        {"Recall@5": ..., "NDCG@5": ..., ...}
    """
    recalls = {int(k): [] for k in ks}
    ndcgs = {int(k): [] for k in ks}

    for _, group in pdf.groupby("user_id"):
        if group[label_col].sum() <= 0:
            continue
        g = group
        if sort:
            g = group.sort_values(score_col, ascending=False)
        labels = g[label_col].values.astype(float)
        for k in ks:
            topk_labels = labels[:k]
            recalls[k].append(float(topk_labels.sum() / labels.sum()))
            ideal = np.sort(labels)[::-1][:k]
            idcg = _dcg(ideal)
            ndcgs[k].append(0.0 if idcg == 0 else _dcg(topk_labels) / idcg)

    res: Dict[str, float] = {}
    for k in ks:
        res[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        res[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return res


def calc_diversity(pdf: pd.DataFrame, cat_col: str = "main_category") -> float:
    """多样性：每用户输出结果中不同类目数，取均值（越大越多样）。"""
    div_list = []
    for _, g in pdf.groupby("user_id"):
        cat_set = set(g[cat_col].dropna().tolist())
        div_list.append(len(cat_set))
    return float(np.mean(div_list)) if div_list else 0.0


def roc_auc(y_true: Sequence, y_score: Sequence) -> float:
    """候选集 AUC（需要 scikit-learn）。"""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))
