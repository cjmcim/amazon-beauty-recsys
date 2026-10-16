"""重排逻辑单元测试（纯 pandas，无需 Spark）。

覆盖：
1. 类目限流：单类目不超过 max_per_cat；
2. 热门惩罚：高热度物品被压分降序；
3. 兜底补足：类目限制后不足 top_n 用剩余候选补齐。
"""

import pandas as pd
import pytest

from src.recsys.rerank.reranker import reorder_one_user


def _df(items, cats, scores, cnt_logs):
    return pd.DataFrame({
        "item_id": items,
        "score": scores,
        "main_category": cats,
        "item_cnt_log": cnt_logs,
    })


def test_category_cap_and_fallback():
    # 6 个物品，A/B 两类各 3 个；top_n=4 → max_per_cat=int(4*0.35)=1
    df = _df(
        items=["i0", "i1", "i2", "i3", "i4", "i5"],
        cats=["A", "A", "A", "B", "B", "B"],
        scores=[10, 9, 8, 7, 6, 5],
        cnt_logs=[0.0] * 6,
    )
    out = reorder_one_user(df, top_n=4, hot_penalty=0.08, max_cat_ratio=0.35)

    assert len(out) == 4
    # 第一优先级：A 里最高分 i0，然后 B 里最高分 i3
    assert out.iloc[0]["item_id"] == "i0"
    assert out.iloc[1]["item_id"] == "i3"
    # 兜底补齐后共 4 条
    assert set(out["item_id"]).issubset({"i0", "i1", "i2", "i3", "i4", "i5"})


def test_hot_penalty_reorders():
    # i0 原始分更高但热度极大（cnt_log=100），i1 热度低 → i1 应排前面
    df = _df(
        items=["i0", "i1"],
        cats=["A", "B"],
        scores=[9.0, 8.0],
        cnt_logs=[100.0, 0.05],
    )
    out = reorder_one_user(df, top_n=2, hot_penalty=0.08, max_cat_ratio=0.35)
    assert out.iloc[0]["item_id"] == "i1"
    # adjust_score 被正确写入
    assert out.iloc[0]["adjust_score"] == pytest.approx(8.0 - 0.08 * 0.05)


def test_all_same_category_no_infinite_loop():
    """全部同一类目且 top_n 大于 max_per_cat：靠兜底补齐，不能死循环/丢条。"""
    df = _df(
        items=[f"i{i}" for i in range(6)],
        cats=["A"] * 6,
        scores=[10, 9, 8, 7, 6, 5],
        cnt_logs=[0.0] * 6,
    )
    out = reorder_one_user(df, top_n=4, hot_penalty=0.08, max_cat_ratio=0.35)
    assert len(out) == 4
