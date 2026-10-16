"""规则后重排 —— 迁移自 legacy「重排模块｜规则型后重排」。

策略：
1. 调整分数 = 精排 score − 热门惩罚 × log1p(物品热度)；
2. 按调整分数降序逐条选取，单个类目占比不超过 max_category_ratio；
3. 兜底：不足 top_n 时用剩余候选补齐（不再做类目限制）；
4. 与「原始精排排序」对比 Recall@K / NDCG@K / 多样性。

修复记录：legacy 中 ``print(...Top{N_OUTPUT}...)`` 的 N_OUTPUT 未定义（NameError），
迁移后统一使用 cfg.rerank.top_n_output。
"""

from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.recsys.config import RerankConfig
from src.recsys.evaluate.metrics import calc_diversity, ranking_metrics

if TYPE_CHECKING:  # 仅类型标注需要，避免纯逻辑测试依赖 pyspark
    from pyspark.sql import DataFrame

logger = logging.getLogger("recsys.rerank.reranker")


def reorder_one_user(group_df: pd.DataFrame, top_n: int, hot_penalty: float,
                     max_cat_ratio: float) -> pd.DataFrame:
    """单用户列表重排。输入需含 item_id / score / main_category / item_cnt_log。"""
    df = group_df.copy().reset_index(drop=True)
    # 1) 调整分数：score − 热门惩罚
    df["adjust_score"] = df["score"] - hot_penalty * df["item_cnt_log"]
    df = df.sort_values("adjust_score", ascending=False).reset_index(drop=True)

    # 2) 类目限流：同一类目最多 max_per_cat 条
    max_per_cat = int(top_n * max_cat_ratio)
    out_list: List[dict] = []
    cat_counter: Dict[str, int] = {}

    for _, row in df.iterrows():
        cat = row["main_category"]
        cnt = cat_counter.get(cat, 0)
        if cnt >= max_per_cat:
            continue
        out_list.append(row.to_dict())  # 统一转 dict，兼容 pandas 3.x 构造
        cat_counter[cat] = cnt + 1
        if len(out_list) >= top_n:
            break

    # 3) 兜底：不足 top_n 时把剩余候选补进来（不再做类目限制）
    if len(out_list) < top_n:
        selected_item_ids = {r["item_id"] for r in out_list}
        remain = df[~df["item_id"].isin(selected_item_ids)].reset_index(drop=True)
        need = top_n - len(out_list)
        out_list.extend(remain.head(need).to_dict("records"))

    return pd.DataFrame(out_list)


def run_rerank(test_result: pd.DataFrame, item_feat, cfg: RerankConfig,
               result_dir) -> Dict:
    """重排阶段入口。

    Args:
        test_result: 精排输出（user_id, item_id, score, label）
        item_feat: 物品特征表（Spark DataFrame，取 main_category / item_train_cnt）
        result_dir: 结果输出目录

    Returns:
        {"df_origin_top", "df_reordered", "metrics", "diversity"}
    """
    from pyspark.sql import DataFrame  # 仅此阶段需要 Spark

    assert isinstance(item_feat, DataFrame), "item_feat 必须是 Spark DataFrame"
    logger.info("=" * 70)
    logger.info("【stage: rerank】规则后重排：多样性 + 热门惩罚")
    logger.info("=" * 70)

    # ---------- 步骤1：物品特征 join 进精排结果 ----------
    item_meta_pdf = item_feat.select(
        "item_id", "main_category", "item_train_cnt"
    ).toPandas()
    # ⚠️ test_result 已含全部特征列（item_train_cnt 等），merge 前只保留核心列，
    # 否则同名列冲突产生 item_train_cnt_x/_y 导致 KeyError
    test_result_w_meta = test_result[["user_id", "item_id", "score", "label"]].merge(
        item_meta_pdf, on="item_id", how="left"
    )
    test_result_w_meta["item_cnt_log"] = np.log1p(
        test_result_w_meta["item_train_cnt"].fillna(0)
    )

    # ---------- 步骤2：全用户重排 ----------
    origin_result_list: List[pd.DataFrame] = []
    reorder_result_list: List[pd.DataFrame] = []

    for uid, group in test_result_w_meta.groupby("user_id"):
        origin_top = group.sort_values("score", ascending=False).head(cfg.top_n_output).copy()
        origin_result_list.append(origin_top)

        reordered = reorder_one_user(
            group, top_n=cfg.top_n_output,
            hot_penalty=cfg.hot_penalty_coeff,
            max_cat_ratio=cfg.max_category_ratio,
        )
        reordered["user_id"] = uid
        reorder_result_list.append(reordered)

    df_origin_top = pd.concat(origin_result_list, ignore_index=True)
    df_reordered = pd.concat(reorder_result_list, ignore_index=True)
    logger.info("原始精排Top%d样本数: %s | 重排后输出样本数: %s",
                cfg.top_n_output, f"{len(df_origin_top):,}", f"{len(df_reordered):,}")

    # ---------- 步骤3：指标对比 ----------
    # 原始精排：按 score 排序评估；重排：按最终行顺序评估
    metric_origin = ranking_metrics(df_origin_top, ks=cfg.eval_topk,
                                    score_col="score", sort=True)
    metric_reorder = ranking_metrics(df_reordered, ks=cfg.eval_topk, sort=False)

    logger.info("指标对比：精排原始排序 VS 规则重排")
    for k in cfg.eval_topk:
        logger.info("  Recall@%-4d origin=%.4f reorder=%.4f",
                    k, metric_origin[f"Recall@{k}"], metric_reorder[f"Recall@{k}"])
        logger.info("  NDCG@%-4d  origin=%.4f reorder=%.4f",
                    k, metric_origin[f"NDCG@{k}"], metric_reorder[f"NDCG@{k}"])

    # ---------- 步骤4：多样性对比 ----------
    div_origin = calc_diversity(df_origin_top)
    div_reorder = calc_diversity(df_reordered)
    logger.info("多样性(平均每用户不同类目数): origin=%.2f reorder=%.2f",
                div_origin, div_reorder)

    # ---------- 保存结果 ----------
    out_csv = result_dir / "reorder_final_result.csv"
    df_reordered.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("✅ 重排结果已保存: %s", out_csv)

    return {
        "df_origin_top": df_origin_top,
        "df_reordered": df_reordered,
        "metrics": {"origin": metric_origin, "reorder": metric_reorder},
        "diversity": {"origin": div_origin, "reorder": div_reorder},
    }
