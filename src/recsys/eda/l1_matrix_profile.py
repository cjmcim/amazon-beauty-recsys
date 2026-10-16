"""L1 矩阵画像：稀疏度、长尾、头部贡献 —— 迁移自 legacy/EDA_L1.py。

输出：日志统计 + 用户/商品交互频次直方图 + 头部贡献占比图。
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.recsys.config import Config
from src.recsys.eda.plots import save_bar
from src.recsys.io_utils import read_interaction_json

logger = logging.getLogger("recsys.eda.l1")


def _head_contribution(cnt_df, total: int, top_pct: float) -> float:
    """按交互数降序取前 top_pct 的个体，返回其贡献占比。"""
    n = int(cnt_df.count() * top_pct)
    return cnt_df.orderBy(F.desc("inter_cnt")).limit(n) \
        .agg(F.sum("inter_cnt").alias("s")).collect()[0]["s"] / total


def run(spark: SparkSession, cfg: Config) -> None:
    logger.info("=" * 70)
    logger.info("【EDA L1】矩阵画像 — 稀疏度、长尾、头部贡献")
    logger.info("=" * 70)

    inter_df = read_interaction_json(spark, cfg.paths.inter_path).cache()
    total_interactions = inter_df.count()
    total_users = inter_df.select("user_id").distinct().count()
    total_items = inter_df.select("item_id").distinct().count()
    logger.info("总交互条数: %s | 独立用户: %s | 独立商品: %s",
                f"{total_interactions:,}", f"{total_users:,}", f"{total_items:,}")

    max_possible = total_users * total_items
    logger.info("理论最大交互(用户×商品): %s | 矩阵稀疏度: %.4f",
                f"{max_possible:,}", 1.0 - total_interactions / max_possible)

    # ---------- 用户侧 ----------
    user_cnt_df = inter_df.groupBy("user_id").count().withColumnRenamed("count", "inter_cnt")
    user_cnt_df.select("inter_cnt").summary().show(truncate=False)

    # 用户交互频次分布（直方图）
    user_cnt_pdf = user_cnt_df.select("inter_cnt").toPandas()
    save_bar(user_cnt_pdf.groupby("inter_cnt").size().reset_index(name="count"),
             "inter_cnt", "count", "User interaction frequency distribution",
             cfg.paths.eda_plots, "l1_user_inter_cnt_dist.png",
             xlabel="inter_cnt", ylabel="user_cnt", rot=0)

    logger.info("头部用户交互贡献模拟:")
    top5 = _head_contribution(user_cnt_df, total_interactions, 0.05)
    top10 = _head_contribution(user_cnt_df, total_interactions, 0.10)
    logger.info("Top5%% 用户贡献交互占比: %.2f%%", top5 * 100)
    logger.info("Top10%% 用户贡献交互占比: %.2f%%", top10 * 100)

    # ---------- 物品侧 ----------
    item_cnt_df = inter_df.groupBy("item_id").count().withColumnRenamed("count", "inter_cnt")
    item_cnt_df.select("inter_cnt").summary().show(truncate=False)
    item_cnt_pdf = item_cnt_df.select("inter_cnt").toPandas()
    save_bar(item_cnt_pdf.groupby("inter_cnt").size().reset_index(name="count"),
             "inter_cnt", "count", "Item interaction frequency distribution",
             cfg.paths.eda_plots, "l1_item_inter_cnt_dist.png",
             xlabel="inter_cnt", ylabel="item_cnt", rot=0)

    logger.info("头部商品交互贡献模拟:")
    top5 = _head_contribution(item_cnt_df, total_interactions, 0.05)
    top10 = _head_contribution(item_cnt_df, total_interactions, 0.10)
    logger.info("Top5%% 商品贡献交互占比: %.2f%%", top5 * 100)
    logger.info("Top10%% 商品贡献交互占比: %.2f%%", top10 * 100)

    inter_df.unpersist()
