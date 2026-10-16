"""精排样本构造 —— 迁移自 legacy 精排 Part 2。

正样本：rating >= pos_rating_th 的真实交互；
负样本两类：
1. hard 负样本：出现在召回候选、但用户未真实高分交互的商品
   （保留用户低分交互样本，语义为「曝光但低分」）；
2. 随机负样本：用户从未交互过的商品（数量受控，避免 crossJoin 爆炸）。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.recsys.config import RankConfig

logger = logging.getLogger("recsys.rank.sampling")


def get_user_last_ts(inter_df: DataFrame) -> DataFrame:
    """每个用户在该集合中的最后交互时间，用作负样本默认 PIT。"""
    return inter_df.groupBy("user_id").agg(F.max("timestamp").alias("pit_time"))


def build_pos(inter_df: DataFrame, name: str, pos_rating_th: float) -> DataFrame:
    """正样本：真实交互且 rating 达标，返回 (user_id, item_id, pit_time, label=1)。"""
    pos = (
        inter_df.filter(F.col("rating") >= pos_rating_th)
        .select("user_id", "item_id",
                F.col("timestamp").alias("pit_time"),
                F.lit(1).alias("label"))
        .dropDuplicates(["user_id", "item_id"])
    )
    logger.info("%s 正样本数: %s", name, f"{pos.count():,}")
    return pos


def sample_hard_neg(pos_df: DataFrame, recall_df: DataFrame,
                    user_last_ts_df: DataFrame, ratio: int,
                    name: str, seed: int) -> DataFrame:
    """Hard 负样本：召回候选 - 正样本对，pit_time 取用户 last_ts。"""
    pos_pairs = pos_df.select("user_id", "item_id").distinct()
    hard = (
        recall_df.select("user_id", "item_id").distinct()
        .join(pos_pairs, on=["user_id", "item_id"], how="left_anti")
        .join(user_last_ts_df, on="user_id", how="inner")
        .withColumn("label", F.lit(0))
    )
    w = Window.partitionBy("user_id").orderBy(F.rand(seed))
    hard = (
        hard.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") <= ratio)
        .drop("rn")
    )
    logger.info("%s Hard负样本数: %s", name, f"{hard.count():,}")
    return hard


def sample_random_neg(pos_df: DataFrame, item_pool_df: DataFrame,
                      user_last_ts_df: DataFrame, all_user_item_pairs: DataFrame,
                      ratio: int, name: str, seed: int) -> DataFrame:
    """全局随机负样本：按用户从物品池随机采样（每用户 ratio 个）。

    实现：给物品池编 1..N 号，每个用户生成 ratio 个随机编号，
    再 join 出物品、剔除该用户已交互过的对。数据量 = 用户数 × ratio，
    避免「用户 × 物品」全笛卡尔积在内存中爆炸。
    """
    if ratio <= 0:
        logger.info("%s 随机负样本数: 0 (ratio<=0)", name)
        return pos_df.select("user_id", "item_id", "pit_time").filter(F.lit(False)) \
                     .withColumn("label", F.lit(0))

    users = pos_df.select("user_id").distinct()
    item_idx = (
        item_pool_df.select("item_id")
        .withColumn("item_idx", F.row_number().over(Window.orderBy(F.col("item_id"))))
    )
    n_items = item_idx.count()
    if n_items == 0:
        raise ValueError("物品池为空，无法构造随机负样本")

    neg = (
        users
        .withColumn("k", F.explode(F.array(*[F.lit(i) for i in range(ratio)])))
        .withColumn("item_idx",
                    (F.floor(F.rand(seed) * F.lit(n_items)) + 1).cast("long"))
        .join(item_idx, on="item_idx", how="inner")
        .join(all_user_item_pairs, on=["user_id", "item_id"], how="left_anti")
        .join(user_last_ts_df, on="user_id", how="inner")
        .withColumn("label", F.lit(0))
        .drop("k", "item_idx")
    )
    logger.info("%s 随机负样本数: %s", name, f"{neg.count():,}")
    return neg
