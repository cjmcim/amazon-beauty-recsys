"""PIT 特征挂载 —— 迁移自 legacy 精排 Part 3。

重大修复背景（保留注释说明）：
早期版本用 user_id + pit_time 等值 join 用户特征，导致 val 负样本
无法匹配、特征全部置 0。修复方案：复用 user_ts_feat 的每条历史快照，
对每个样本取「<= pit_time 的最近一条快照」。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger("recsys.rank.pit")


def attach_features_pit(sample_df: DataFrame, user_ts_feat_df: DataFrame,
                        item_feat_df: DataFrame) -> DataFrame:
    """PIT 挂载用户特征 + 物品特征。

    Args:
        sample_df: 必须包含 user_id, item_id, pit_time, label
        user_ts_feat_df: user_ts_feat (user_id, hist_ts, 用户 PIT 快照特征)
        item_feat_df: 物品静态特征

    Returns:
        挂载好用户最近快照 + 物品特征的样本表（含交叉 flag）
    """
    s = sample_df.select("user_id", "item_id", "pit_time", "label")
    joined = (
        s.join(user_ts_feat_df, on="user_id", how="left")
        .filter(F.col("pit_time") > F.col("hist_ts"))
    )
    # ⚠️ partitionBy 必须含 item_id：负样本 pit_time 统一为 user last_ts，
    # 若只按 (user_id, pit_time) 分区，同用户所有负样本会竞争 rn=1 而折叠成 1 条。
    w_nearest = Window.partitionBy("user_id", "item_id", "pit_time").orderBy(F.desc("hist_ts"))
    nearest = (
        joined.withColumn("rn", F.row_number().over(w_nearest))
        .filter(F.col("rn") == 1)
        .drop("rn", "hist_ts")
    )
    df = nearest.join(item_feat_df, on="item_id", how="left")
    df = df.fillna({
        "user_hist_cnt": 0,
        "user_avg_rating": 0.0,
        "user_rating_std": 0.0,
        "user_max_rating": 0.0,
        "user_min_rating": 0.0,
        "user_avg_text_len": 0.0,
        "user_pos_cnt": 0,
        "item_train_cnt": 0,
        "item_train_avg_rating": 0.0,
        "item_train_rating_std": 0.0,
        "item_train_pos_cnt": 0,
        "item_train_avg_text_len": 0.0,
    })
    # 交叉 flag（用户历史类目/品牌与当前物品是否一致）
    df = df.withColumn(
        "user_interact_cat_flag",
        F.when(
            (F.col("user_hist_cats").isNotNull())
            & (F.array_contains(F.col("user_hist_cats"), F.col("main_category"))),
            1,
        ).otherwise(0),
    ).withColumn(
        "user_interact_brand_flag",
        F.when(
            (F.col("user_hist_brands").isNotNull())
            & (F.array_contains(F.col("user_hist_brands"), F.col("brand"))),
            1,
        ).otherwise(0),
    )
    miss_cnt = df.filter(F.col("user_hist_cnt").isNull()).count()
    if miss_cnt > 0:
        logger.warning("⚠️ attach_features_pit 用户特征缺失行数: %s", miss_cnt)
    return df
