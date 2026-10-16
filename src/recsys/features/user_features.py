"""用户侧特征 —— 迁移自 legacy 特征工程 Part B/C/E。

核心概念：PIT（Point-In-Time）快照。
- user_dynamic: 对 train 交互按时间排序，用窗口函数在「当前行之前」聚合用户历史
  （unboundedPreceding ~ -1 行），严格保证当前样本看不到未来信息；
- user_ts_feat: 用户每个历史时间点的快照表（hist_ts + 特征），
  供 val/test 及精排负样本做「最近一次 <= 样本时间的快照匹配」；
- attach_val_test_feat: val/test 样本取 <= label 时间的最近快照挂载。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.recsys.features.context import add_context_features

logger = logging.getLogger("recsys.features.user")


def _pit_window():
    """PIT 窗口：当前行之前的全部历史（函数内构造，兼容 PySpark 4 的上下文要求）。"""
    return Window.partitionBy("user_id") \
        .orderBy("timestamp") \
        .rowsBetween(Window.unboundedPreceding, -1)


def build_user_dynamic(train: DataFrame) -> DataFrame:
    """Part B: 用户动态特征（PIT 快照）。"""
    logger.info("=" * 60)
    logger.info("【Part B: 用户动态特征（PIT 快照）】")
    logger.info("=" * 60)

    w_pit = _pit_window()
    user_dynamic = (
        train.select("user_id", "item_id", "timestamp", "rating", "reviewText",
                     "brand", "main_category")
        .withColumn("user_hist_cnt", F.count("item_id").over(w_pit))
        .withColumn("user_avg_rating", F.avg("rating").over(w_pit))
        .withColumn("user_rating_std", F.stddev("rating").over(w_pit))
        .withColumn("user_max_rating", F.max("rating").over(w_pit))
        .withColumn("user_min_rating", F.min("rating").over(w_pit))
        .withColumn("user_avg_text_len", F.avg(F.length(F.col("reviewText"))).over(w_pit))
        .withColumn("user_pos_cnt", F.count(F.when(F.col("rating") >= 4, True)).over(w_pit))
        .withColumn("user_hist_cats", F.collect_set("main_category").over(w_pit))
        .withColumn("user_hist_brands", F.collect_set("brand").over(w_pit))
        .withColumn("user_hist_items", F.collect_set("item_id").over(w_pit))
    )
    user_dynamic.cache()
    logger.info("train 样本用户 PIT 快照行数: %s", f"{user_dynamic.count():,}")
    return user_dynamic


def build_user_item_set(train: DataFrame) -> DataFrame:
    """Part C: 用户历史物品集合（供召回过滤，全局 train 即可）。"""
    logger.info("=" * 60)
    logger.info("【Part C: 用户历史物品集合】")
    logger.info("=" * 60)
    user_item_set = (
        train.select("user_id", "item_id")
        .distinct()
        .groupBy("user_id")
        .agg(F.collect_set("item_id").alias("user_hist_items_global"))
    )
    user_item_set.cache()
    logger.info("用户历史物品集合用户数: %s", f"{user_item_set.count():,}")
    return user_item_set


def build_user_ts_feat(user_dynamic: DataFrame) -> tuple[DataFrame, DataFrame]:
    """从 user_dynamic 拆出两份对外接口表：
    - user_ts_feat: (user_id, hist_ts, 用户特征...) 供精排 PIT 最近快照匹配
    - user_feat_pit: 同数据但 hist_ts 重命名为 pit_time（兼容旧接口命名）
    """
    feat_cols = [
        "user_id",
        F.col("timestamp").alias("hist_ts"),
        "user_hist_cnt", "user_avg_rating", "user_rating_std",
        "user_max_rating", "user_min_rating", "user_avg_text_len", "user_pos_cnt",
        "user_hist_cats", "user_hist_brands", "user_hist_items",
    ]
    user_ts_feat = user_dynamic.select(*feat_cols).cache()
    user_feat_pit = user_ts_feat.withColumnRenamed("hist_ts", "pit_time").cache()
    return user_ts_feat, user_feat_pit


def attach_item_and_context(df: DataFrame, item_feat: DataFrame) -> DataFrame:
    """挂载物品特征 + 上下文特征 + 用户历史交叉 flag（Part E 公共部分）。"""
    df = df.join(item_feat, on="item_id", how="left")
    df = add_context_features(df)
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
    return df


def build_train_feature(user_dynamic: DataFrame, item_feat: DataFrame) -> DataFrame:
    """Part E: train 特征（user_dynamic 已含 PIT 用户特征，只需挂物品+上下文）。"""
    train_feat = attach_item_and_context(
        user_dynamic.select(
            "user_id", "item_id", "timestamp",
            "user_hist_cnt", "user_avg_rating", "user_rating_std",
            "user_max_rating", "user_min_rating", "user_avg_text_len", "user_pos_cnt",
            "user_hist_cats", "user_hist_brands", "user_hist_items",
        ),
        item_feat,
    )
    return train_feat


def attach_val_test_feat(samples_df: DataFrame, user_ts_feat: DataFrame,
                         item_feat: DataFrame, name: str) -> DataFrame:
    """val/test 样本：取 <= label 时间的最近一条用户快照，再挂物品特征。

    ⚠️ 必须满足 PIT：样本的 label_ts 严格大于快照的 hist_ts。
    """
    s = samples_df.select(
        "user_id", "item_id",
        F.col("timestamp").alias("label_ts"),
    )
    joined = (
        s.join(user_ts_feat, on="user_id", how="left")
        .filter(F.col("label_ts") > F.col("hist_ts"))
    )
    # ⚠️ partitionBy 必须含 item_id：避免同用户同时刻多条样本互相折叠
    w_nearest = Window.partitionBy("user_id", "item_id", "label_ts").orderBy(F.desc("hist_ts"))
    nearest = (
        joined.withColumn("rn", F.row_number().over(w_nearest))
        .filter(F.col("rn") == 1)
        .drop("rn", "hist_ts")
        .withColumnRenamed("label_ts", "timestamp")
    )
    feat = attach_item_and_context(nearest, item_feat)
    logger.info("%s 特征表行数: %s", name, f"{feat.count():,}")
    return feat


def fill_feat_missing(df: DataFrame) -> DataFrame:
    """特征缺失填充（统一默认值，与 legacy 完全一致）。"""
    return df.fillna({
        "user_hist_cnt": 0,
        "user_avg_rating": 0.0,
        "user_rating_std": 0.0,
        "user_max_rating": 0.0,
        "user_min_rating": 0.0,
        "user_avg_text_len": 0.0,
        "user_pos_cnt": 0,
        "user_interact_cat_flag": 0,
        "user_interact_brand_flag": 0,
        "item_train_cnt": 0,
        "item_train_avg_rating": 0.0,
        "item_train_rating_std": 0.0,
        "item_train_pos_cnt": 0,
        "item_train_avg_text_len": 0.0,
    })


def check_feat_missing(df: DataFrame, name: str) -> None:
    """特征缺失校验：user_hist_cnt / item_train_cnt 任一为 null 即视为缺失行。"""
    missing = df.filter(
        F.col("user_hist_cnt").isNull() | F.col("item_train_cnt").isNull()
    ).count()
    logger.info("%s 特征缺失行数: %s", name, missing)
