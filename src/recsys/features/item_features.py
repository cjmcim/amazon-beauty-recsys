"""物品静态特征 —— 迁移自 legacy 特征工程 Part A。

item_feat = meta 属性 + train 上统计的交互热度/评分/文本长度。
只统计 train，保证 val/test 评估无泄露。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger("recsys.features.item")


def build_item_feature(meta_clean: DataFrame, train: DataFrame) -> DataFrame:
    """构建物品特征表（item_id 粒度）。"""
    logger.info("=" * 60)
    logger.info("【Part A: 物品静态特征表 item_feat】")
    logger.info("=" * 60)

    item_attr = meta_clean.select(
        "item_id",
        "brand",
        "main_category",
        "price",
        "sales_rank_beauty",
        "also_bought_items",
        "also_viewed_items",
    )

    item_train_stat = (
        train.select("item_id", "rating", "reviewText")
        .groupBy("item_id")
        .agg(
            F.count("*").alias("item_train_cnt"),
            F.avg("rating").alias("item_train_avg_rating"),
            F.stddev("rating").alias("item_train_rating_std"),
            F.count(F.when(F.col("rating") >= 4, True)).alias("item_train_pos_cnt"),
            F.avg(F.length(F.col("reviewText"))).alias("item_train_avg_text_len"),
        )
    )

    item_feat = item_attr.join(item_train_stat, on="item_id", how="left")
    item_feat = item_feat.fillna({
        "item_train_cnt": 0,
        "item_train_avg_rating": 0.0,
        "item_train_rating_std": 0.0,
        "item_train_pos_cnt": 0,
        "item_train_avg_text_len": 0.0,
        "price": -1.0,
        "sales_rank_beauty": -1,
        "brand": "unknown",
        "main_category": "unknown",
    })
    empty_arr = F.array().cast("array<string>")
    item_feat = (
        item_feat
        .withColumn("also_bought_items",
                    F.when(F.col("also_bought_items").isNull(), empty_arr)
                    .otherwise(F.col("also_bought_items")))
        .withColumn("also_viewed_items",
                    F.when(F.col("also_viewed_items").isNull(), empty_arr)
                    .otherwise(F.col("also_viewed_items")))
    )
    item_feat.cache()
    logger.info("物品特征表商品数: %s", f"{item_feat.count():,}")
    return item_feat
