"""L0 数据理解：单表质量 + 跨表匹配校验 —— 迁移自 legacy/EDA_L0.py。

输出：日志中的统计结论 + outputs/eda/plots 下的缺失分布图。
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.recsys.config import Config
from src.recsys.eda.plots import save_bar
from src.recsys.io_utils import read_interaction_json, read_meta_json

logger = logging.getLogger("recsys.eda.l0")


def _null_count_table(df, table_name: str, out_dir) -> None:
    """全字段 NULL 缺失统计 + 缺失分布图。"""
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).toPandas().T.reset_index()
    null_counts.columns = ["field", "null_cnt"]
    logger.info("【%s】全字段 NULL 缺失统计:\n%s", table_name,
                null_counts.to_string(index=False))
    save_bar(null_counts, "field", "null_cnt", f"{table_name} NULL 缺失统计",
             out_dir, f"{table_name}_null_counts.png")


def run(spark: SparkSession, cfg: Config) -> None:
    logger.info("=" * 70)
    logger.info("【EDA L0】数据理解：单表质量 + 跨表匹配校验")
    logger.info("=" * 70)

    # ---------- L0-1 交互表 ----------
    inter_df = read_interaction_json(spark, cfg.paths.inter_path).cache()
    logger.info("交互表总行数: %s | 独立用户: %s | 独立商品: %s",
                f"{inter_df.count():,}",
                f"{inter_df.select('user_id').distinct().count():,}",
                f"{inter_df.select('item_id').distinct().count():,}")

    _null_count_table(inter_df, "inter", cfg.paths.eda_plots)

    dup_rows_cnt = inter_df.count() - inter_df.distinct().count()
    logger.info("完全重复交互行数量: %s", f"{dup_rows_cnt:,}")

    # 文本空串
    inter_df.select(
        F.count(F.when(F.col("reviewText") == "", True)).alias("reviewText_empty_str"),
        F.count(F.when(F.col("summary") == "", True)).alias("summary_empty_str"),
    ).show(truncate=False)

    # helpful 数组
    inter_df.select(
        F.count(F.when(F.col("helpful").isNull(), True)).alias("helpful_null"),
        F.count(F.when(F.size(F.col("helpful")) == 0, True)).alias("helpful_empty"),
        F.count(F.when(F.size(F.col("helpful")) == 2, True)).alias("helpful_valid_len2"),
        F.count(F.when((F.size(F.col("helpful")) != 0) & (F.size(F.col("helpful")) != 2),
                      True)).alias("helpful_abnormal_len"),
    ).show(truncate=False)

    # 数值字段
    inter_df.select(
        F.min("rating").alias("min_rating"), F.max("rating").alias("max_rating"),
        F.mean("rating").alias("avg_rating"),
        F.count(F.when((F.col("rating") < 1) | (F.col("rating") > 5), True))
         .alias("rating_out_of_range"),
        F.min("timestamp").alias("min_ts"), F.max("timestamp").alias("max_ts"),
        F.count(F.when(F.col("timestamp") <= 0, True)).alias("ts_invalid"),
        F.from_unixtime(F.min("timestamp")).alias("earliest_time"),
        F.from_unixtime(F.max("timestamp")).alias("latest_time"),
    ).show(truncate=False)

    # ---------- L0-2 meta 表 ----------
    meta_df, corrupt_cnt = read_meta_json(spark, cfg.paths.meta_path)
    logger.info("meta 损坏行(_corrupt_record非空): %s", f"{corrupt_cnt:,}")
    meta_df.cache()
    logger.info("meta 有效商品数: %s", f"{meta_df.count():,}")
    item_dup = meta_df.groupBy("item_id").count().filter(F.col("count") > 1)
    logger.info("meta 中重复 item_id 数量: %s", f"{item_dup.count():,}")

    _null_count_table(meta_df, "meta", cfg.paths.eda_plots)

    meta_df.select(
        F.count(F.when(F.col("title") == "", True)).alias("title_empty_str"),
        F.count(F.when(F.col("brand") == "", True)).alias("brand_empty_str"),
        F.count(F.when(F.col("description") == "", True)).alias("description_empty_str"),
    ).show(truncate=False)

    # categories 数组
    meta_df.select(
        F.count(F.when(F.col("categories").isNull(), True)).alias("categories_null"),
        F.count(F.when(F.size(F.col("categories")) == 0, True)).alias("categories_empty"),
        F.count(F.when(F.size(F.col("categories")) > 0, True)).alias("categories_valid"),
    ).show(truncate=False)
    logger.info("一级类目 Top20:")
    meta_df.select(F.explode("categories").alias("cat_path")) \
        .select(F.col("cat_path")[0].alias("main_cate")) \
        .groupBy("main_cate").count().orderBy(F.desc("count")).show(20, truncate=False)

    # salesRank / related 覆盖率
    meta_df.select(
        F.count(F.when(F.col("salesRank.Beauty").isNotNull(), True)).alias("has_beauty_rank"),
        F.count(F.when(F.col("salesRank.Beauty").isNull(), True)).alias("no_beauty_rank"),
    ).show(truncate=False)
    related_exprs = [
        F.count(F.when(F.col(f"related.{sub}").isNotNull(), True)).alias(f"{sub}_not_null")
        for sub in ["also_bought", "also_viewed", "bought_together", "buy_after_viewing"]
    ]
    meta_df.select(*related_exprs).show(truncate=False)

    # ---------- L0-3 跨表匹配 ----------
    reviews_asins = inter_df.select("item_id").distinct()
    meta_asins = meta_df.select("item_id").distinct()
    missing_meta = reviews_asins.join(meta_asins, on="item_id", how="left_anti")
    missing_meta_cnt = missing_meta.count()
    extra_meta = meta_asins.join(reviews_asins, on="item_id", how="left_anti")
    review_item_total = reviews_asins.count()

    logger.info("交互表独立商品总数: %s | meta表独立商品总数: %s",
                f"{review_item_total:,}", f"{meta_asins.count():,}")
    logger.info("交互有、meta缺失的商品数: %s | 商品元数据匹配率: %.2f%%",
                f"{missing_meta_cnt:,}",
                (1 - missing_meta_cnt / review_item_total) * 100)
    logger.info("meta有、但交互表无行为的商品数: %s", f"{extra_meta.count():,}")
    if missing_meta_cnt > 0:
        logger.info("交互表中有、但meta缺失的商品样例:")
        missing_meta.show(10)

    inter_df.unpersist()
    meta_df.unpersist()
