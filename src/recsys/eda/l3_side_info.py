"""L3 侧信息与偏差探查：商品元信息 & 交互联合分析 —— 迁移自 legacy/EDA_L3.py。

核心性能优化（保留）：只做一次 inter×meta join，后续全部复用。
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.recsys.config import Config
from src.recsys.eda.plots import save_bar
from src.recsys.io_utils import read_interaction_json, read_meta_json

logger = logging.getLogger("recsys.eda.l3")


def run(spark: SparkSession, cfg: Config) -> None:
    logger.info("=" * 70)
    logger.info("【EDA L3】侧信息与偏差探查 — 商品元信息 & 交互联合分析")
    logger.info("=" * 70)

    inter_df = read_interaction_json(spark, cfg.paths.inter_path).cache()
    meta_df, _ = read_meta_json(spark, cfg.paths.meta_path)
    meta_df.cache()
    _ = inter_df.count(), meta_df.count()

    # 只执行一次 join，后续全部复用，消除重复 shuffle
    # 用 on="item_id" 让 Spark 合并 join 键，避免出现两个 item_id 列导致歧义
    item_meta_join_df = inter_df.join(meta_df, on="item_id", how="left").cache()
    _ = item_meta_join_df.count()

    # ---------- L3-1 关键字段覆盖率 ----------
    logger.info("【L3-1 商品元表关键字段覆盖率】")
    meta_key_cols = ["item_id", "brand", "price", "categories", "salesRank"]
    meta_df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c + "_null_cnt") for c in meta_key_cols]
    ).show()

    inter_item_set = inter_df.select("item_id").distinct()
    hit_meta_item = meta_df.join(inter_item_set, on="item_id", how="inner").count()
    inter_distinct_item_cnt = inter_item_set.count()
    logger.info("meta总商品数: %s | 交互可匹配商品数: %s | 匹配率: %.2f%%",
                f"{meta_df.count():,}", f"{hit_meta_item:,}",
                hit_meta_item / inter_distinct_item_cnt * 100)

    # ---------- L3-2 price ----------
    logger.info("【L3-2 price 价格分布探查】")
    df_price = (
        meta_df.filter(F.col("price").isNotNull())
        .withColumn("price_val", F.col("price").cast("double"))
        .filter(F.col("price_val") > 0)
    )
    df_price.select(
        F.percentile_approx("price_val", [0.0, 0.25, 0.5, 0.75, 1.0], 10000).alias("price_quantile"),
        F.avg("price_val").alias("price_avg"),
    ).show()

    price_bin_df = df_price.withColumn(
        "price_bin",
        F.when(F.col("price_val") < 10, "0-10")
        .when(F.col("price_val") < 30, "10-30")
        .when(F.col("price_val") < 80, "30-80")
        .otherwise(">80"),
    )
    price_bin_inter = item_meta_join_df.join(
        price_bin_df.select("item_id", "price_bin"), on="item_id", how="inner"
    )
    price_bin_cnt = price_bin_inter.groupBy("price_bin").count().orderBy("price_bin")
    logger.info("不同价格档位总交互数量:")
    price_bin_cnt.show()
    save_bar(price_bin_cnt.toPandas(), "price_bin", "count",
             "Interaction count by price bin", cfg.paths.eda_plots,
             "l3_price_bin.png", xlabel="price_bin", rot=0)

    # ---------- L3-3 brand ----------
    logger.info("【L3-3 品牌分布探查】")
    brand_cnt = meta_df.filter(
        F.col("brand").isNotNull() & (F.trim(F.col("brand")) != "")
    ).select(F.countDistinct("brand")).collect()[0][0]
    logger.info("元表有效品牌总数量: %s", brand_cnt)

    brand_agg = (
        item_meta_join_df.filter(F.col("brand").isNotNull() & (F.trim(F.col("brand")) != ""))
        .groupBy("brand").count().withColumnRenamed("count", "inter_cnt")
    )
    logger.info("交互量 Top10 品牌:")
    brand_agg.orderBy(F.col("inter_cnt").desc()).show(10)
    top10_brand_inter = brand_agg.limit(10).select(F.sum("inter_cnt")).collect()[0][0] or 0
    all_brand_inter = brand_agg.select(F.sum("inter_cnt")).collect()[0][0] or 0
    if all_brand_inter:
        logger.info("Top10品牌交互: %s | 占全部品牌交互: %.2f%%",
                    f"{top10_brand_inter:,}", top10_brand_inter / all_brand_inter * 100)
    else:
        logger.info("品牌交互总量为 0，跳过占比计算")
    save_bar(brand_agg.orderBy(F.col("inter_cnt").desc()).limit(10).toPandas(),
             "brand", "inter_cnt", "Top10 brand interaction",
             cfg.paths.eda_plots, "l3_brand_top10.png", xlabel="brand", rot=45)

    # ---------- L3-4 categories ----------
    logger.info("【L3-4 类目 categories 探查】")
    df_cat = (
        meta_df.filter(F.col("categories").isNotNull() & (F.size(F.col("categories")) > 0))
        .withColumn("cat_list", F.flatten(F.col("categories")))
        # 保护：flatten 后数组 size>0 才取最后一级，否则置 null
        .withColumn("main_cat",
                    F.when(F.size(F.col("cat_list")) > 0, F.element_at(F.col("cat_list"), -1)))
        .filter(F.col("main_cat").isNotNull())
    )
    cat_inter = item_meta_join_df.join(
        df_cat.select("item_id", "main_cat"), on="item_id", how="inner"
    )
    cat_agg = cat_inter.groupBy("main_cat").count().withColumnRenamed("count", "inter_cnt")
    logger.info("各一级类目交互数量 Top15:")
    cat_agg.orderBy(F.col("inter_cnt").desc()).show(15)
    save_bar(cat_agg.orderBy(F.col("inter_cnt").desc()).limit(15).toPandas(),
             "main_cat", "inter_cnt", "Top15 category interaction",
             cfg.paths.eda_plots, "l3_category_top15.png", xlabel="main_cat", rot=45)

    # ---------- L3-5 salesRank ----------
    logger.info("【L3-5 salesRank 销量排名探查】")
    df_rank = (
        meta_df.filter(F.col("salesRank").isNotNull())
        .withColumn("rank_value", F.col("salesRank")["Beauty"].cast("long"))
        .filter(F.col("rank_value") > 0)
    )
    df_rank.select(
        F.percentile_approx("rank_value", [0.25, 0.5, 0.75], 10000).alias("rank_quantile")
    ).show()
    rank_bin = df_rank.withColumn(
        "rank_bin",
        F.when(F.col("rank_value") <= 1000, "top_1000")
        .when(F.col("rank_value") <= 10000, "1000-10000")
        .otherwise(">10000"),
    )
    rank_bin_inter = item_meta_join_df.join(
        rank_bin.select("item_id", "rank_bin"), on="item_id", how="inner"
    )
    rank_bin_cnt = rank_bin_inter.groupBy("rank_bin").count().orderBy("rank_bin")
    logger.info("不同销量排名档位交互数量:")
    rank_bin_cnt.show()
    save_bar(rank_bin_cnt.toPandas(), "rank_bin", "count",
             "Interaction count by sales rank bin", cfg.paths.eda_plots,
             "l3_rank_bin.png", xlabel="rank_bin", rot=0)

    # ---------- L3-6 related ----------
    logger.info("【L3-6 related 关联商品探查】")
    df_related = (
        meta_df.filter(F.col("related").isNotNull())
        .withColumn("has_also_viewed", F.size(F.col("related").getField("also_viewed")) > 0)
        .withColumn("has_also_bought", F.size(F.col("related").getField("also_bought")) > 0)
    )
    df_related.select(
        F.count(F.when(F.col("has_also_viewed"), True)).alias("has_also_viewed_cnt"),
        F.count(F.when(F.col("has_also_bought"), True)).alias("has_also_bought_cnt"),
    ).show()

    meta_df.unpersist()
    inter_df.unpersist()
    item_meta_join_df.unpersist()
    logger.info("L3 探查完成")
