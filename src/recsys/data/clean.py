"""数据清洗模块 —— 迁移自 legacy/Data_clean.py。

接口：
    run_clean(spark, cfg, logger) -> {"inter": df_final, "meta": meta_clean}

与 legacy 的差异（工程化改造）：
1. 原先「全局变量 df_final / meta_clean」隐式传递 → 改为函数显式返回；
2. 硬编码路径 → 全部来自 cfg；
3. print → logger；
4. 废弃的 spark.sql.autoBroadcastJoinThreshold → 仅保留 adaptive 版本；
5. 注释掉的 parquet 落地 → 交给 pipeline checkpoint 统一管理。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

from src.recsys.config import Config

logger = logging.getLogger("recsys.data.clean")


def clean_interaction(inter_df: DataFrame, dedup_user_item: bool) -> DataFrame:
    """交互表清洗（legacy 3.1 ~ 3.4）。"""
    n_before = inter_df.count()
    logger.info("清洗前交互行数: %s", f"{n_before:,}")

    # 3.1 过滤非法评分 / 非法时间戳
    inter_clean = inter_df.filter(
        F.col("user_id").isNotNull()
        & F.col("item_id").isNotNull()
        & F.col("rating").isNotNull()
        & F.col("rating").between(1, 5)
        & F.col("timestamp").isNotNull()
        & (F.col("timestamp") > 0)
    )
    n_after_filter = inter_clean.count()
    logger.info("① 过滤 null/非法rating/非法timestamp后: %s (剔除 %s 行)",
                f"{n_after_filter:,}", f"{n_before - n_after_filter:,}")

    # 3.2 提取 helpful 数组（EDA 已确认长度=2）
    inter_clean = (
        inter_clean
        .withColumn("helpful_useful", F.col("helpful")[0])
        .withColumn("helpful_total", F.col("helpful")[1])
        .drop("helpful")
    )

    # 3.3 (user,item) 重复交互去重：只保留 timestamp 最新一条
    if dedup_user_item:
        w = Window.partitionBy("user_id", "item_id").orderBy(F.desc("timestamp"))
        inter_clean = (
            inter_clean
            .withColumn("rn", F.row_number().over(w))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )
        n_after_dedup = inter_clean.count()
        logger.info("② (user,item)去重后: %s (剔除 %s 条重复)",
                    f"{n_after_dedup:,}", f"{n_after_filter - n_after_dedup:,}")

    # 3.4 文本空字符串标记（保留文本，供特征工程决定是否使用）
    inter_clean = inter_clean.withColumn(
        "reviewText_empty",
        F.when(F.length(F.col("reviewText")) == 0, 1).otherwise(0),
    )
    inter_clean.cache()
    n_clean = inter_clean.count()
    logger.info("✅ 交互表清洗完成，最终行数: %s (总剔除 %s 行, %.2f%%)",
                f"{n_clean:,}", f"{n_before - n_clean:,}",
                (n_before - n_clean) / n_before * 100 if n_before else 0)
    return inter_clean


def clean_meta(meta_raw: DataFrame) -> DataFrame:
    """meta 表清洗（legacy 4.1 ~ 4.6）。"""
    raw_count = meta_raw.count()
    logger.info("meta原始总行数: %s", f"{raw_count:,}")

    # 4.1 过滤损坏行
    # 说明：仅当 Spark 检测到损坏记录时 schema 中才会有 _corrupt_record 列
    # （例如原始 meta_Beauty.json 是 Python repr 格式时）；文件是标准 JSON、
    # 无损坏行时该列不存在，直接跳过过滤即可。
    corrupt_cols = [f.name for f in meta_raw.schema.fields if f.name == "_corrupt_record"]
    if corrupt_cols:
        meta_clean = meta_raw.filter(F.col("_corrupt_record").isNull())
        clean_count = meta_clean.count()
        logger.info("① 过滤_corrupt_record后: %s (剔除 %s 行)",
                    f"{clean_count:,}", f"{raw_count - clean_count:,}")
    else:
        meta_clean = meta_raw
        clean_count = raw_count
        logger.info("① 数据为合法 JSON（无损坏记录列），跳过 corrupt 过滤: %s 行",
                    f"{clean_count:,}")

    # 4.2 选择字段 + 提取嵌套结构
    meta_clean = meta_clean.select(
        F.col("asin").alias("item_id"),
        F.col("brand"),
        F.col("price"),
        F.col("categories"),
        F.col("salesRank.Beauty").alias("sales_rank_beauty"),
        F.col("related.also_bought").alias("also_bought_items"),
        F.col("related.also_viewed").alias("also_viewed_items"),
    )

    # 4.3 价格：字符串去 $ 转 double，非正/缺失 → -1 标记
    price_field_type = [
        f.dataType for f in meta_clean.schema.fields if f.name == "price"
    ][0]
    if isinstance(price_field_type, StringType):
        meta_clean = meta_clean.withColumn(
            "price",
            F.regexp_replace(F.regexp_replace("price", "\\$", ""), ",", "").cast("double"),
        )
    meta_clean = meta_clean.withColumn(
        "price",
        F.when(F.col("price").isNotNull() & (F.col("price") > 0), F.col("price"))
        .otherwise(-1.0),
    )

    # 4.4 品牌空值处理
    meta_clean = (
        meta_clean
        .withColumn("brand", F.trim(F.col("brand")))
        .withColumn("brand", F.when(F.col("brand") == "", "unknown").otherwise(F.col("brand")))
        .fillna({"brand": "unknown"})
    )

    # 4.5 类目解析：flatten 后取最后一级为主类目（带空数组保护）
    meta_clean = (
        meta_clean
        .withColumn("cat_list", F.flatten(F.col("categories")))
        .withColumn(
            "main_category",
            F.when(F.size(F.col("cat_list")) > 0, F.element_at(F.col("cat_list"), -1))
            .otherwise("unknown"),
        )
        .drop("categories", "cat_list")
    )

    # 4.6 meta 内部重复 asin 去重
    meta_clean = meta_clean.dropDuplicates(["item_id"])
    meta_clean.cache()

    logger.info("✅ meta表清洗完成，最终行数: %s", f"{meta_clean.count():,}")
    return meta_clean


def join_and_finalize(inter_clean: DataFrame, meta_clean: DataFrame,
                      keep_missing_meta: bool) -> DataFrame:
    """跨表关联（legacy 5.1）。"""
    df_final = inter_clean.join(meta_clean, on="item_id", how="left")
    df_final = df_final.withColumn(
        "meta_missing", F.when(F.col("brand").isNull(), 1).otherwise(0)
    )

    if keep_missing_meta:
        # 1) 基础常量字段 fillna
        df_final = df_final.fillna({
            "brand": "unknown",
            "main_category": "unknown",
            "price": -1.0,
            "sales_rank_beauty": -1,
        })
        # 2) 数组字段必须用 when 填空数组，不能放 fillna
        empty_arr = F.array().cast("array<string>")
        df_final = (
            df_final
            .withColumn(
                "also_bought_items",
                F.when(F.col("also_bought_items").isNull(), empty_arr)
                .otherwise(F.col("also_bought_items")),
            )
            .withColumn(
                "also_viewed_items",
                F.when(F.col("also_viewed_items").isNull(), empty_arr)
                .otherwise(F.col("also_viewed_items")),
            )
        )
    else:
        n_before_join = df_final.count()
        df_final = df_final.filter(F.col("meta_missing") == 0)
        logger.info("丢弃meta缺失交互: %s 行", f"{n_before_join - df_final.count():,}")

    df_final.cache()
    logger.info("关联后最终总行数: %s (meta缺失交互: %s)",
                f"{df_final.count():,}",
                f"{df_final.filter(F.col('meta_missing') == 1).count():,}")
    return df_final


def run_clean(spark: SparkSession, cfg: Config) -> dict:
    """清洗阶段入口。

    Returns:
        {"inter": df_final, "meta": meta_clean}
        - inter: 交互表左关联 meta 后的宽表（特征/切分/召回/精排的直接输入）
        - meta:  清洗后的商品元数据表
    """
    from src.recsys.io_utils import read_interaction_json

    logger.info("=" * 60)
    logger.info("【stage: clean】数据清洗")
    logger.info("=" * 60)

    inter_raw = read_interaction_json(spark, cfg.paths.inter_path)
    # 注意：clean_meta 需要保留 _corrupt_record 列做损坏行过滤，必须读原始 parsed df
    meta_raw = spark.read.json(str(cfg.paths.meta_path))

    inter_clean = clean_interaction(inter_raw, cfg.data_clean.dedup_user_item)
    meta_clean = clean_meta(meta_raw)
    df_final = join_and_finalize(inter_clean, meta_clean, cfg.data_clean.keep_missing_meta)

    # 展示与校验（legacy「清洗效果复核」）
    logger.info("清洗后独立用户数: %s | 独立商品数: %s",
                f"{df_final.select('user_id').distinct().count():,}",
                f"{df_final.select('item_id').distinct().count():,}")
    if cfg.data_clean.dedup_user_item:
        dup_cnt = (df_final.groupBy("user_id", "item_id").count()
                   .filter(F.col("count") > 1).count())
        logger.info("(user,item)重复对数量: %s (应为0)", dup_cnt)
    logger.info("非法评分(不在1-5)数量: %s",
                df_final.filter(~F.col("rating").between(1, 5)).count())

    inter_clean.unpersist()
    meta_raw.unpersist()
    return {"inter": df_final, "meta": meta_clean}
