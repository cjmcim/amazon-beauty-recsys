"""L2 行为与信号探查：评分、helpful、时间、文本、重复交互 —— 迁移自 legacy/EDA_L2.py。

输出：日志统计 + 评分分布 / 按年交互量 / helpful 赞同率 / 文本长度图。
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.recsys.config import Config
from src.recsys.eda.plots import save_bar, save_hist
from src.recsys.io_utils import read_interaction_json

logger = logging.getLogger("recsys.eda.l2")


def run(spark: SparkSession, cfg: Config) -> None:
    logger.info("=" * 70)
    logger.info("【EDA L2】行为与信号探查 — 评分、helpful、时间、文本、重复交互")
    logger.info("=" * 70)

    inter_df = read_interaction_json(spark, cfg.paths.inter_path).cache()
    _ = inter_df.count()  # 触发 cache

    # ---------- L2-1 评分信号 ----------
    logger.info("【L2-1 评分信号】")
    total_inter = inter_df.count()
    rating_dist = (
        inter_df.groupBy("rating")
        .agg(F.count("*").alias("cnt"))
        .withColumn("ratio", F.round(F.col("cnt") / total_inter * 100, 2))
        .orderBy("rating")
    )
    rating_dist.show()
    rating_pdf = rating_dist.toPandas()
    save_bar(rating_pdf, "rating", "cnt", "Rating distribution",
             cfg.paths.eda_plots, "l2_rating_dist.png", xlabel="rating", rot=0)

    # ---------- L2-2 helpful 投票 ----------
    logger.info("【L2-2 helpful 投票探查】")
    df_help = (
        inter_df.filter(F.col("helpful").isNotNull())
        .withColumn("help_useful", F.col("helpful")[0])
        .withColumn("help_total", F.col("helpful")[1])
    )
    df_help_rate = (
        df_help.filter(F.col("help_total") > 0)
        .withColumn("help_rate", F.col("help_useful") / F.col("help_total"))
    )
    df_help_rate.select(
        F.percentile_approx("help_rate", [0.25, 0.5, 0.75], 10000).alias("percentiles"),
        F.avg("help_rate").alias("avg_rate"),
    ).show()
    save_hist(df_help_rate.select("help_rate").toPandas()["help_rate"],
              "Helpful rate distribution", cfg.paths.eda_plots,
              "l2_helpful_rate.png")

    # ---------- L2-3 时间序列 ----------
    logger.info("【L2-3 交互时间时序探查】")
    time_stat = inter_df.select(
        F.min("timestamp").alias("ts_min"), F.max("timestamp").alias("ts_max"),
        F.from_unixtime(F.min("timestamp")).alias("dt_min"),
        F.from_unixtime(F.max("timestamp")).alias("dt_max"),
    ).collect()[0]
    logger.info("时间戳范围: %s ~ %s | 日期范围: %s ~ %s",
                time_stat["ts_min"], time_stat["ts_max"],
                time_stat["dt_min"], time_stat["dt_max"])

    year_cnt = (
        inter_df.withColumn("year", F.year(F.from_unixtime("timestamp")))
        .groupBy("year").count().orderBy("year")
    )
    logger.info("按年统计交互数量:")
    year_cnt.show()
    save_bar(year_cnt.toPandas(), "year", "count", "Interaction count by year",
             cfg.paths.eda_plots, "l2_year_inter.png", xlabel="year", rot=0)

    # 相邻交互时间差 < 60s 检测（疑似批量/刷评）
    w_user_time = Window.partitionBy("user_id").orderBy("timestamp")
    short_interval_cnt = (
        inter_df.withColumn("prev_ts", F.lag("timestamp").over(w_user_time))
        .withColumn("time_diff_sec", F.col("timestamp") - F.col("prev_ts"))
        .filter(F.col("time_diff_sec") < 60)
        .count()
    )
    logger.info("相邻交互间隔<60s行为对数: %s", f"{short_interval_cnt:,}")

    # ---------- L2-4 文本字段 ----------
    logger.info("【L2-4 评论文本字段探查】")
    df_textlen = inter_df.withColumn("review_len", F.length(F.col("reviewText"))) \
        .withColumn("summary_len", F.length(F.col("summary")))
    df_textlen.select(
        F.percentile_approx("review_len", [0, 0.25, 0.5, 0.75, 1], 10000).alias("review_len_quantile"),
        F.percentile_approx("summary_len", [0, 0.25, 0.5, 0.75, 1], 10000).alias("summary_len_quantile"),
    ).show()
    save_hist(df_textlen.select("review_len").toPandas()["review_len"],
              "ReviewText length distribution", cfg.paths.eda_plots,
              "l2_review_len.png")

    # ---------- L2-5 user-item 重复交互 ----------
    logger.info("【L2-5 user-item 重复交互对探查】")
    user_item_cnt = inter_df.groupBy("user_id", "item_id").agg(F.count("*").alias("ui_record_cnt"))
    ui_stat = user_item_cnt.select(
        F.count(F.when(F.col("ui_record_cnt") == 1, True)).alias("ui_once"),
        F.count(F.when(F.col("ui_record_cnt") > 1, True)).alias("ui_multi"),
        F.sum(F.when(F.col("ui_record_cnt") > 1, F.col("ui_record_cnt"))).alias("total_dup_rows"),
    ).collect()[0]
    # 注意：无重复交互时 SUM 返回 NULL（SQL 语义），必须兜底为 0
    dup_rows = ui_stat["total_dup_rows"] or 0
    logger.info("仅交互1次的(user,item)对数: %s | 多次交互对数: %s | 重复冗余行数: %s",
                f"{ui_stat['ui_once']:,}", f"{ui_stat['ui_multi']:,}",
                f"{dup_rows:,}")
    logger.info("重复交互 Top5 样例:")
    user_item_cnt.filter(F.col("ui_record_cnt") > 1) \
        .orderBy(F.col("ui_record_cnt").desc()).show(5)

    inter_df.unpersist()
    logger.info("L2 探查完成")
