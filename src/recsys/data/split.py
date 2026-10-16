"""时序数据集切分模块 —— 迁移自 legacy 主流水线「时序切分」段落。

接口：
    run_split(df_final, cfg, logger) -> dict
    {"train", "val", "test", "train_inter", "val_inter", "test_inter"}

设计要点（与 legacy 完全一致）：
1. 按 timestamp 分位数切 80/10/10；
2. val/test 只保留在 train 中出现过的用户（冷用户不评估）；
3. 一切统计口径均从 train 计算，后续特征/召回/精排只依赖本模块输出。
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.recsys.config import Config

logger = logging.getLogger("recsys.data.split")


def get_dataset_stats(df: DataFrame, name: str) -> None:
    """打印一个数据集的规模与时间范围（仅用于日志观察）。"""
    stats = df.agg(
        F.count("*").alias("sample_cnt"),
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
        F.countDistinct("user_id").alias("user_cnt"),
        F.countDistinct("item_id").alias("item_cnt"),
    ).collect()[0]
    logger.info("【%s】样本量: %s | 独立用户: %s | 独立商品: %s | 时间范围: %s ~ %s",
                name, f"{stats['sample_cnt']:,}", f"{stats['user_cnt']:,}",
                f"{stats['item_cnt']:,}", stats["min_ts"], stats["max_ts"])


def run_split(df_final: DataFrame, cfg: Config) -> dict:
    """时序切分阶段入口。"""
    logger.info("=" * 60)
    logger.info("【stage: split】时序数据集切分 (%.0f/%.0f/%.0f)",
                cfg.split.train_ratio * 100, cfg.split.val_ratio * 100,
                cfg.split.test_ratio * 100)
    logger.info("=" * 60)

    s = cfg.split
    quantiles = df_final.approxQuantile(
        "timestamp", [s.train_ratio, s.train_ratio + s.val_ratio], 0.01
    )
    train_cut_ts, val_cut_ts = quantiles[0], quantiles[1]
    logger.info("时间分位数切分阈值：train<=%.0f, val<=%.0f", train_cut_ts, val_cut_ts)

    train_all = df_final.filter(F.col("timestamp") <= train_cut_ts)
    val_all = df_final.filter(
        (F.col("timestamp") > train_cut_ts) & (F.col("timestamp") <= val_cut_ts)
    )
    test_all = df_final.filter(F.col("timestamp") > val_cut_ts)

    # 只有 train 中出现的用户才有资格进入 val/test 评估（防冷用户）
    train_user_df = train_all.select("user_id").distinct()
    train_user_count = train_user_df.count()
    logger.info("训练集独立用户数: %s", f"{train_user_count:,}")

    val = val_all.join(F.broadcast(train_user_df), on="user_id", how="inner")
    test = test_all.join(F.broadcast(train_user_df), on="user_id", how="inner")
    train = train_all

    get_dataset_stats(train, "训练集")
    get_dataset_stats(val, "验证集")
    get_dataset_stats(test, "测试集")

    # val/test 与 train 用户取 inner join 后，冷用户天然为 0（保留一行说明）
    logger.info("【校验】val/test 漏网冷用户数量: 0 / 0（构造上 inner join 保证）")

    train.cache()
    val.cache()
    test.cache()

    return {
        "train": train,
        "val": val,
        "test": test,
        # 接口别名：后续特征/召回/精排统一使用 *_inter 命名
        "train_inter": train,
        "val_inter": val,
        "test_inter": test,
    }
