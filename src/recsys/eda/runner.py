"""EDA 阶段编排：依次运行 L0~L3。"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession

from src.recsys.config import Config
from src.recsys.eda.l0_data_quality import run as l0
from src.recsys.eda.l1_matrix_profile import run as l1
from src.recsys.eda.l2_signal_probe import run as l2
from src.recsys.eda.l3_side_info import run as l3

logger = logging.getLogger("recsys.eda.runner")


def run_eda(spark: SparkSession, cfg: Config) -> dict:
    """EDA 阶段入口：L0 数据质量 → L1 矩阵画像 → L2 行为信号 → L3 侧信息。"""
    logger.info("=" * 70)
    logger.info("【stage: eda】探索性数据分析 L0~L3")
    logger.info("=" * 70)
    l0(spark, cfg)
    l1(spark, cfg)
    l2(spark, cfg)
    l3(spark, cfg)
    logger.info("EDA 完成，图表输出至: %s", cfg.paths.eda_plots)
    return {}  # EDA 不产出下游依赖的 DataFrame
