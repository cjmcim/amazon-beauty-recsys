"""数据读写与原始 JSON 解析工具。

职责：
1. 读取 Amazon Beauty 原始 JSON（交互表 / 元数据表），完成统一的字段别名映射；
2. 提供 Spark / pandas / json 等类型化读写原语，供 pipeline 的 checkpoint 复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.recsys.logging_utils import get_logger

logger = get_logger("io_utils")

# 交互表统一别名（与 legacy/EDA_L0.py 保持一致）
# 注意：用 selectExpr 字符串而非模块级 F.col —— PySpark 4.x 不允许在
# SparkContext 未激活时构造 Column，字符串表达式在运行时解析，兼容 3.x/4.x。
INTER_SELECT_EXPR = [
    "asin AS item_id",
    "reviewerID AS user_id",
    "CAST(overall AS float) AS rating",
    "CAST(unixReviewTime AS long) AS timestamp",
    "helpful",
    "reviewText",
    "summary",
]

# meta 表统一别名（保留原 EDA 探查所需的全部字段）
META_SELECT_EXPR = [
    "asin AS item_id",
    "brand",
    "categories",
    "description",
    "imUrl",
    "price",
    "related",
    "salesRank",
    "title",
]


def read_interaction_json(spark: SparkSession, path: str | Path) -> DataFrame:
    """读取交互表原始 JSON 并做统一别名映射（不做任何过滤/删除）。"""
    raw = spark.read.json(str(path))
    df = raw.selectExpr(*INTER_SELECT_EXPR)
    logger.info("读取交互表: %s", path)
    return df


def read_meta_json(spark: SparkSession, path: str | Path) -> tuple[DataFrame, int]:
    """读取 meta 原始 JSON，返回 (有效数据[已选列], 损坏行数)。

    与 legacy 一致：探查阶段只看损坏数量，清洗阶段再对原始数据过滤。
    ``_corrupt_record`` 列只在原始 parsed df 上存在，选列后即丢弃。
    """
    raw = spark.read.json(str(path))
    # 仅当 Spark 检测到损坏记录时 schema 中才会有 _corrupt_record 列
    # （标准 JSON 文件无损坏行时该列不存在，直接跳过过滤）
    corrupt_cols = [f.name for f in raw.schema.fields if f.name == "_corrupt_record"]
    if corrupt_cols:
        corrupt_cnt = raw.filter(F.col("_corrupt_record").isNotNull()).count()
        valid = raw.filter(F.col("_corrupt_record").isNull()).selectExpr(*META_SELECT_EXPR)
    else:
        corrupt_cnt = 0
        valid = raw.selectExpr(*META_SELECT_EXPR)
    logger.info("读取 meta 表: %s（损坏行 %d）", path, corrupt_cnt)
    return valid, corrupt_cnt


# ---------------------------------------------------------------------------
# 类型化读写原语（checkpoint 用）
# ---------------------------------------------------------------------------
def save_spark_df(df: DataFrame, path: str | Path) -> None:
    df.write.mode("overwrite").parquet(str(path))


def load_spark_df(spark: SparkSession, path: str | Path) -> DataFrame:
    return spark.read.parquet(str(path))


def save_pandas_df(df: Any, path: str | Path) -> None:
    df.to_pickle(str(path))


def load_pandas_df(path: str | Path) -> Any:
    import pandas as pd

    return pd.read_pickle(str(path))


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_lgb_model(model, path: Path) -> None:
    """LightGBM 模型持久化。"""
    model.save_model(str(path))


def load_lgb_model(path: Path):
    """LightGBM 模型加载。"""
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))
