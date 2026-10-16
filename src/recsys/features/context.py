"""上下文特征：从 timestamp 提取年/月（轻量纯函数）。"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_context_features(df: DataFrame) -> DataFrame:
    """追加 interact_year / interact_month 两列。"""
    return (
        df
        .withColumn("interact_year", F.year(F.from_unixtime("timestamp")))
        .withColumn("interact_month", F.month(F.from_unixtime("timestamp")))
    )
