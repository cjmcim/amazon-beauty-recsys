"""召回评估：基于 val 正样本计算 Recall@K。"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def recall_at_k(recall_df: DataFrame, val_pos: DataFrame, k: int) -> float:
    """每用户取 recall_df 分数最高的 k 条，与 val 正样本求命中率均值。"""
    w = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    topk = recall_df.withColumn("rn", F.row_number().over(w)) \
        .filter(F.col("rn") <= k).drop("rn")

    hit_cnt = (
        topk.join(val_pos, on=["user_id", "item_id"], how="inner")
        .groupBy("user_id")
        .agg(F.count("*").alias("hit_cnt"))
    )
    val_users = val_pos.select("user_id").distinct()
    real_cnt = val_pos.groupBy("user_id").agg(F.count("*").alias("real_cnt"))
    per_user = (
        val_users.join(real_cnt, on="user_id", how="inner")
        .join(hit_cnt, on="user_id", how="left")
        .fillna({"hit_cnt": 0})
        .withColumn("recall", F.col("hit_cnt") / F.col("real_cnt"))
    )
    return per_user.agg(F.avg("recall")).collect()[0][0]
