"""召回层引擎 —— 迁移自 legacy「召回层完整代码【修复定稿版】」。

策略（与 legacy 完全一致）：
1. 所有热度 / 共现 / 相似度素材只从 train 计算，val/test 仅做评估 → 严格防泄露；
2. 多路召回：热门召回（Top HOT_TOP_K）+ ItemCF 召回（相似度 TopK）；
3. 多路合并：同 item 取最高分，记录来源，全局截断 FINAL_RECALL_K；
4. 拟合 StringIndexer（brand / main_category），精排复用防编码漂移。

对外输出（dict）：
    recall_candidate_df / val_recall_candidate / test_recall_candidate
    brand_model / cate_model
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

from src.recsys.config import RecallConfig
from src.recsys.recall.evaluate import recall_at_k

logger = logging.getLogger("recsys.recall.engine")


class RecallBase:
    """召回全局基础素材（全部只从 train 计算，只算一次）。"""

    def __init__(self, train: DataFrame, cfg: RecallConfig):
        self.cfg = cfg
        # 训练集正反馈（rating>=4）
        self.train_pos = (
            train.filter(F.col("rating") >= 4)
            .select("user_id", "item_id", "timestamp")
            .dropDuplicates(["user_id", "item_id"])
            .cache()
        )
        self.user_hist_items = (
            self.train_pos.groupBy("user_id")
            .agg(F.collect_set("item_id").alias("hist_items"))
        )
        # 热门召回素材
        self.item_hot = (
            self.train_pos.groupBy("item_id")
            .agg(F.count("*").alias("hot_score"))
            .orderBy(F.desc("hot_score"))
            .cache()
        )
        self.hot_items_base = (
            self.item_hot.limit(cfg.hot_top_k * 2)
            .select(F.col("item_id").alias("rec_item"), "hot_score")
        )
        # ItemCF 相似度素材（余弦式：cooc / sqrt(pop_i * pop_j)）
        self.item_sim_base = self._build_item_sim()

    def _build_item_sim(self) -> DataFrame:
        cfg = self.cfg
        user_items = self.train_pos.groupBy("user_id") \
            .agg(F.collect_set("item_id").alias("items"))
        cooc = (
            user_items.select("user_id", F.explode("items").alias("item_i"))
            .join(user_items.select("user_id", F.explode("items").alias("item_j")),
                  on="user_id")
            .filter(F.col("item_i") < F.col("item_j"))
            .groupBy("item_i", "item_j")
            .agg(F.count("*").alias("cooc_cnt"))
            .filter(F.col("cooc_cnt") >= cfg.min_cooc)
        )
        cooc_sym = cooc.select(
            F.col("item_j").alias("item_i"), F.col("item_i").alias("item_j"), "cooc_cnt"
        )
        item_sim_base = cooc.unionByName(cooc_sym)

        item_pop = self.item_hot.select(
            F.col("item_id").alias("item_i"), F.col("hot_score").alias("pop_i")
        )
        item_sim_base = (
            item_sim_base
            .join(item_pop, on="item_i", how="left")
            .join(item_pop.select(F.col("item_i").alias("item_j"),
                                  F.col("pop_i").alias("pop_j")),
                  on="item_j", how="left")
            .withColumn("pop_i", F.when(F.col("pop_i").isNull(), 1).otherwise(F.col("pop_i")))
            .withColumn("pop_j", F.when(F.col("pop_j").isNull(), 1).otherwise(F.col("pop_j")))
            .withColumn("sim_score", F.col("cooc_cnt") / F.sqrt(F.col("pop_i") * F.col("pop_j")))
        )
        item_sim_base.cache()
        logger.info("物品相似对数量: %s", f"{item_sim_base.count():,}")
        return item_sim_base

    def fit_indexers(self, train: DataFrame) -> None:
        """StringIndexer 词典只在 train 拟合，精排复用（防编码漂移）。"""
        self.brand_model = StringIndexer(
            inputCol="brand", outputCol="brand_id", handleInvalid="keep"
        ).fit(train.select("brand").fillna("unknown"))
        self.cate_model = StringIndexer(
            inputCol="main_category", outputCol="cate_id", handleInvalid="keep"
        ).fit(train.select("main_category").fillna("unknown"))
        logger.info("brand 编码数量(含未知桶): %s | main_category: %s",
                    len(self.brand_model.labels), len(self.cate_model.labels))

    def unpersist(self) -> None:
        for df in (self.train_pos, self.item_hot, self.item_sim_base):
            df.unpersist()


def run_recall_for_users(user_df: DataFrame, base: RecallBase) -> DataFrame:
    """对一批 user 生成多路合并召回候选。

    Args:
        user_df: 至少包含 user_id 列的 DataFrame（train/val/test 用户均可）
        base: RecallBase（train 素材）
    Returns:
        user_id, item_id, recall_score, recall_sources
    """
    cfg = base.cfg
    users = user_df.select("user_id").distinct()

    # ========== 热门召回 ==========
    hot_recall = users.crossJoin(F.broadcast(base.hot_items_base))
    hot_recall = (
        hot_recall.join(base.user_hist_items, on="user_id", how="left")
        .filter(~F.array_contains(F.col("hist_items"), F.col("rec_item"))
                | F.col("hist_items").isNull())
        .select("user_id", "rec_item",
                F.col("hot_score").alias("recall_score"),
                F.lit("hot").alias("recall_type"))
    )
    w_hot = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    hot_recall = (
        hot_recall.withColumn("rn", F.row_number().over(w_hot))
        .filter(F.col("rn") <= cfg.hot_top_k)
        .drop("rn")
    )

    # ========== ItemCF 召回 ==========
    user_hist_item_df = base.train_pos.select(
        "user_id", F.col("item_id").alias("hist_item")
    ).join(users, on="user_id")
    itemcf_cand = (
        user_hist_item_df.join(
            base.item_sim_base.select(
                F.col("item_i").alias("hist_item"),
                F.col("item_j").alias("rec_item"), "sim_score"
            ),
            on="hist_item", how="inner",
        )
        .groupBy("user_id", "rec_item")
        .agg(F.sum("sim_score").alias("recall_score"))
    )
    itemcf_cand = (
        itemcf_cand.join(base.user_hist_items, on="user_id", how="left")
        .filter(~F.array_contains(F.col("hist_items"), F.col("rec_item"))
                | F.col("hist_items").isNull())
        .select("user_id", "rec_item", "recall_score",
                F.lit("itemcf").alias("recall_type"))
    )
    w_cf = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    itemcf_recall = (
        itemcf_cand.withColumn("rn", F.row_number().over(w_cf))
        .filter(F.col("rn") <= cfg.itemcf_top_k)
        .drop("rn")
    )

    # ========== 多路合并：同 item 取最高分，记录来源，全局截断 ==========
    recall_all = itemcf_recall.unionByName(hot_recall)
    recall_merged = (
        recall_all.groupBy("user_id", "rec_item")
        .agg(
            F.max("recall_score").alias("recall_score"),
            F.concat_ws(",", F.collect_set("recall_type")).alias("recall_sources"),
        )
    )
    w_final = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    recall_out = (
        recall_merged.withColumn("rn", F.row_number().over(w_final))
        .filter(F.col("rn") <= cfg.final_recall_k)
        .drop("rn")
        .withColumnRenamed("rec_item", "item_id")
    )
    recall_out.cache()
    return recall_out


def run_recall(train: DataFrame, val: DataFrame, test: DataFrame,
               cfg: RecallConfig) -> dict:
    """召回阶段入口。"""
    logger.info("=" * 70)
    logger.info("【stage: recall】ItemCF + 热门召回 + 多路合并 + 评估")
    logger.info("=" * 70)

    base = RecallBase(train, cfg)
    logger.info("训练集正反馈交互数: %s", f"{base.train_pos.count():,}")
    logger.info("有热度物品数: %s", f"{base.item_hot.count():,}")
    base.fit_indexers(train)

    recall_candidate_df = run_recall_for_users(train.select("user_id").distinct(), base)
    val_recall_candidate = run_recall_for_users(val.select("user_id").distinct(), base)
    test_recall_candidate = run_recall_for_users(test.select("user_id").distinct(), base)

    logger.info("三份召回候选行数: train=%s val=%s test=%s",
                f"{recall_candidate_df.count():,}",
                f"{val_recall_candidate.count():,}",
                f"{test_recall_candidate.count():,}")

    # ---------- 召回评估（基于 val 正样本 Recall@K） ----------
    val_pos = (
        val.filter(F.col("rating") >= 4)
        .select("user_id", "item_id")
        .dropDuplicates(["user_id", "item_id"])
        .withColumn("label", F.lit(1))
    )
    logger.info("召回 Recall@K 评估（基于验证集 val 正反馈）:")
    for k in cfg.eval_k_list:
        if k <= cfg.final_recall_k:
            logger.info("  Recall@%-4d= %.4f", k, recall_at_k(val_recall_candidate, val_pos, k))

    hit_user_cnt = (
        val_recall_candidate.join(val_pos, on=["user_id", "item_id"], how="inner")
        .select("user_id").distinct().count()
    )
    total_val_user = val_pos.select("user_id").distinct().count()
    logger.info("验证集有正反馈用户数: %s | 召回至少命中1个的用户数: %s | 占比: %.2f%%",
                f"{total_val_user:,}", f"{hit_user_cnt:,}",
                hit_user_cnt / max(total_val_user, 1) * 100)

    base.unpersist()
    return {
        "recall_candidate_df": recall_candidate_df,
        "val_recall_candidate": val_recall_candidate,
        "test_recall_candidate": test_recall_candidate,
        "brand_model": base.brand_model,
        "cate_model": base.cate_model,
    }
