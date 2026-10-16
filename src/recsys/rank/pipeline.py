"""精排阶段编排 —— 迁移自 legacy「精排融合版｜完整修复版」Part 2~6。

数据流：
    train/val 交互 + 召回候选 → 正负样本 → PIT 挂载特征
    → 复用召回 StringIndexer 编码 → LightGBM 训练 → 测试集候选打分评估
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.recsys.config import Config
from src.recsys.io_utils import save_lgb_model
from src.recsys.rank.features import (
    align_features,
    apply_cat_codes,
    apply_recall_indexer,
    split_feature_types,
    to_pandas_features,
)
from src.recsys.rank.pit import attach_features_pit
from src.recsys.rank.sampling import (
    build_pos,
    get_user_last_ts,
    sample_hard_neg,
    sample_random_neg,
)
from src.recsys.rank.train import score_test_candidates, train_lightgbm

logger = logging.getLogger("recsys.rank.pipeline")


def _preflight_check(**kwargs: Any) -> None:
    """前置校验：依赖产物必须全部就绪（替代 legacy 的 globals() 检查）。"""
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise RuntimeError(f"精排前置校验失败，缺失输入：{missing}")


def run_rank(cfg: Config, *, train_inter: DataFrame, val_inter: DataFrame,
             test_inter: DataFrame, item_feat: DataFrame, user_ts_feat: DataFrame,
             recall_candidate_df: DataFrame, val_recall_candidate: DataFrame,
             test_recall_candidate: DataFrame, brand_model, cate_model) -> Dict:
    """精排阶段入口。

    Returns:
        {"model", "imp", "metrics", "auc", "test_result", "sample_stats"}
    """
    logger.info("=" * 70)
    logger.info("【stage: rank】Hard负样本 + PIT 时序特征 + LightGBM 精排")
    logger.info("=" * 70)

    _preflight_check(
        train_inter=train_inter, val_inter=val_inter, test_inter=test_inter,
        item_feat=item_feat, user_ts_feat=user_ts_feat,
        recall_candidate_df=recall_candidate_df,
        val_recall_candidate=val_recall_candidate,
        test_recall_candidate=test_recall_candidate,
        brand_model=brand_model, cate_model=cate_model,
    )

    rc = cfg.rank
    seed = cfg.pipeline.seed

    # -------------------------- Part 2 构造正负样本 --------------------------
    logger.info("【Part 2】构造正负样本")
    train_pos = build_pos(train_inter, "训练集", rc.pos_rating_th)
    val_pos = build_pos(val_inter, "验证集", rc.pos_rating_th)

    train_user_last_ts = get_user_last_ts(train_inter)
    val_user_last_ts = get_user_last_ts(val_inter)
    train_all_user_item = train_inter.select("user_id", "item_id").distinct()
    val_all_user_item = val_inter.select("user_id", "item_id").distinct()
    item_pool = item_feat.select("item_id").distinct().cache()

    train_hard = sample_hard_neg(train_pos, recall_candidate_df, train_user_last_ts,
                                 rc.hard_neg_ratio, "训练集", seed)
    train_rand = sample_random_neg(train_pos, item_pool, train_user_last_ts,
                                   train_all_user_item, rc.random_neg_ratio, "训练集", seed)
    train_sample = (train_pos.unionByName(train_hard).unionByName(train_rand)
                    .dropDuplicates(["user_id", "item_id"]).cache())

    val_hard = sample_hard_neg(val_pos, val_recall_candidate, val_user_last_ts,
                               rc.hard_neg_ratio, "验证集", seed)
    val_rand = sample_random_neg(val_pos, item_pool, val_user_last_ts,
                                 val_all_user_item, rc.random_neg_ratio, "验证集", seed)
    val_sample = (val_pos.unionByName(val_hard).unionByName(val_rand)
                  .dropDuplicates(["user_id", "item_id"]).cache())

    # 样本构成（结构化返回，供实验运行报告记录数据量）
    sample_stats = {
        "train": {
            "pos": train_pos.count(), "hard_neg": train_hard.count(),
            "random_neg": train_rand.count(), "total": train_sample.count(),
        },
        "val": {
            "pos": val_pos.count(), "hard_neg": val_hard.count(),
            "random_neg": val_rand.count(), "total": val_sample.count(),
        },
    }

    logger.info("训练集: 正=%s hard负=%s random负=%s 合计=%s",
                f"{sample_stats['train']['pos']:,}", f"{sample_stats['train']['hard_neg']:,}",
                f"{sample_stats['train']['random_neg']:,}", f"{sample_stats['train']['total']:,}")
    logger.info("验证集: 正=%s hard负=%s random负=%s 合计=%s",
                f"{sample_stats['val']['pos']:,}", f"{sample_stats['val']['hard_neg']:,}",
                f"{sample_stats['val']['random_neg']:,}", f"{sample_stats['val']['total']:,}")

    # -------------------------- Part 3 PIT 特征挂载 --------------------------
    logger.info("【Part 3】PIT 特征挂载（最近快照匹配）")
    train_feat_df = attach_features_pit(train_sample, user_ts_feat, item_feat).cache()
    val_feat_df = attach_features_pit(val_sample, user_ts_feat, item_feat).cache()
    logger.info("训练特征表: %s | 验证特征表: %s",
                f"{train_feat_df.count():,}", f"{val_feat_df.count():,}")

    # -------------------------- Part 4 特征转换 --------------------------
    logger.info("【Part 4】特征转换，复用召回 StringIndexer 模型")
    feature_cols, cat_cols, num_cols = split_feature_types(train_feat_df)
    train_feat_df = apply_recall_indexer(train_feat_df, brand_model, cate_model)
    val_feat_df = apply_recall_indexer(val_feat_df, brand_model, cate_model)

    train_pdf = to_pandas_features(train_feat_df, feature_cols, cat_cols)
    val_pdf = to_pandas_features(val_feat_df, feature_cols, cat_cols)
    use_features, cat_in_use, X_train, X_val, y_train, y_val, cat_map = align_features(
        train_pdf, val_pdf, feature_cols, cat_cols
    )

    # -------------------------- Part 5 LightGBM 训练 --------------------------
    logger.info("【Part 5】LightGBM 训练")
    model, imp = train_lightgbm(
        X_train, y_train, X_val, y_val, cat_in_use,
        rc.lgb_params, rc.num_boost_round, rc.early_stopping_rounds,
    )

    # -------------------------- Part 6 测试集候选打分 --------------------------
    logger.info("【Part 6】测试集候选排序评估")
    test_user_last_ts = get_user_last_ts(train_inter)
    test_cand = (
        test_recall_candidate.select("user_id", "item_id")
        .join(test_user_last_ts, on="user_id", how="left")
        .withColumn("label", F.lit(0))
    )
    test_feat_df = attach_features_pit(test_cand, user_ts_feat, item_feat)
    test_feat_df = apply_recall_indexer(test_feat_df, brand_model, cate_model)
    # test 与 train 使用同一类别映射编码（cat_cols 原始类别名；cat_in_use 已数值化）
    test_pdf = to_pandas_features(test_feat_df, use_features, cat_cols)
    apply_cat_codes(test_pdf, cat_cols, cat_map)
    for c in use_features:
        if c not in test_pdf.columns:
            test_pdf[c] = 0

    score_out = score_test_candidates(model, test_pdf, test_inter,
                                      use_features, rc.pos_rating_th, rc.eval_topk)

    # 模型持久化（legacy: model.save_model("lgb_rank_model_fused.txt")）
    model_path = cfg.paths.model_dir / "lgb_rank_model.txt"
    save_lgb_model(model, model_path)
    logger.info("✅ 精排模型已保存: %s", model_path)
    imp.to_csv(cfg.paths.results / "feature_importance.csv", index=False, encoding="utf-8")

    return {
        "model": model,
        "imp": imp,
        "metrics": score_out["metrics"],
        "auc": score_out["auc"],
        "test_result": score_out["test_result"],
        "sample_stats": sample_stats,
    }
