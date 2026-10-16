"""LightGBM 精排训练 —— 迁移自 legacy 精排 Part 5/6。"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.recsys.evaluate.metrics import ranking_metrics, roc_auc

logger = logging.getLogger("recsys.rank.train")


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   cat_in_use: List[str], lgb_params: Dict,
                   num_boost_round: int,
                   early_stopping_rounds: int) -> Tuple[lgb.Booster, pd.DataFrame]:
    """训练精排模型，返回 (model, feature_importance)。

    注：cat_in_use 恒为空列表（align_features 已把类别特征统一编码为
    数值 codes，规避 LightGBM 4.3.0 Windows 大数据 categorical 卡死）。
    """
    lgb_train = lgb.Dataset(X_train, label=y_train,
                            categorical_feature=cat_in_use, free_raw_data=True)
    lgb_val = lgb.Dataset(X_val, label=y_val,
                          categorical_feature=cat_in_use,
                          reference=lgb_train, free_raw_data=True)

    model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    logger.info("best_iteration: %s", model.best_iteration)
    try:
        logger.info("best val auc: %.4f", model.best_score["val"]["auc"])
    except Exception:
        pass

    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info("特征重要性 Top 15:\n%s", imp.head(15).to_string(index=False))
    return model, imp


def score_test_candidates(model: lgb.Booster, test_pdf: pd.DataFrame,
                          test_inter: DataFrame, use_features: List[str],
                          pos_rating_th: float,
                          eval_topk: List[int]) -> Dict:
    """测试集候选打分 + 离线指标 + AUC。"""
    test_pdf["score"] = model.predict(test_pdf[use_features],
                                      num_iteration=model.best_iteration)

    test_label = (
        test_inter.filter(F.col("rating") >= pos_rating_th)
        .select("user_id", "item_id").dropDuplicates().toPandas()
    )
    test_label["label"] = 1

    # ⚠️ test_pdf 自带 label=0（候选集构造时 lit(0)），merge 前先移除，
    # 否则与 test_label 的 label 列冲突产生 label_x/label_y 导致 KeyError
    test_result = test_pdf.drop(columns=["label"], errors="ignore").merge(
        test_label, on=["user_id", "item_id"], how="left"
    )
    test_result["label"] = test_result["label"].fillna(0).astype(int)

    metrics = ranking_metrics(test_result, ks=eval_topk,
                              score_col="score", label_col="label")
    auc = None
    try:
        auc = roc_auc(test_result["label"], test_result["score"])
    except Exception as e:
        logger.warning("AUC 计算失败: %s", e)

    logger.info("精排离线指标（测试集召回候选）:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
    if auc is not None:
        logger.info("  Candidate AUC: %.4f", auc)
    return {"metrics": metrics, "auc": auc, "test_result": test_result}
