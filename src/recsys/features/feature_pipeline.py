"""特征工程阶段编排 —— 串联 Part A~F。

对外输出（dict）：
    item_feat       物品静态特征表
    user_ts_feat    用户 PIT 快照表（hist_ts 版本，供精排最近快照匹配）
    user_feat_pit   用户 PIT 快照表（pit_time 版本，兼容命名）
    user_item_set   用户历史物品集合（供召回过滤）
    train_feat / val_feat / test_feat  三份带标签的特征表（仅本阶段内部校验用）
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.recsys.features.item_features import build_item_feature
from src.recsys.features.user_features import (
    attach_val_test_feat,
    build_train_feature,
    build_user_dynamic,
    build_user_item_set,
    build_user_ts_feat,
    check_feat_missing,
    fill_feat_missing,
)

logger = logging.getLogger("recsys.features.pipeline")


def run_feature(train: DataFrame, val: DataFrame, test: DataFrame,
                meta_clean: DataFrame) -> dict:
    """特征工程阶段入口。"""
    logger.info("=" * 60)
    logger.info("【stage: feature】特征工程（PIT 防泄露）")
    logger.info("=" * 60)

    # Part A 物品静态特征
    item_feat = build_item_feature(meta_clean, train)

    # Part B 用户动态特征（PIT 快照）
    user_dynamic = build_user_dynamic(train)
    user_ts_feat, user_feat_pit = build_user_ts_feat(user_dynamic)

    # Part C 用户历史物品集合
    user_item_set = build_user_item_set(train)

    # Part E 样本特征挂载
    train_feat = build_train_feature(user_dynamic, item_feat)
    val_feat = attach_val_test_feat(val, user_ts_feat, item_feat, "验证集")
    test_feat = attach_val_test_feat(test, user_ts_feat, item_feat, "测试集")

    train_feat.cache()
    val_feat.cache()
    test_feat.cache()
    logger.info("特征表行数 train=%s val=%s test=%s",
                f"{train_feat.count():,}", f"{val_feat.count():,}",
                f"{test_feat.count():,}")

    # Part F 缺失校验与填充
    check_feat_missing(train_feat, "训练集")
    check_feat_missing(val_feat, "验证集")
    check_feat_missing(test_feat, "测试集")
    train_feat = fill_feat_missing(train_feat)
    val_feat = fill_feat_missing(val_feat)
    test_feat = fill_feat_missing(test_feat)

    # 时间范围校验（应严格递增）
    for name, df in [("train_feat", train_feat), ("val_feat", val_feat),
                     ("test_feat", test_feat)]:
        r = df.select(F.min("timestamp"), F.max("timestamp")).collect()[0]
        logger.info("%s 时间范围: %s ~ %s", name, r[0], r[1])

    logger.info("最终特征列: %s", train_feat.columns)
    user_dynamic.unpersist()
    return {
        "item_feat": item_feat,
        "user_ts_feat": user_ts_feat,
        "user_feat_pit": user_feat_pit,
        "user_item_set": user_item_set,
        "train_feat": train_feat,
        "val_feat": val_feat,
        "test_feat": test_feat,
    }
