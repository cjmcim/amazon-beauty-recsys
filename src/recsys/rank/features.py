"""特征转换 —— 迁移自 legacy 精排 Part 4。

主线逻辑：区分特征属性（类别/数值）→ 清洗 → 对齐训练/验证特征 → 模型输入 X, y。

特殊分支（防编码漂移）：
brand / main_category 在召回阶段已用 StringIndexer 编码，精排直接复用
召回 fit 好的模型 transform，不再重新 fit。其余类别特征交给树模型
的 pandas category dtype 处理。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

logger = logging.getLogger("recsys.rank.features")

# 明确不作为模型特征的列
_EXCLUDE_COLS = {"user_id", "item_id", "pit_time", "label", "timestamp", "rating", "hist_ts"}

# 明确的类别特征（含 flag 型 0/1 特征）
_LIKELY_CAT = {"brand", "main_category", "price_bin", "cate_id", "brand_id",
               "user_interact_cat_flag", "user_interact_brand_flag"}


def split_feature_types(feat_df: DataFrame) -> Tuple[List[str], List[str]]:
    """区分类别特征 / 数值特征（依据白名单 + schema 类型）。"""
    all_cols = set(feat_df.columns)
    feature_cols = sorted(list(all_cols - _EXCLUDE_COLS))
    cat_cols = [
        c for c in feature_cols
        if c in _LIKELY_CAT
        or feat_df.schema[c].dataType.simpleString() == "string"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


def apply_recall_indexer(df: DataFrame, brand_model, cate_model) -> DataFrame:
    """复用召回阶段 fit 好的 StringIndexer 做编码（不重新 fit）。"""
    df = df.fillna({"brand": "unknown", "main_category": "unknown"})
    df = brand_model.transform(df)
    df = cate_model.transform(df)
    return df


def to_pandas_features(spark_df: DataFrame, feature_cols: List[str],
                       cat_cols: List[str]) -> pd.DataFrame:
    """Spark 特征表 → pandas，做数值化/对数化/类别化清洗。

    - 类别特征: astype(str).fillna("unknown").astype("category")
    - 数值特征: to_numeric；cnt/count/pos_cnt/hist_cnt 类做 log1p
    """
    use_cols = ["user_id", "item_id", "label"] + [c for c in feature_cols if c in spark_df.columns]
    pdf = spark_df.select(*use_cols).toPandas()
    for c in feature_cols:
        if c not in pdf.columns:
            continue
        if c in cat_cols:
            pdf[c] = pdf[c].astype(str).fillna("unknown").astype("category")
        else:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
            if any(k in c for k in ["cnt", "count", "pos_cnt", "hist_cnt"]):
                pdf[c] = np.log1p(pdf[c].fillna(0))
            else:
                pdf[c] = pdf[c].fillna(0)
    return pdf


def apply_cat_codes(pdf: pd.DataFrame, cat_cols: List[str],
                    cat_map: dict | None = None) -> dict:
    """category 列统一编码并转 int32 数值 codes（规避 LightGBM 训练卡死）。

    背景：LightGBM 4.3.0 在 Windows 上，pandas category / int 编码的
    categorical 特征在「大数据 × 多类别（如 brand 3885 类）」下训练死锁
    （实测数值特征 50 轮 1.3s，含 categorical 则 CPU 零增长卡死）。
    方案：类别统一编码后转为 int codes 作为数值特征，不再走 categorical 路径。

    cat_map 为 None 时按 pdf 自身编码并返回映射；否则以 cat_map 为基准编码
    （train 先编码生成映射，val/test 复用同一映射保证编码一致）。
    """
    if cat_map is None:
        cat_map = {
            c: list(pdf[c].cat.categories)
            for c in cat_cols
            if c in pdf.columns and pdf[c].dtype.name == "category"
        }
    for c in cat_cols:
        if c in pdf.columns and pdf[c].dtype.name == "category":
            pdf[c] = pdf[c].cat.set_categories(
                cat_map.get(c, list(pdf[c].cat.categories))
            )
            # NaN/未见类别 → codes=-1，作为数值特征参与树分裂
            pdf[c] = pdf[c].cat.codes.astype("int32")
    return cat_map


def align_features(train_pdf: pd.DataFrame, val_pdf: pd.DataFrame,
                   feature_cols: List[str],
                   cat_cols: List[str]) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame,
                                                 pd.Series, pd.Series, dict]:
    """对齐训练/验证特征列，输出模型输入 X, y。

    Returns:
        (use_features, cat_in_use, X_train, X_val, y_train, y_val, cat_map)
        cat_in_use 恒为空列表（类别已数值化）；cat_map 供 test 阶段同编码。
    """
    use_features = [c for c in feature_cols if c in train_pdf.columns and c in val_pdf.columns]
    cat_in_use = [c for c in cat_cols if c in use_features]
    X_train, y_train = train_pdf[use_features], train_pdf["label"]
    X_val, y_val = val_pdf[use_features], val_pdf["label"]

    # ⚠️ 关键修复：统一 train/val 的 category 编码并转数值 codes。
    # 两表各自 astype("category") 编码不同（brand: train=0..3884, val=0..2060），
    # 且 LightGBM categorical 路径在本机大数据下死锁；统一编码后数值化规避。
    cat_map = apply_cat_codes(X_train, cat_in_use)
    apply_cat_codes(X_val, cat_in_use, cat_map)

    logger.info("数值特征数: %s | 类别特征(已统一编码转 codes): %s | 最终入模特征数: %s",
                len([c for c in use_features if c not in cat_in_use]),
                len(cat_in_use), len(use_features))
    logger.info("类别特征: %s", cat_in_use)
    return use_features, [], X_train, X_val, y_train, y_val, cat_map
