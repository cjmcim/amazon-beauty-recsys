# ==============================================================
# 时序切分 → 特征工程 → 召回 → 精排
# 约束：全部相似度、热度统计只使用train集合，val/test只做评估，严格PIT防泄露
# ==============================================================

import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ===================== 全局配置区 =====================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# 召回配置
ITEMCF_TOP_K = 100
HOT_TOP_K = 50
FINAL_RECALL_K = 300
MIN_COOC = 3
K_LIST = [5, 10, 20, 50, 100, 200]

# 精排配置
HARD_NEG_RATIO = 3
RANDOM_NEG_RATIO = 1
POS_RATING_TH = 4.0
EVAL_TOPK = [5, 10, 20]
RANDOM_SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1) < 1e-6, "比例总和必须=1"


# ==============================================================================
# 【模块】时序数据集切分
# ==============================================================================
print("=" * 60)
print("【时序数据集切分｜保留性能优化 + 保证时间顺序正确】")
print("=" * 60)

quantiles = df_final.approxQuantile("timestamp", [TRAIN_RATIO, TRAIN_RATIO+VAL_RATIO], 0.01)
train_cut_ts = quantiles[0]
val_cut_ts = quantiles[1]
print(f"时间分位数切分阈值：训练集截止时间戳={train_cut_ts:.0f}, 验证集截止时间戳={val_cut_ts:.0f}")

train_all = df_final.filter(F.col("timestamp") <= train_cut_ts)
val_all   = df_final.filter( (F.col("timestamp") > train_cut_ts) & (F.col("timestamp") <= val_cut_ts) )
test_all  = df_final.filter(F.col("timestamp") > val_cut_ts)

train_user_df = train_all.select("user_id").distinct()
train_user_count = train_user_df.count()
print(f"训练集独立用户数: {train_user_count:,}")

val = val_all.join(F.broadcast(train_user_df), on="user_id", how="inner")
test = test_all.join(F.broadcast(train_user_df), on="user_id", how="inner")
train = train_all

def get_dataset_stats(df, name):
    stats = df.agg(
        F.count("*").alias("sample_cnt"),
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
        F.countDistinct("user_id").alias("user_cnt"),
        F.countDistinct("item_id").alias("item_cnt")
    ).collect()[0]
    print(f"\n【{name}】")
    print(f"  样本量: {stats['sample_cnt']:,} | 独立用户: {stats['user_cnt']:,} | 独立商品: {stats['item_cnt']:,}")
    print(f"  时间范围: {stats['min_ts']} ~ {stats['max_ts']}")
    return stats

train_stats = get_dataset_stats(train, "训练集")
val_stats = get_dataset_stats(val, "验证集")
test_stats = get_dataset_stats(test, "测试集")

val_new_user = val.join(F.broadcast(train_user_df), on="user_id", how="left_anti").count()
test_new_user = test.join(F.broadcast(train_user_df), on="user_id", how="left_anti").count()
print(f"\n【校验】验证集/测试集 漏网冷用户数量: {val_new_user} / {test_new_user}")

train.cache()
val.cache()
test.cache()

# ✅【接口对齐关键】生成全局标准交互别名，供后续特征、召回、精排直接使用
train_inter = train
val_inter   = val
test_inter  = test

print("\n✅ 时序切分完成！已生成别名 train_inter / val_inter / test_inter")


# ==============================================================================
# 【模块】特征工程完整代码（修复版 PIT快照）
# 输出：item_feat、user_ts_feat、user_feat_pit、train_feat、val_feat、test_feat
# ==============================================================================
print("=" * 60)
print("【Part A: 物品静态特征表 item_feat】")
print("=" * 60)

item_attr = meta_clean.select(
    "item_id",
    "brand",
    "main_category",
    "price",
    "sales_rank_beauty",
    "also_bought_items",
    "also_viewed_items"
)

item_train_stat = train.select(
    "item_id", "rating", "reviewText"
).groupBy("item_id").agg(
    F.count("*").alias("item_train_cnt"),
    F.avg("rating").alias("item_train_avg_rating"),
    F.stddev("rating").alias("item_train_rating_std"),
    F.count(F.when(F.col("rating") >= 4, True)).alias("item_train_pos_cnt"),
    F.avg(F.length(F.col("reviewText"))).alias("item_train_avg_text_len")
)

item_feat = item_attr.join(item_train_stat, on="item_id", how="left")
item_feat = item_feat.fillna({
    "item_train_cnt": 0,
    "item_train_avg_rating": 0.0,
    "item_train_rating_std": 0.0,
    "item_train_pos_cnt": 0,
    "item_train_avg_text_len": 0.0,
    "price": -1.0,
    "sales_rank_beauty": -1,
    "brand": "unknown",
    "main_category": "unknown"
})
empty_arr = F.array().cast("array<string>")
item_feat = item_feat \
    .withColumn("also_bought_items",
                F.when(F.col("also_bought_items").isNull(), empty_arr)
                 .otherwise(F.col("also_bought_items"))) \
    .withColumn("also_viewed_items",
                F.when(F.col("also_viewed_items").isNull(), empty_arr)
                 .otherwise(F.col("also_viewed_items")))
item_feat.cache()
print(f"物品特征表商品数: {item_feat.count():,}")

print("=" * 60)
print("【Part B: 用户动态特征（PIT 快照，含交叉集合）】")
print("=" * 60)

w_pit = Window.partitionBy("user_id") \
    .orderBy("timestamp") \
    .rowsBetween(Window.unboundedPreceding, -1)

user_dynamic = train.select(
    "user_id", "item_id", "timestamp", "rating", "reviewText",
    "brand", "main_category"
).withColumn("user_hist_cnt", F.count("item_id").over(w_pit)) \
 .withColumn("user_avg_rating", F.avg("rating").over(w_pit)) \
 .withColumn("user_rating_std", F.stddev("rating").over(w_pit)) \
 .withColumn("user_max_rating", F.max("rating").over(w_pit)) \
 .withColumn("user_min_rating", F.min("rating").over(w_pit)) \
 .withColumn("user_avg_text_len", F.avg(F.length(F.col("reviewText"))).over(w_pit)) \
 .withColumn("user_pos_cnt", F.count(F.when(F.col("rating") >= 4, True)).over(w_pit)) \
 .withColumn("user_hist_cats", F.collect_set("main_category").over(w_pit)) \
 .withColumn("user_hist_brands", F.collect_set("brand").over(w_pit)) \
 .withColumn("user_hist_items", F.collect_set("item_id").over(w_pit))
user_dynamic.cache()
print(f"train 样本用户 PIT 快照行数: {user_dynamic.count():,}")

print("=" * 60)
print("【Part C: 用户历史物品集合（供召回过滤，全局 train 即可）】")
print("=" * 60)
user_item_set = train.select("user_id", "item_id") \
    .distinct() \
    .groupBy("user_id") \
    .agg(F.collect_set("item_id").alias("user_hist_items_global"))
user_item_set.cache()
print(f"用户历史物品集合用户数: {user_item_set.count():,}")

print("=" * 60)
print("【Part D: 上下文特征函数】")
print("=" * 60)
def add_context_features(df):
    return df.withColumn("interact_year", F.year(F.from_unixtime("timestamp"))) \
             .withColumn("interact_month", F.month(F.from_unixtime("timestamp")))
print("上下文特征函数已定义：interact_year / interact_month")

print("=" * 60)
print("【Part E: 样本特征挂载】")
print("=" * 60)

def attach_item_and_context(df):
    df = df.join(item_feat, on="item_id", how="left")
    df = add_context_features(df)
    df = df.withColumn(
        "user_interact_cat_flag",
        F.when(
            (F.col("user_hist_cats").isNotNull()) &
            (F.array_contains(F.col("user_hist_cats"), F.col("main_category"))),
            1
        ).otherwise(0)
    ).withColumn(
        "user_interact_brand_flag",
        F.when(
            (F.col("user_hist_brands").isNotNull()) &
            (F.array_contains(F.col("user_hist_brands"), F.col("brand"))),
            1
        ).otherwise(0)
    )
    return df

train_feat = attach_item_and_context(
    user_dynamic.select(
        "user_id", "item_id", "timestamp",
        "user_hist_cnt", "user_avg_rating", "user_rating_std",
        "user_max_rating", "user_min_rating", "user_avg_text_len", "user_pos_cnt",
        "user_hist_cats", "user_hist_brands", "user_hist_items"
    )
)

user_ts_feat = user_dynamic.select(
    "user_id",
    F.col("timestamp").alias("hist_ts"),
    "user_hist_cnt", "user_avg_rating", "user_rating_std",
    "user_max_rating", "user_min_rating", "user_avg_text_len", "user_pos_cnt",
    "user_hist_cats", "user_hist_brands", "user_hist_items"
)

# ✅【接口对齐关键】对外输出 user_feat_pit，hist_ts重命名pit_time，保留原始user_ts_feat用于精排PIT最近快照匹配
user_feat_pit = user_ts_feat.withColumnRenamed("hist_ts", "pit_time").cache()
user_ts_feat.cache()

def attach_val_test_feat(samples_df, name):
    s = samples_df.select(
        "user_id", "item_id",
        F.col("timestamp").alias("label_ts")
    )
    joined = s.join(user_ts_feat, on="user_id", how="left") \
              .filter(F.col("label_ts") > F.col("hist_ts"))
    w_nearest = Window.partitionBy("user_id", "label_ts") \
                      .orderBy(F.desc("hist_ts"))
    nearest = joined.withColumn("rn", F.row_number().over(w_nearest)) \
                    .filter(F.col("rn") == 1) \
                    .drop("rn", "hist_ts") \
                    .withColumnRenamed("label_ts", "timestamp")
    feat = attach_item_and_context(nearest)
    print(f"{name} 特征表行数: {feat.count():,}")
    return feat

val_feat = attach_val_test_feat(val, "验证集")
test_feat = attach_val_test_feat(test, "测试集")

train_feat.cache()
val_feat.cache()
test_feat.cache()

print(f"\n✅ 特征表构建完成")
print(f"train_feat: {train_feat.count():,} 行")
print(f"val_feat  : {val_feat.count():,} 行")
print(f"test_feat : {test_feat.count():,} 行")

print("=" * 60)
print("【Part F: 特征校验与缺失填充】")
print("=" * 60)

def check_feat_missing(df, name):
    missing = df.filter(
        F.col("user_hist_cnt").isNull() |
        F.col("item_train_cnt").isNull()
    ).count()
    print(f"{name} 特征缺失行数: {missing}")

check_feat_missing(train_feat, "训练集")
check_feat_missing(val_feat, "验证集")
check_feat_missing(test_feat, "测试集")

def fill_feat_missing(df):
    return df.fillna({
        "user_hist_cnt": 0,
        "user_avg_rating": 0.0,
        "user_rating_std": 0.0,
        "user_max_rating": 0.0,
        "user_min_rating": 0.0,
        "user_avg_text_len": 0.0,
        "user_pos_cnt": 0,
        "user_interact_cat_flag": 0,
        "user_interact_brand_flag": 0,
        "item_train_cnt": 0,
        "item_train_avg_rating": 0.0,
        "item_train_rating_std": 0.0,
        "item_train_pos_cnt": 0,
        "item_train_avg_text_len": 0.0
    })

train_feat = fill_feat_missing(train_feat)
val_feat   = fill_feat_missing(val_feat)
test_feat  = fill_feat_missing(test_feat)

def time_range(df, name):
    r = df.select(F.min("timestamp"), F.max("timestamp")).collect()[0]
    print(f"{name} 时间范围: {r[0]} ~ {r[1]}")

print("\n时间范围校验（应严格递增）：")
time_range(train_feat, "train_feat")
time_range(val_feat, "val_feat")
time_range(test_feat, "test_feat")

print("\n最终特征列总览：")
print(train_feat.columns)
print("\n✅ 特征工程完成！对外输出 item_feat / user_ts_feat / user_feat_pit")



# ==============================================================================
# 【模块】召回层完整代码【修复定稿版】
# ✅封装 run_recall_for_users(user_df) 函数，产出三份召回候选，全部基于train的热度、物品相似度，无泄露
# ==============================================================================
print("=" * 70)
print("【召回层 · 定稿版】ItemCF + 热门召回 + 多路合并 + 评估 + 特征转换")
print("=" * 70)

# ---------- 召回全局基础素材（全部只从train计算，只计算一次！） ----------
train_pos = train.filter(F.col("rating") >= 4) \
                 .select("user_id", "item_id", "timestamp") \
                 .dropDuplicates(["user_id", "item_id"])
train_pos.cache()
print(f"训练集正反馈交互数: {train_pos.count():,}")

user_hist_items_global_recall = train_pos.groupBy("user_id") \
    .agg(F.collect_set("item_id").alias("hist_items"))

# 热门召回前置准备
item_hot = train_pos.groupBy("item_id") \
    .agg(F.count("*").alias("hot_score")) \
    .orderBy(F.desc("hot_score"))
item_hot.cache()
print(f"有热度物品数: {item_hot.count():,}")

hot_items_base = item_hot.limit(HOT_TOP_K * 2) \
    .select(F.col("item_id").alias("rec_item"), "hot_score")

# 计算ItemCF共现与相似度（全局只算一次）
user_items = train_pos.groupBy("user_id").agg(F.collect_set("item_id").alias("items"))
cooc = user_items.select("user_id", F.explode("items").alias("item_i")) \
    .join(user_items.select("user_id", F.explode("items").alias("item_j")), on="user_id") \
    .filter(F.col("item_i") < F.col("item_j")) \
    .groupBy("item_i", "item_j") \
    .agg(F.count("*").alias("cooc_cnt")) \
    .filter(F.col("cooc_cnt") >= MIN_COOC)
cooc_sym = cooc.select(F.col("item_j").alias("item_i"), F.col("item_i").alias("item_j"), "cooc_cnt")
item_sim_base = cooc.unionByName(cooc_sym)

item_pop = item_hot.select(F.col("item_id").alias("item_i"), F.col("hot_score").alias("pop_i"))
item_sim_base = item_sim_base.join(item_pop, on="item_i", how="left") \
    .join(item_pop.select(F.col("item_i").alias("item_j"), F.col("pop_i").alias("pop_j")), on="item_j", how="left") \
    .withColumn("pop_i", F.when(F.col("pop_i").isNull(), 1).otherwise(F.col("pop_i"))) \
    .withColumn("pop_j", F.when(F.col("pop_j").isNull(), 1).otherwise(F.col("pop_j"))) \
    .withColumn("sim_score", F.col("cooc_cnt") / F.sqrt(F.col("pop_i") * F.col("pop_j")))
item_sim_base.cache()
print(f"物品相似对数量: {item_sim_base.count():,}")

# StringIndexer 词典（只在train拟合，后续精排复用）
brand_model = StringIndexer(inputCol="brand", outputCol="brand_id", handleInvalid="keep") \
    .fit(train.select("brand").fillna("unknown"))
cate_model = StringIndexer(inputCol="main_category", outputCol="cate_id", handleInvalid="keep") \
    .fit(train.select("main_category").fillna("unknown"))
print(f"brand 编码数量（含未知桶）: {len(brand_model.labels):,}")
print(f"main_category 编码数量（含未知桶）: {len(cate_model.labels):,}")

# =====================【核心通用召回函数】=====================
def run_recall_for_users(user_df):
    """
    输入：一批user_id的DataFrame
    输出：该批用户的多路合并召回候选表 user_id,item_id,recall_score,recall_sources
    ⚠️ 全部相似度、热度素材来自train，不会读取val/test数据，严格防泄露
    """
    users = user_df.select("user_id").distinct() # 注意：这是 train/val/test 里的所有用户
    # ========== 热门召回 ==========（召回全局中最热门的top商品推荐给每个用户）
    hot_recall = users.crossJoin(F.broadcast(hot_items_base))
    hot_recall = hot_recall.join(user_hist_items_global_recall, on="user_id", how="left") \
        .filter(~F.array_contains(F.col("hist_items"), F.col("rec_item")) | F.col("hist_items").isNull()) \
        .select("user_id", "rec_item", F.col("hot_score").alias("recall_score"), F.lit("hot").alias("recall_type"))
    w_hot = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    hot_recall = hot_recall.withColumn("rn", F.row_number().over(w_hot)) \
        .filter(F.col("rn") <= HOT_TOP_K).drop("rn")

    # ========== ItemCF召回 ==========（用户历史买过 A，就给该用户推荐 A 的相似物品； 相似物品：喜欢 A 商品的用户，很多也喜欢 B，则 A 和 B 相似）
    # 用户历史物品（只能用train的历史交互）
    user_hist_item_df = train_pos.select("user_id", F.col("item_id").alias("hist_item")).join(users, on="user_id")
    itemcf_cand = user_hist_item_df.join(
        item_sim_base.select(F.col("item_i").alias("hist_item"), F.col("item_j").alias("rec_item"), "sim_score"),
        on="hist_item", how="inner"
    ).groupBy("user_id", "rec_item").agg(F.sum("sim_score").alias("recall_score"))

    itemcf_cand = itemcf_cand.join(user_hist_items_global_recall, on="user_id", how="left") \
        .filter(~F.array_contains(F.col("hist_items"), F.col("rec_item")) | F.col("hist_items").isNull()) \
        .select("user_id", "rec_item", "recall_score", F.lit("itemcf").alias("recall_type"))
    w_cf = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    itemcf_recall = itemcf_cand.withColumn("rn", F.row_number().over(w_cf)) \
        .filter(F.col("rn") <= ITEMCF_TOP_K).drop("rn")

    # ==========多路合并去重，取最高分，记录来源，全局截断FINAL_RECALL_K ==========
    recall_all = itemcf_recall.unionByName(hot_recall) # 合并两路召回
    recall_merged = recall_all.groupBy("user_id", "rec_item") \
        .agg(F.max("recall_score").alias("recall_score"),
             F.concat_ws(",", F.collect_set("recall_type")).alias("recall_sources"))
    w_final = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    recall_out = recall_merged.withColumn("rn", F.row_number().over(w_final)) \
        .filter(F.col("rn") <= FINAL_RECALL_K) \
        .drop("rn") \
        .withColumnRenamed("rec_item", "item_id")
    recall_out.cache()
    return recall_out

# ============ 调用函数，生成三份召回候选（接口对齐关键！！） ============
recall_candidate_df = run_recall_for_users(train.select("user_id").distinct())    # 训练用户召回候选（训练hard负）
val_recall_candidate = run_recall_for_users(val.select("user_id").distinct())      # 验证用户召回候选（验证hard负）
test_recall_candidate = run_recall_for_users(test.select("user_id").distinct())   # 测试用户召回候选（测试打分）

print(f"\n✅三份召回候选生成完毕")
print(f"训练用户召回候选行数: {recall_candidate_df.count():,}")
print(f"验证用户召回候选行数: {val_recall_candidate.count():,}")
print(f"测试用户召回候选行数: {test_recall_candidate.count():,}")

# ---------- 召回评估（基于val正样本，Recall@K） ----------
val_pos = val.filter(F.col("rating") >= 4) \
             .select("user_id", "item_id") \
             .dropDuplicates(["user_id", "item_id"]) \
             .withColumn("label", F.lit(1))
val_users = val_pos.select("user_id").distinct()
real_cnt = val_pos.groupBy("user_id").agg(F.count("*").alias("real_cnt"))

def recall_at_k(recall_df, k):
    w = Window.partitionBy("user_id").orderBy(F.desc("recall_score"))
    topk = recall_df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= k).drop("rn")
    hit_cnt = topk.join(val_pos, on=["user_id", "item_id"], how="inner").groupBy("user_id").agg(F.count("*").alias("hit_cnt"))
    per_user = val_users.join(real_cnt, on="user_id", how="inner") \
                        .join(hit_cnt, on="user_id", how="left").fillna({"hit_cnt":0}) \
                        .withColumn("recall", F.col("hit_cnt") / F.col("real_cnt"))
    return per_user.agg(F.avg("recall")).collect()[0][0]

print("\n召回 Recall@K 评估（基于验证集 val 正反馈）:")
for k in K_LIST:
    if k <= FINAL_RECALL_K:
        recall_k = recall_at_k(val_recall_candidate, k)
        print(f"  Recall@{k:<4}= {recall_k:.4f}")

hit_user_cnt = val_recall_candidate.join(val_pos, on=["user_id", "item_id"], how="inner").select("user_id").distinct().count()
total_val_user = val_users.count()
print(f"\n辅助指标：")
print(f"  验证集有正反馈用户数: {total_val_user:,}")
print(f"  召回命中至少1个正反馈的用户数: {hit_user_cnt:,}")
print(f"  命中用户占比: {hit_user_cnt / max(total_val_user, 1) * 100:.2f}%")

# 释放召回中间表
train_pos.unpersist()
item_hot.unpersist()
item_sim_base.unpersist()

print("\n✅召回模块全部完成，输出：recall_candidate_df / val_recall_candidate / test_recall_candidate / brand_model / cate_model")


# ==============================================================================
# 【模块4】精排完整修复版【彻底修复PIT快照匹配，复用召回StringIndexer，接口全部对齐】
# 重大修复：废弃等值join user_id+pit_time；复用user_ts_feat实现【每个样本取<=pit_time最近用户快照】，解决val负样本特征全部置0
# ==============================================================================
print("=" * 70)
print("【精排融合版｜完整修复版】Hard负样本 + 时序验证 + 候选集NDCG/Recall评估")
print("=" * 70)

# -------------------------- 前置接口校验 --------------------------
def preflight_check():
    global_vars = globals()
    miss_list = []
    check_vars = [
        "train_inter", "val_inter", "test_inter",
        "item_feat", "user_ts_feat",
        "recall_candidate_df", "val_recall_candidate", "test_recall_candidate",
        "brand_model", "cate_model"
    ]
    for v in check_vars:
        if v not in global_vars:
            miss_list.append(v)
    if len(miss_list) > 0:
        raise RuntimeError(f"❌精排前置校验失败，缺失变量：{miss_list}")
preflight_check()

# --------------------------工具函数 --------------------------
def get_user_last_ts(inter_df):
    """每个用户在该集合中的最后交互时间，用作负样本默认 PIT"""
    return inter_df.groupBy("user_id").agg(F.max("timestamp").alias("pit_time"))

def build_pos(inter_df, name):
     """正样本：真实交互且 rating 达标"""
    pos = inter_df.filter(F.col("rating") >= POS_RATING_TH) \
        .select("user_id", "item_id", F.col("timestamp").alias("pit_time"), F.lit(1).alias("label")) \
        .dropDuplicates(["user_id", "item_id"])
    print(f"{name} 正样本数: {pos.count():,}")
    return pos

def sample_hard_neg(pos_df, recall_df, user_last_ts_df, ratio, name):
     """
    Hard 负样本：出现在召回候选中，但用户未真实高分交互
    关键：给负样本赋予用户 last_ts 作为 pit_time，保证能挂上用户特征
    ✅ 语义：允许保留用户低分交互样本，作为负例（曝光但低分）
    """
    pos_pairs = pos_df.select("user_id", "item_id").distinct()
    hard = recall_df.select("user_id", "item_id").distinct() \
        .join(pos_pairs, on=["user_id", "item_id"], how="left_anti") \
        .join(user_last_ts_df, on="user_id", how="inner") \
        .withColumn("label", F.lit(0))
    w = Window.partitionBy("user_id").orderBy(F.rand(RANDOM_SEED))
    hard = hard.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= ratio).drop("rn")
    print(f"{name} Hard负样本数: {hard.count():,}")
    return hard

def sample_random_neg(pos_df, item_pool_df, user_last_ts_df, all_user_item_pairs, ratio, name):
     """
    全局随机负样本（控制数量，避免全量 crossJoin 爆炸）
    随机负样本定义：用户**从未发生任何交互**的物品
    注意：大数据集需替换为按用户采样物品，当前适配小数据集
    ✅修复：传入 all_user_item_pairs，过滤用户【全部历史交互（高分+低分）】
    """
    users = pos_df.select("user_id").distinct()
    cand = users.crossJoin(F.broadcast(item_pool_df)) \
        .join(all_user_item_pairs, on=["user_id", "item_id"], how="left_anti") \
        .join(user_last_ts_df, on="user_id", how="inner") \
        .withColumn("label", F.lit(0))
    w = Window.partitionBy("user_id").orderBy(F.rand(RANDOM_SEED))
    neg = cand.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= ratio).drop("rn")
    print(f"{name} 随机负样本数: {neg.count():,}")
    return neg

# ✅【PIT重大修复：不再等值join，复用特征工程逻辑：每个样本取pit_time之前最近一条user_ts_feat快照】
def attach_features_pit(sample_df, user_ts_feat_df, item_feat_df):
    """
    PIT 挂载用户特征 + 物品特征
    sample_df: 必须包含 user_id, pit_time（样本的查询时间）
    user_ts_feat_df: user_ts_feat(user_id, hist_ts, 用户全部PIT快照)
    item_feat_df:物品静态特征
    返回：挂载好用户PIT快照+物品特征的样本
    """
    s = sample_df.select("user_id", "item_id", "pit_time", "label")
    joined = s.join(user_ts_feat_df, on="user_id", how="left") \
              .filter(F.col("pit_time") > F.col("hist_ts"))
    w_nearest = Window.partitionBy("user_id", "pit_time") \
                      .orderBy(F.desc("hist_ts"))
    nearest = joined.withColumn("rn", F.row_number().over(w_nearest)) \
                    .filter(F.col("rn") == 1) \
                    .drop("rn", "hist_ts")
    df = nearest.join(item_feat_df, on="item_id", how="left")
    # 缺失填充
    df = df.fillna({
        "user_hist_cnt": 0,
        "user_avg_rating": 0.0,
        "user_rating_std": 0.0,
        "user_max_rating": 0.0,
        "user_min_rating": 0.0,
        "user_avg_text_len": 0.0,
        "user_pos_cnt": 0,
        "item_train_cnt": 0,
        "item_train_avg_rating": 0.0,
        "item_train_rating_std": 0.0,
        "item_train_pos_cnt": 0,
        "item_train_avg_text_len": 0.0
    })
    # 计算交叉flag
    df = df.withColumn(
        "user_interact_cat_flag",
        F.when(
            (F.col("user_hist_cats").isNotNull()) &
            (F.array_contains(F.col("user_hist_cats"), F.col("main_category"))),
            1
        ).otherwise(0)
    ).withColumn(
        "user_interact_brand_flag",
        F.when(
            (F.col("user_hist_brands").isNotNull()) &
            (F.array_contains(F.col("user_hist_brands"), F.col("brand"))),
            1
        ).otherwise(0)
    )
    miss_cnt = df.filter(F.col("user_hist_cnt").isNull()).count()
    if miss_cnt > 0:
        print(f"⚠️ attach_features_pit 用户特征缺失行数:{miss_cnt}")
    return df

def to_pandas_features(spark_df, feature_cols, cat_cols):
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

def calc_ranking_metrics(pdf, score_col="score", label_col="label", ks=(5,10,20)):
    recalls = {k:[] for k in ks}
    ndcgs = {k:[] for k in ks}
    for _, group in pdf.groupby("user_id"):
        if group[label_col].sum() <= 0:
            continue
        g = group.sort_values(score_col, ascending=False)
        labels = g[label_col].values.astype(float)
        for k in ks:
            topk_labels = labels[:k]
            recalls[k].append(topk_labels.sum() / labels.sum())
            def dcg(rels):
                idx = np.arange(1, len(rels)+1)
                return np.sum((2**rels -1)/np.log2(idx+1))
            k_labs = labels[:k]
            ideal = np.sort(labels)[::-1][:k]
            idcg = dcg(ideal)
            ndcgs[k].append(0.0 if idcg ==0 else dcg(k_labs)/idcg)
    res = {}
    for k in ks:
        res[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0R5
        res[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return res

# -------------------------- Part2 构造正负样本 --------------------------
print("\n【Part 2】构造正负样本")
train_pos = build_pos(train_inter, "训练集")
val_pos   = build_pos(val_inter, "验证集")

train_user_last_ts = get_user_last_ts(train_inter)
val_user_last_ts   = get_user_last_ts(val_inter)

train_all_user_item = train_inter.select("user_id", "item_id").distinct()
val_all_user_item = val_inter.select("user_id", "item_id").distinct()
item_pool = item_feat.select("item_id").distinct().cache()

train_hard = sample_hard_neg(train_pos, recall_candidate_df, train_user_last_ts, HARD_NEG_RATIO, "训练集")
train_rand = sample_random_neg(train_pos, item_pool, train_user_last_ts, train_all_user_item, RANDOM_NEG_RATIO, "训练集")
train_sample = train_pos.unionByName(train_hard).unionByName(train_rand).dropDuplicates(["user_id", "item_id"]).cache()

val_hard = sample_hard_neg(val_pos, val_recall_candidate, val_user_last_ts, HARD_NEG_RATIO, "验证集")
val_rand = sample_random_neg(val_pos, item_pool, val_user_last_ts, val_all_user_item, RANDOM_NEG_RATIO, "验证集")
val_sample = val_pos.unionByName(val_hard).unionByName(val_rand).dropDuplicates(["user_id", "item_id"]).cache()

print(f"\n训练集明细：正样本:{train_pos.count():,}, hard负:{train_hard.count():,}, random负:{train_rand.count():,}")
print(f"验证集明细：正样本:{val_pos.count():,}, hard负:{val_hard.count():,}, random负:{val_rand.count():,}")
print(f"训练样本总数: {train_sample.count():,}")
print(f"验证样本总数: {val_sample.count():,}")

# -------------------------- Part3 PIT挂载特征（使用修复后的attach_features_pit） --------------------------
print("\n【Part 3】PIT 特征挂载【修复版，最近快照匹配】")
train_feat_df = attach_features_pit(train_sample, user_ts_feat, item_feat).cache()
val_feat_df   = attach_features_pit(val_sample, user_ts_feat, item_feat).cache()
print(f"训练特征表: {train_feat_df.count():,}")
print(f"验证特征表: {val_feat_df.count():,}")

# -------------------------- Part4 特征转换 --------------------------
"""
整体主线逻辑：区分特征属性（类别 / 数值）→ 特征处理清洗 → 对齐训练 / 验证特征 → 输出模型输入 X、y。
额外特殊分支：`brand`、`main_category` 这两个物品特征，召回链路已经用 `StringIndexer` 做过编码。为了防止编码漂移，精排阶段直接复用召回已经 fit 好的模型
做 transform，不再重新 fit。其余类别特征没有在召回使用，不需要跨模块对齐，统一放到 pandas 里用 category 类型交给树模型处理。
"""
print("\n【Part 4】特征转换，复用召回StringIndexer模型")

# 4.1 剔除，筛选特征，区分「类别特征 / 数值特征」
exclude_cols = {"user_id", "item_id", "pit_time", "label", "timestamp", "rating", "hist_ts"}
all_cols = set(train_feat_df.columns)
feature_cols = sorted(list(all_cols - exclude_cols))
likely_cat = {"brand", "main_category", "price_bin", "cate_id", "brand_id",
              "user_interact_cat_flag", "user_interact_brand_flag"}
cat_cols = [c for c in feature_cols if c in likely_cat or train_feat_df.schema[c].dataType.simpleString() == "string"]
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"数值特征数: {len(num_cols)} | 类别特征数: {len(cat_cols)}")
print("类别特征:", cat_cols)

# 4.2 特征转换与清洗
# 4.2.1 brand category 召回阶段做过 StringIndexer编码，单独处理防止特征漂移
# ✅复用召回阶段 fit 好的模型做 transform，精排不再重新 fit，跨模块对齐
def apply_recall_indexer(df):
    df = df.fillna({"brand":"unknown", "main_category":"unknown"})
    df = brand_model.transform(df)
    df = cate_model.transform(df)
    return df

train_feat_df = apply_recall_indexer(train_feat_df)
val_feat_df = apply_recall_indexer(val_feat_df)

# 4.2.2 其他特征做特征转换与清洗
train_pdf = to_pandas_features(train_feat_df, feature_cols, cat_cols)
val_pdf   = to_pandas_features(val_feat_df, feature_cols, cat_cols)

# 4.3 对齐特征，生成模型输入 X、y
use_features = [c for c in feature_cols if c in train_pdf.columns and c in val_pdf.columns]
X_train, y_train = train_pdf[use_features], train_pdf["label"]
X_val, y_val     = val_pdf[use_features], val_pdf["label"]
cat_in_use = [c for c in cat_cols if c in use_features]

# -------------------------- Part5 LightGBM训练 --------------------------
print("\n【Part 5】LightGBM 训练")
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_in_use, free_raw_data=True)
lgb_val   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_in_use, reference=lgb_train, free_raw_data=True)

model = lgb.train(
    LGB_PARAMS,
    lgb_train,
    num_boost_round=800,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)
print(f"best_iteration: {model.best_iteration}")
try:
    print(f"best val auc: {model.best_score['val']['auc']:.4f}")
except Exception:
    pass

imp = pd.DataFrame({
    "feature": use_features,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
print("\n特征重要性 Top 15:")
print(imp.head(15).to_string(index=False))

# -------------------------- Part6 测试集候选打分 --------------------------
print("\n【Part 6】测试集候选排序评估")
# 测试候选补充pit_time，取用户训练集合最后交互时间
test_user_last_ts = get_user_last_ts(train_inter)
test_cand = test_recall_candidate.select("user_id", "item_id") \
    .join(test_user_last_ts, on="user_id", how="left") \
    .withColumn("label", F.lit(0))

test_feat_df = attach_features_pit(test_cand, user_ts_feat, item_feat)
test_feat_df = apply_recall_indexer(test_feat_df)
test_pdf = to_pandas_features(test_feat_df, use_features, cat_in_use)

for c in use_features:
    if c not in test_pdf.columns:
        test_pdf[c] = 0

test_pdf["score"] = model.predict(test_pdf[use_features], num_iteration=model.best_iteration)

test_label = test_inter.filter(F.col("rating") >= POS_RATING_TH) \
    .select("user_id", "item_id").dropDuplicates().toPandas() # test_label：真实样本对: user_id - item_id
test_label["label"] = 1  # 测试集真实正样本列表，user_id+item_id，label=1

test_result = test_pdf.merge(test_label, on=["user_id", "item_id"], how="left")
test_result["label"] = test_result["label"].fillna(0).astype(int)

metrics = calc_ranking_metrics(test_result, score_col="score", label_col="label", ks=EVAL_TOPK)
print("\n===== 精排离线指标（测试集召回候选） =====")
for k, v in metrics.items():  # k → key，指标名称（字符串），例如 ndcg@5   v → value，这个指标对应的数值（浮点数），例如 0.6123
    print(f"{k}: {v:.4f}")
try:
    auc = roc_auc_score(test_result["label"], test_result["score"])
    print(f"Candidate AUC: {auc:.4f}")
except Exception:
    pass

# -------------------------- Part7 模型保存 --------------------------
model.save_model("lgb_rank_model_fused.txt")
print("\n✅ 精排完成，模型已保存: lgb_rank_model_fused.txt")
print("""
【输出，可直接送入重排模块】
- lgb_rank_model_fused.txt：LightGBM精排模型
- imp：特征重要性DataFrame
- test_result：用户+候选item+精排score+真实label
- metrics：Recall@K / NDCG@K 离线指标
""")


# ============================================================
# 重排模块｜规则型后重排（多样性+热门惩罚）
# 输入：test_result 来自精排输出 pandas df
# 字段要求输入df：user_id, item_id, score, label
# 依赖物品特征表 item_feat 获取商品类目、热度信息
# ============================================================
import pandas as pd
import numpy as np

print("=" * 70)
print("【重排模块｜规则后重排：多样性 + 热门惩罚】")
print("=" * 70)

# ===================== 重排配置区 =====================
TOP_N_OUTPUT = 20               # 每个用户最终输出推荐条数
HOT_PENALTY_COEFF = 0.08        # 热门惩罚系数，越大压制爆款越强
MAX_CATEGORY_RATIO = 0.35       # 单个类目在结果中最大占比（0~1）
EVAL_TOPK = [5,10,20]
# ======================================================

# --------------------------
# 步骤1：把物品特征 join 进test_result（需要类目、热度用于重排规则）
# --------------------------
# item_feat是Spark表，取出需要的两列转为pandas
item_meta_pdf = item_feat.select(
    "item_id",
    "main_category",
    "item_train_cnt"
).toPandas()

# 合并元信息给精排输出结果
test_result_w_meta = test_result.merge(
    item_meta_pdf,
    on="item_id",
    how="left"
)

# 热度做log缩放用于惩罚
test_result_w_meta["item_cnt_log"] = np.log1p(test_result_w_meta["item_train_cnt"].fillna(0))

# --------------------------
# 重排核心函数：单用户列表重排
# 输入该用户df：item_id / score / label / main_category / item_cnt_log
# 返回重排之后df
# --------------------------
def reorder_one_user(group_df, top_n, hot_penalty, max_cat_ratio):
    df = group_df.copy().reset_index(drop=True)
    # 1.生成调整分数：原始score − 热门惩罚项
    df["adjust_score"] = df["score"] - hot_penalty * df["item_cnt_log"]
    # 先按adjust_score降序
    df = df.sort_values("adjust_score", ascending=False).reset_index(drop=True)

    max_per_cat = int(top_n * max_cat_ratio)
    out_list = []
    cat_counter = dict()

    for _, row in df.iterrows():
        
        cat = row["main_category"]
        cnt = cat_counter.get(cat, 0)
        if cnt >= max_per_cat:
            continue
        out_list.append(row)
        cat_counter[cat] = cnt + 1
        if len(out_list) >= top_n:
            break

    # 兜底：如果筛选后不足top_n，把剩余候选补进来（不再做类目限制）
    if len(out_list) < top_n:
        selected_item_ids = {r["item_id"] for r in out_list}
        remain = df[~df["item_id"].isin(selected_item_ids)].reset_index(drop=True)
        need = top_n - len(out_list)
        out_list.extend(remain.head(need).to_dict("records"))

    return pd.DataFrame(out_list)

# --------------------------
# 步骤2：对全部用户执行重排
# --------------------------
reorder_result_list = []
origin_result_list = []

for uid, group in test_result_w_meta.groupby("user_id"):
    # 原始精排排序结果（用于对比基线）
    origin_top = group.sort_values("score", ascending=False).head(TOP_N_OUTPUT).copy()
    origin_result_list.append(origin_top)

    # 执行重排
    reordered = reorder_one_user(
        group,
        top_n=TOP_N_OUTPUT,
        hot_penalty=HOT_PENALTY_COEFF,
        max_cat_ratio=MAX_CATEGORY_RATIO
    )
    reordered["user_id"] = uid
    reorder_result_list.append(reordered)

df_origin_top = pd.concat(origin_result_list, ignore_index=True)
df_reordered = pd.concat(reorder_result_list, ignore_index=True)

print(f"原始精排Top{N_OUTPUT}样本数: {len(df_origin_top)}")
print(f"重排后输出样本数: {len(df_reordered)}")

# --------------------------
# 评估工具函数：NDCG / Recall@K
# --------------------------
def calc_ranking_metrics_df(pdf, ks):
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    for _, group in pdf.groupby("user_id"):
        total_pos = group["label"].sum()
        if total_pos <= 1e-6:
            continue
        labels = group["label"].values.astype(float)
        for k in ks:
            topk = labels[:k]
            recalls[k].append(np.sum(topk)/total_pos)

            def dcg(rels):
                idx = np.arange(1, len(rels)+1)
                return np.sum((2**rels - 1)/np.log2(idx+1))
            k_labs = labels[:k]
            ideal = np.sort(labels)[::-1][:k]
            idcg = dcg(ideal)
            ndcgs[k].append(0.0 if idcg < 1e-6 else dcg(k_labs)/idcg)
    res = {}
    for k in ks:
        res[f"Recall@{k}"] = float(np.mean(recalls[k])) if len(recalls[k])>0 else 0.0
        res[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if len(ndcgs[k])>0 else 0.0
    return res

# --------------------------
# 步骤3：对比【原始精排】VS【重排之后】指标
# --------------------------
metric_origin = calc_ranking_metrics_df(df_origin_top, EVAL_TOPK)
metric_reorder = calc_ranking_metrics_df(df_reordered, EVAL_TOPK)

print("\n==== 指标对比：精排原始排序 VS 规则重排 ====")
print(f"{'Metric':<12} {'Origin(精排)':<12} {'Reorder(重排)':<12}")
for k in EVAL_TOPK:
    kr = f"Recall@{k}"
    kn = f"NDCG@{k}"
    print(f"{kr:<12} {metric_origin[kr]:<12.4f} {metric_reorder[kr]:<12.4f}")
    print(f"{kn:<12} {metric_origin[kn]:<12.4f} {metric_reorder[kn]:<12.4f}")

# --------------------------
# 步骤4：简单多样性统计（实验报告可用）
# 统计每个用户输出结果内类目数量，均值越高多样性越好
# --------------------------
def calc_diversity(pdf):
    div_list = []
    for _,g in pdf.groupby("user_id"):
        cat_set = set(g["main_category"].dropna().tolist())
        div_list.append(len(cat_set))
    return np.mean(div_list)

div_origin = calc_diversity(df_origin_top)
div_reorder = calc_diversity(df_reordered)
print(f"\n多样性指标(平均每用户不同类目数):")
print(f"精排原始排序: {div_origin:.2f}")
print(f"重排后排序:   {div_reorder:.2f}")

# --------------------------
# 保存输出结果
# --------------------------
df_reordered.to_csv("reorder_final_result.csv", index=False, encoding="utf‑8")
print("\n✅重排完成，输出文件: reorder_final_result.csv")
print("输出字段：user_id,item_id,score,adjust_score,label,main_category,item_train_cnt,item_cnt_log")
print("""
说明：
1. adjust_score为重排调整后打分，重排逻辑基于该值；score为原始LightGBM精排分数
2. 观察现象：重排通常会小幅降低NDCG/Recall，但是提升类目多样性；
3. 调参：HOT_PENALTY_COEFF增大，热门压制更强；MAX_CATEGORY_RATIO越小，类目约束越强。
""")



# ==========================================================
# 全链路执行完毕：统一释放Spark缓存 + Python内存回收
# 注意：必须所有计算、打印、文件导出全部完成后再执行！
# ==========================================================
import gc

print("\n" + "="*60)
print("【全链路结束，开始释放全部Spark缓存与内存】")
print("="*60)

# ---------------- Spark DataFrame 缓存释放 ----------------
# 原始基础表
df_final.unpersist()
meta_clean.unpersist()

# 时序切分
train.unpersist()
val.unpersist()
test.unpersist()

# 特征工程 FeatureStore
item_feat.unpersist()
user_ts_feat.unpersist()
user_feat_pit.unpersist()
user_item_set.unpersist()

train_feat.unpersist()
val_feat.unpersist()
test_feat.unpersist()

# 召回模块输出表
recall_candidate_df.unpersist()
val_recall_candidate.unpersist()
test_recall_candidate.unpersist()

# ---------------- pandas 对象释放（重排输出等Python内存对象） ----------------
try:
    del test_result
    del test_result_w_meta
    del df_origin_top
    del df_reordered
    del item_meta_pdf
except NameError:
    # 变量不存在时跳过，防止交互式环境多次运行报错
    pass

# 强制垃圾回收
gc.collect()

print("✅ 全部缓存与内存资源释放完成。")
