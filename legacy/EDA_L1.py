import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("AmazonBeauty_EDA_L1") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark 初始化完成")


# ====== 请修改为你的实际路径 ======
INTER_PATH = "D:\Amazon_beauty_project\Beauty_5.json"
META_PATH  = "D:\Amazon_beauty_project\meta_Beauty.json"


# 读取，不做任何删除，EDA只探查
inter_raw = spark.read.json(INTER_PATH)

# 统一别名，但保留 helpful / reviewText / summary 供探查
inter_df = inter_raw.select(
    F.col("asin").alias("item_id"),
    F.col("reviewerID").alias("user_id"),
    F.col("overall").cast("float").alias("rating"),
    F.col("unixReviewTime").cast("long").alias("timestamp"),
    F.col("helpful"),
    F.col("reviewText"),
    F.col("summary"),
    F.col("reviewTime")
)

inter_df.cache()


# 读取
meta_raw = spark.read.json(META_PATH)

meta_raw = meta_raw.cache()

meta_valid = meta_raw.filter(F.col("_corrupt_record").isNull())

meta_df = meta_valid.select(
    F.col("asin").alias("item_id"),
    F.col("brand"),
    F.col("categories"),
    F.col("description"),
    F.col("imUrl"),
    F.col("price"),
    F.col("related"),
    F.col("salesRank"),
    F.col("title")
)

meta_df.cache()


print("=" * 70)
print("L1 矩阵画像 — 稀疏度、长尾、头部贡献、5‑core模拟探查")
print("=" * 70)

# ========== L1‑1 整体矩阵骨架 ==========
print("\n【L1‑1 整体矩阵骨架】")

total_interactions = inter_df.count()
total_users = inter_df.select("user_id").distinct().count()
total_items = inter_df.select("item_id").distinct().count()

print(f"总交互条数: {total_interactions:,}")
print(f"独立用户数: {total_users:,}")
print(f"独立商品数: {total_items:,}")

# 理论最大可能交互数、稀疏度
max_possible = total_users * total_items
sparsity = 1.0 - (total_interactions / max_possible)
print(f"理论最大交互(用户×商品): {max_possible:,}")
print(f"矩阵稀疏度: {sparsity:.4f}")

# ========== L1‑2 用户侧：用户交互频次分布 ==========
print("\n【L1‑2 用户侧：用户交互频次分布】")
user_cnt_df = inter_df.groupBy("user_id").count().withColumnRenamed("count","inter_cnt")

print("\n用户交互次数统计摘要(min/max/mean/分位数):")
user_cnt_df.select("inter_cnt").summary().show(truncate=False)

print("\n用户交互次数区间分布:")
user_cnt_df.groupBy("inter_cnt").count()\
    .orderBy("inter_cnt")\
    .show(20, truncate=False)

# 头部用户贡献占比
print("\n头部用户交互贡献模拟:")
# 按用户交互数降序，累加看top 5% / top10%用户占据多少交互
user_cnt_sorted = user_cnt_df.orderBy(F.desc("inter_cnt"))
user_cnt_sorted.createOrReplaceTempView("user_cnt_sorted")

top5pct_user = spark.sql(f"""
SELECT SUM(inter_cnt) AS top5pct_inter
FROM (SELECT inter_cnt FROM user_cnt_sorted LIMIT {int(total_users * 0.05)})
""").collect()[0]["top5pct_inter"]

top10pct_user = spark.sql(f"""
SELECT SUM(inter_cnt) AS top10pct_inter
FROM (SELECT inter_cnt FROM user_cnt_sorted LIMIT {int(total_users * 0.10)})
""").collect()[0]["top10pct_inter"]

print(f"Top5% 用户贡献交互占比: {top5pct_user / total_interactions * 100:.2f} %")
print(f"Top10% 用户贡献交互占比: {top10pct_user / total_interactions * 100:.2f} %")

# ========== L1‑3 物品侧：商品被交互频次分布 ==========
print("\n【L1‑3 物品侧：商品被交互频次分布】")
item_cnt_df = inter_df.groupBy("item_id").count().withColumnRenamed("count","inter_cnt")

print("\n商品被交互次数统计摘要(min/max/mean/分位数):")
item_cnt_df.select("inter_cnt").summary().show(truncate=False)

print("\n商品被交互次数区间分布:")
item_cnt_df.groupBy("inter_cnt").count()\
    .orderBy("inter_cnt")\
    .show(20, truncate=False)

# 头部商品贡献占比
item_cnt_sorted = item_cnt_df.orderBy(F.desc("inter_cnt"))
item_cnt_sorted.createOrReplaceTempView("item_cnt_sorted")

top5pct_item = spark.sql(f"""
SELECT SUM(inter_cnt) AS top5pct_inter
FROM (SELECT inter_cnt FROM item_cnt_sorted LIMIT {int(total_items * 0.05)})
""").collect()[0]["top5pct_inter"]

top10pct_item = spark.sql(f"""
SELECT SUM(inter_cnt) AS top10pct_inter
FROM (SELECT inter_cnt FROM item_cnt_sorted LIMIT {int(total_items * 0.10)})
""").collect()[0]["top10pct_inter"]

print(f"Top5% 商品贡献交互占比: {top5pct_item / total_interactions * 100:.2f} %")
print(f"Top10% 商品贡献交互占比: {top10pct_item / total_interactions * 100:.2f} %")

