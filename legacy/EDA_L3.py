import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("AmazonBeauty_EDA_L3") \
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

from pyspark.sql import functions as F

print("=" * 70)
print("L3 侧信息与偏差探查 — 商品元信息 & 交互联合分析【修复版】")
print("=" * 70)

meta_valid.cache()
_ = meta_valid.count()
inter_df.cache()
_ = inter_df.count()

# ======================
# 核心性能优化：只执行一次join，后续全部复用，消除重复shuffle
# ======================
item_meta_join_df = inter_df.join(
    meta_valid,
    inter_df.item_id == meta_valid.asin,
    "left"
).cache()
_ = item_meta_join_df.count()

# ========== L3‑1 元表基础质量探查（关键字段覆盖率） ==========
print("\n【L3‑1 商品元表关键字段覆盖率】")
meta_key_cols = ["asin","brand","price","categories","salesRank"]
meta_valid.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c + "_null_cnt") for c in meta_key_cols]
).show()

meta_total_item = meta_valid.select("asin").distinct().count()
inter_item_set = inter_df.select("item_id").distinct()
hit_meta_item = meta_valid.join(inter_item_set, meta_valid.asin == inter_item_set.item_id, "inner").count()
inter_distinct_item_cnt = inter_item_set.count()
print(f"meta元表总商品数：{meta_total_item}")
print(f"交互表中存在、并且在meta表可以匹配到的商品数：{hit_meta_item}")
print(f"商品匹配率：{hit_meta_item / inter_distinct_item_cnt * 100:.2f} %")

# ========== L3‑2 price价格字段探查 ==========
print("\n【L3‑2 price价格分布探查】")
df_price = meta_valid.filter(F.col("price").isNotNull())\
    .withColumn("price_val", F.col("price").cast("double"))\
    .filter(F.col("price_val") > 0)

df_price.select(
    F.percentile_approx("price_val", [0.0,0.25,0.5,0.75,1.0],10000).alias("price_quantile"),
    F.avg("price_val").alias("price_avg")
).show()

price_bin_df = df_price.withColumn(
    "price_bin",
    F.when(F.col("price_val") < 10, "0‑10")
    .when(F.col("price_val") < 30, "10‑30")
    .when(F.col("price_val") < 80, "30‑80")
    .otherwise(">80")
)
# 使用预join好的数据集
price_bin_inter = item_meta_join_df.join(
    price_bin_df.select("asin","price_bin"),
    on="asin",
    how="inner"
)
print("\n不同价格档位总交互数量：")
price_bin_inter.groupBy("price_bin").count().orderBy("price_bin").show()

# ========== L3‑3 brand品牌探查 ==========
print("\n【L3‑3 品牌分布探查】")
df_brand = meta_valid.filter(F.col("brand").isNotNull() & (F.trim(F.col("brand")) != ""))
brand_cnt = df_brand.select(F.countDistinct("brand")).collect()[0][0]
print(f"元表有效品牌总数量：{brand_cnt}")

brand_inter_df = item_meta_join_df.filter(F.col("brand").isNotNull() & (F.trim(F.col("brand")) != ""))
brand_agg = brand_inter_df.groupBy("brand").count().withColumnRenamed("count","inter_cnt")
print("\n交互量Top10品牌：")
brand_agg.orderBy(F.col("inter_cnt").desc()).show(10)

top10_brand_inter = brand_agg.limit(10).select(F.sum("inter_cnt")).collect()[0][0]
all_brand_inter = brand_agg.select(F.sum("inter_cnt")).collect()[0][0]
print(f"Top10品牌交互总数量：{top10_brand_inter}，占全部品牌交互：{top10_brand_inter/all_brand_inter*100:.2f}%")

# ========== L3‑4 categories类目探查【修复数组越界风险】 ==========
print("\n【L3‑4 类目categories探查】")
df_cat = meta_valid.filter(F.col("categories").isNotNull() & (F.size(F.col("categories"))>0))
df_cat = df_cat.withColumn("cat_list", F.flatten(F.col("categories")))
# 增加保护：flatten之后数组size>0才取最后一级，否则置null
df_cat = df_cat.withColumn(
    "main_cat",
    F.when(F.size(F.col("cat_list")) > 0, F.element_at(F.col("cat_list"), -1))
).filter(F.col("main_cat").isNotNull())

cat_inter = item_meta_join_df.join(
    df_cat.select("asin","main_cat"),
    on="asin",
    how="inner"
)
cat_agg = cat_inter.groupBy("main_cat").count().withColumnRenamed("count","inter_cnt")
print("各一级类目交互数量Top15：")
cat_agg.orderBy(F.col("inter_cnt").desc()).show(15)

# ========== L3‑5 salesRank销量排名探查【修复map无序取值bug】 ==========
print("\n【L3‑5 salesRank销量排名探查】")
df_rank = meta_valid.filter(F.col("salesRank").isNotNull())
# 只取key为Beauty的排名，其他类目直接丢弃，不随机取map_values第一个
df_rank = df_rank.withColumn("rank_value", F.col("salesRank")["Beauty"].cast("long"))
df_rank_valid = df_rank.filter(F.col("rank_value") > 0)

df_rank_valid.select(
    F.percentile_approx("rank_value", [0.25,0.5,0.75],10000).alias("rank_quantile")
).show()

rank_bin = df_rank_valid.withColumn(
    "rank_bin",
    F.when(F.col("rank_value") <= 1000, "top_1000")
    .when(F.col("rank_value") <=10000, "1000‑10000")
    .otherwise(">10000")
)
rank_bin_inter = item_meta_join_df.join(
    rank_bin.select("asin","rank_bin"),
    on="asin",
    how="inner"
)
print("\n不同销量排名档位交互数量：")
rank_bin_inter.groupBy("rank_bin").count().orderBy("rank_bin").show()

# ========== L3‑6 related关联商品探查【修复数据类型不匹配问题】 ==========
print("\n【L3‑6 related关联商品探查】")
df_related = meta_valid.filter(F.col("related").isNotNull())

# 直接使用 getField (或 . 语法) 访问 STRUCT 中的字段，判断数组是否非空
df_related = df_related\
    .withColumn("has_also_viewed", F.size(F.col("related").getField("also_viewed")) > 0)\
    .withColumn("has_also_bought", F.size(F.col("related").getField("also_bought")) > 0)

df_related.select(
    F.count(F.when(F.col("has_also_viewed"), True)).alias("has_also_viewed_cnt"),
    F.count(F.when(F.col("has_also_bought"), True)).alias("has_also_bought_cnt")
).show()

# ========== 释放全部缓存 ==========
meta_valid.unpersist()
inter_df.unpersist()
item_meta_join_df.unpersist()

print("\n==== L3 探查完成 ====")
