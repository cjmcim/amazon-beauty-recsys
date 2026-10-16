import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("AmazonBeauty_EDA_Optimized") \
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

# 查看数据
pd_review = inter_raw.limit(5).toPandas()
print("\n===== 用户交互表前5条数据 =====")
display(pd_review)

# 统一别名，但保留 helpful / reviewText / summary 供探查
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
print(f"交互表总行数: {inter_df.count():,}")
print(f"独立用户数: {inter_df.select('user_id').distinct().count():,}")
print(f"独立商品数: {inter_df.select('item_id').distinct().count():,}")


# 读取
meta_raw = spark.read.json(META_PATH)

# 查看
pd_meta = meta_raw.limit(5).toPandas()
print("\n===== 商品元数据表前5条数据 =====")
display(pd_meta)


meta_raw = meta_raw.cache()

# 统计损坏行，不直接删除，EDA先看数量+
corrupt_cnt = meta_raw.filter(F.col("_corrupt_record").isNotNull()).count()
print(f"meta 表损坏行(_corrupt_record非空)数量: {corrupt_cnt}")

# 探查使用过滤后的有效表，但原始 meta_raw 保留以备对比
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
print(f"meta 有效商品数: {meta_df.count():,}")

# 探查：是否存在重复 item_id（不去重，只看数量）
item_dup = meta_df.groupBy("item_id").count().filter(F.col("count") > 1)
print(f"meta 中重复 item_id 数量: {item_dup.count()}")


print("=" * 70)
print("L0 数据理解：单表质量 + 跨表匹配校验")
print("=" * 70)

# ========== L0‑1 交互表单表质量检查 ==========
print("\n【L0‑1 交互表单表质量】")

# 整体部分
print("\n 1.1 整体：")

print("\n 1.1-1 交互表样本预览")
inter_df.show(5, truncate=60)

print("\n 1.1-2 数据类型 & 字段理解（schema）")
inter_df.printSchema()

print("\n 1.1-3 全字段NULL缺失统计")
inter_df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in inter_df.columns]
).show()

print("\n 1.1-4 检测重复数据")
dup_rows_cnt = inter_df.count() - inter_df.distinct().count()
print(f"完全重复交互行数量：{dup_rows_cnt}")


# 分字段部分
print("\n 1.2 各字段：")

print("\n 1.2-1 文本类数据（reviewText summary）")
print("\n 空字符串检测（空文本isNull无法识别）")
inter_df.select(
    F.count(F.when(F.col("reviewText") == "", True)).alias("reviewText_empty_str"),
    F.count(F.when(F.col("summary") == "", True)).alias("summary_empty_str")
).show()

print("\n1.2‑2 数组空值检测（helpful）")
inter_df.select(
    F.count(F.when(F.col("helpful").isNull(), True)).alias("helpful_null"),
    F.count(F.when(F.size(F.col("helpful")) == 0, True)).alias("helpful_empty_array(空数组[])"),
    F.count(F.when((F.size(F.col("helpful")) == 2), True)).alias("helpful_valid_len2(标准[a,b])"),
    F.count(F.when((F.size(F.col("helpful")) != 0) & (F.size(F.col("helpful")) != 2), True)).alias("helpful_abnormal_len(不空长度也不为2的异常数组)")
).show()


print("\n 1.2-3 数值类数据（rating timestamp）")
print("\n 基础统计：")
print("【用户评分统计】")
inter_df.select("rating").summary().show()
print("【评分分布】")
inter_df.groupBy("rating").count().orderBy("rating").show()

print("\n 值域校验：rating、timestamp合法范围")
inter_df.select(
    F.min("rating").alias("min_rating"),
    F.max("rating").alias("max_rating"),
    F.mean("rating").alias("avg_rating"),
    F.count(F.when((F.col("rating") < 1) | (F.col("rating") > 5), True)).alias("rating_out_of_range_cnt"),
    F.min("timestamp").alias("min_ts"),
    F.max("timestamp").alias("max_ts"),
    F.count(F.when((F.col("timestamp") <= 0), True)).alias("ts_invalid_cnt"),
    F.from_unixtime(F.min("timestamp")).alias("earliest_time"),
    F.from_unixtime(F.max("timestamp")).alias("latest_time")
).show(truncate=False)


# ========== L0‑2 Meta元数据表单表质量检查 ==========
print("\n【L0‑2 Meta元数据表单表质量】")

# ---------- 2.1 整体部分 ----------
print("\n 2.1 整体：")

print("\n 2.1-1 meta表样本预览")
meta_df.show(5, truncate=30)

print("\n 2.1-2 数据类型 & 字段理解（schema）")
meta_df.printSchema()

print("\n 2.1-3 全字段NULL缺失统计")
meta_df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in meta_df.columns]
).show()

print("\n 2.1-4 检测重复数据（主键item_id重复）")
meta_dup_item = meta_df.groupBy("item_id").count().filter(F.col("count") > 1)
print(f"meta内部重复asin数量：{meta_dup_item.count()}")
if meta_dup_item.count() > 0:
    print("重复asin样例：")
    meta_dup_item.show(5)

# ---------- 2.2 各字段部分 ----------
print("\n 2.2 各字段：")

# 2.2-1 文本类数据（title / brand / description）
print("\n 2.2-1 文本类数据（title brand description）")
print("\n 空字符串检测（空文本isNull无法识别）")
meta_df.select(
    F.count(F.when(F.col("title") == "", True)).alias("title_empty_str"),
    F.count(F.when(F.col("brand") == "", True)).alias("brand_empty_str"),
    F.count(F.when(F.col("description") == "", True)).alias("description_empty_str")
).show()

# 2.2-2 数值类数据（price）
print("\n 2.2-2 数值类数据（price）")
# 先判断price实际类型，兼容 string(带$) 和 double
price_field = meta_df.schema["price"]
print(f"price 字段原始类型: {price_field.dataType}")

if isinstance(price_field.dataType, StringType):
    # 字符串格式：去除 $ 和逗号，再转 double（仅用于探查，不修改原表）
    price_probe = meta_df.filter(F.col("price").isNotNull() & (F.col("price") != "")) \
        .withColumn("price_num",
            F.regexp_replace(F.regexp_replace("price", "\\$", ""), ",", "").cast("double"))
else:
    price_probe = meta_df.filter(F.col("price").isNotNull()) \
        .withColumn("price_num", F.col("price"))

print("\n 基础统计：")
price_probe.select("price_num").summary().show()

print("\n 值域校验：")
price_probe.select(
    F.min("price_num").alias("min_price"),
    F.max("price_num").alias("max_price"),
    F.mean("price_num").alias("mean_price"),
    F.count(F.when(F.col("price_num") <= 0, True)).alias("price_le_0_cnt"),
    F.count(F.when(F.col("price_num") > 10000, True)).alias("price_gt_10000_cnt(疑似异常高价)")
).show(truncate=False)

# 2.2-3 数组类数据（categories）
print("\n 2.2-3 数组类数据（categories 二维嵌套数组）")
print("\n null / 空数组 / 有效数组 统计：")
meta_df.select(
    F.count(F.when(F.col("categories").isNull(), True)).alias("categories_null"),
    F.count(F.when(F.size(F.col("categories")) == 0, True)).alias("categories_empty_array(空数组[])"),
    F.count(F.when(F.size(F.col("categories")) > 0, True)).alias("categories_valid(非空数组)")
).show()

print("\n 外层数组长度分布（一件商品对应几条类目路径）：")
meta_df.select(F.size("categories").alias("cat_outer_len")) \
    .groupBy("cat_outer_len").count().orderBy("cat_outer_len").show()

print("\n 展开后类目路径长度分布（每条路径有几级类目）：")
meta_df.select(F.explode("categories").alias("cat_path")) \
    .select(F.size("cat_path").alias("path_len")) \
    .groupBy("path_len").count().orderBy("path_len").show()

print("\n 一级类目 Top20：")
meta_df.select(F.explode("categories").alias("cat_path")) \
    .select(F.col("cat_path")[0].alias("main_cate")) \
    .groupBy("main_cate").count().orderBy(F.desc("count")).show(20, truncate=False)

# 2.2-4 结构体类数据（salesRank / related）
print("\n 2.2-4 结构体类数据（salesRank / related）")

print("\n salesRank.Beauty 美妆类目销量排名覆盖率：")
meta_df.select(
    F.count(F.when(F.col("salesRank.Beauty").isNotNull(), True)).alias("has_beauty_rank"),
    F.count(F.when(F.col("salesRank.Beauty").isNull(), True)).alias("no_beauty_rank")
).show()

print("\n related 关联商品各子字段覆盖率：")
related_exprs = []
for sub in ["also_bought", "also_viewed", "bought_together", "buy_after_viewing"]:
    col_path = f"related.{sub}"
    related_exprs.append(
        F.count(F.when(F.col(col_path).isNotNull(), True)).alias(f"{sub}_not_null")
    )
meta_df.select(*related_exprs).show()


# ========== L0‑3 跨表关联匹配校验 ==========
print("\n【L0‑3 交互表与Meta表跨表匹配情况】")

# ---------- 3.1 整体匹配统计 ----------
print("\n 3.1 整体匹配统计：")
reviews_asins = inter_df.select("item_id").distinct()
meta_asins = meta_df.select("item_id").distinct()

review_item_total = reviews_asins.count()
meta_item_total = meta_asins.count()
missing_meta = reviews_asins.join(meta_asins, on="item_id", how="left_anti")
missing_meta_cnt = missing_meta.count()

print(f"交互表独立商品总数：{review_item_total}")
print(f"meta表独立商品总数：{meta_item_total}")
print(f"交互有、meta缺失的商品数：{missing_meta_cnt}")
print(f"商品元数据匹配率：{(1 - missing_meta_cnt / review_item_total) * 100:.2f}%")

# 反向：meta中有、但交互表没有的商品（纯元数据，无用户行为）
extra_meta = meta_asins.join(reviews_asins, on="item_id", how="left_anti")
print(f"meta有、但交互表无行为的商品数：{extra_meta.count()}")

# ---------- 3.2 缺失商品样例 ----------
print("\n 3.2 缺失商品样例：")
if missing_meta_cnt > 0:
    print("交互表中有、但meta缺失的商品样例：")
    missing_meta.show(10)
else:
    print("无缺失，交互表所有商品均能在meta中找到。")