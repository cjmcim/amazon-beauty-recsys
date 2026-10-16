import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("AmazonBeauty_Clean") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1") \
    .getOrCreate()

# 新增：关闭本地文件权限校验，解决Windows HADOOP_HOME报错
spark.conf.set("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
spark.conf.set("spark.hadoop.fs.file.permissions.enabled", "false")

spark.sparkContext.setLogLevel("WARN")

# ========== 路径配置（改成你的实际路径） ==========
INTER_PATH = "D:\Amazon_beauty_project\Beauty_5.json"
META_PATH  = "D:\Amazon_beauty_project\meta_Beauty.json"
OUTPUT_DIR = "D:\Amazon_beauty_project"   # 清洗后数据保存目录

# ========== 清洗策略开关 ==========
DEDUP_USER_ITEM = True      # (user,item)重复交互是否去重（True:只保留最新一条）
KEEP_MISSING_META = True    # meta缺失的商品交互是否保留（True保留，商品特征为null）



# 读取原始数据，全程不动原始文件
inter_raw = spark.read.json(INTER_PATH)
meta_raw = spark.read.json(META_PATH)

print(f"原始交互表行数: {inter_raw.count():,}")
print(f"原始meta表行数: {meta_raw.count():,}")

# ========== 交互表基础别名映射 ==========
inter_df = inter_raw.select(
    F.col("reviewerID").alias("user_id"),
    F.col("asin").alias("item_id"),
    F.col("overall").cast("float").alias("rating"),
    F.col("unixReviewTime").cast("long").alias("timestamp"),
    F.col("helpful"),
    F.col("reviewText"),
    F.col("summary")
)



print("=" * 60)
print("【交互表清洗】")
print("=" * 60)

n_before = inter_df.count()
print(f"清洗前交互行数: {n_before:,}")

# ---------- 3.1 过滤非法评分（rating 必须在[1,5]） ----------
inter_clean = inter_df.filter(
    F.col("user_id").isNotNull() &
    F.col("item_id").isNotNull() &
    F.col("rating").isNotNull() &
    F.col("rating").between(1, 5) &
    F.col("timestamp").isNotNull() &
    (F.col("timestamp") > 0)
)
n_after_filter = inter_clean.count()
print(f"① 过滤null/非法rating/非法timestamp后: {n_after_filter:,} "
      f"(剔除 {n_before - n_after_filter:,} 行)")

# ---------- 3.2 提取helpful数组（已确认全部为长度2） ----------
inter_clean = inter_clean.withColumn("helpful_useful", F.col("helpful")[0]) \
    .withColumn("helpful_total", F.col("helpful")[1]) \
    .drop("helpful")

# ---------- 3.3 (user,item)重复交互去重 ----------
if DEDUP_USER_ITEM:
    # 每个(user,item)只保留timestamp最新的一条
    w = Window.partitionBy("user_id", "item_id").orderBy(F.desc("timestamp"))
    inter_clean = inter_clean.withColumn("rn", F.row_number().over(w)) \
        .filter(F.col("rn") == 1) \
        .drop("rn")
    n_after_dedup = inter_clean.count()
    print(f"② (user,item)去重后: {n_after_dedup:,} (剔除 {n_after_filter - n_after_dedup:,} 条重复)")

# ---------- 3.4 文本空字符串处理（保留但标记，供特征工程决定是否使用） ----------
inter_clean = inter_clean.withColumn(
    "reviewText_empty", F.when(F.length(F.col("reviewText")) == 0, 1).otherwise(0)
)

inter_clean.cache()
n_clean = inter_clean.count()
print(f"\n✅ 交互表清洗完成，最终行数: {n_clean:,}")
print(f"   清洗总剔除: {n_before - n_clean:,} 行 ({(n_before-n_clean)/n_before*100:.2f}%)")


print("=" * 60)
print("【meta表清洗】")
print("=" * 60)

meta_raw.cache()
raw_count = meta_raw.count()  # 触发解析并缓存到内存
print(f"meta原始总行数: {raw_count:,}")

# ---------- 4.1 过滤损坏行 ----------
# 此时因为数据已经缓存，Spark 可以直接在内存中安全地访问 _corrupt_record 列
meta_clean = meta_raw.filter(F.col("_corrupt_record").isNull())
clean_count = meta_clean.count()

print(f"① 过滤_corrupt_record后: {clean_count:,} "
      f"(剔除 {raw_count - clean_count:,} 行)")

# ---------- 4.2 选择字段 + 提取嵌套结构 ----------
meta_clean = meta_clean.select(
    F.col("asin").alias("item_id"),
    F.col("brand"),
    F.col("price"),
    F.col("categories"),
    F.col("salesRank.Beauty").alias("sales_rank_beauty"),
    F.col("related.also_bought").alias("also_bought_items"),
    F.col("related.also_viewed").alias("also_viewed_items")
)

# ---------- 4.3 处理价格：字符串去$、转double、填充缺失 ----------
# 判断price类型（兼容string/double）
price_field_type = [f.dataType for f in meta_clean.schema.fields if f.name == "price"][0]
if isinstance(price_field_type, StringType):
    meta_clean = meta_clean.withColumn(
        "price",
        F.regexp_replace(F.regexp_replace("price", "\\$", ""), ",", "").cast("double")
    )

# 缺失/非正价格 → -1 标记（特征工程阶段处理）
meta_clean = meta_clean.withColumn(
    "price", F.when(F.col("price").isNotNull() & (F.col("price") > 0), F.col("price")).otherwise(-1.0)
)

# ---------- 4.4 品牌空值处理 ----------
meta_clean = meta_clean.withColumn("brand", F.trim(F.col("brand"))) \
    .withColumn("brand", F.when(F.col("brand") == "", "unknown").otherwise(F.col("brand"))) \
    .fillna({"brand": "unknown"})

# ---------- 4.5 类目解析：flatten后取最后一级为主类目（带空数组保护） ----------
meta_clean = meta_clean.withColumn("cat_list", F.flatten(F.col("categories"))) \
    .withColumn(
        "main_category",
        F.when(F.size(F.col("cat_list")) > 0, F.element_at(F.col("cat_list"), -1)).otherwise("unknown")
    ) \
    .drop("categories", "cat_list")

# ---------- 4.6 meta内部重复asin去重 ----------
meta_clean = meta_clean.dropDuplicates(["item_id"])

meta_clean.cache()
print(f"\n✅ meta表清洗完成，最终行数: {meta_clean.count():,}")
meta_clean.printSchema()
meta_clean.show(5, truncate=100)



print("=" * 60)
print("【跨表关联，清洗逻辑执行（Windows调试版，不落地保存文件）】")
print("=" * 60)

# ---------- 5.1 交互表左关联meta表 ----------
df_final = inter_clean.join(meta_clean, on="item_id", how="left")

# 处理meta缺失商品的标记（对应L0‑3匹配率结论）
df_final = df_final.withColumn(
    "meta_missing", F.when(F.col("brand").isNull(), 1).otherwise(0)
)

KEEP_MISSING_META = True # 你按需切换True/False
if KEEP_MISSING_META:
    # 1. 先用fillna填充基础常量字段（字符串、数字）
    df_final = df_final.fillna({
        "brand": "unknown",
        "main_category": "unknown",
        "price": -1.0,
        "sales_rank_beauty": -1
    })
    # 2. 数组字段单独用when填充空数组，不能放fillna
    empty_arr = F.array().cast("array<string>")
    df_final = df_final\
        .withColumn("also_bought_items", F.when(F.col("also_bought_items").isNull(), empty_arr).otherwise(F.col("also_bought_items")))\
        .withColumn("also_viewed_items", F.when(F.col("also_viewed_items").isNull(), empty_arr).otherwise(F.col("also_viewed_items")))
else:
    # 丢弃meta缺失的交互
    n_before_join = df_final.count()
    df_final = df_final.filter(F.col("meta_missing") == 0)
    print(f"丢弃meta缺失交互: {n_before_join - df_final.count():,} 行")

df_final.cache()
print(f"关联后最终总行数: {df_final.count():,}")
print(f"meta缺失的交互数: {df_final.filter(F.col('meta_missing') == 1).count():,}")

# ========== 注释掉parquet写入，Windows下跳过持久化 ==========
# df_final.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/inter_clean")
# meta_clean.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/meta_clean")

# ---------- 新增：样例校验，肉眼检查清洗效果 ----------
print("\n===== 清洗后数据样例预览（10行） =====")
df_final.select(
    "user_id","item_id","rating","timestamp",
    "brand","main_category","price","meta_missing",
    "also_bought_items","also_viewed_items"
).show(10, truncate=80)

# ---------- 新增：校验：确认数组字段无null（验证填充生效） ----------
print("\n===== 校验：also_bought_items为null的行数 =====")
print(df_final.filter(F.col("also_bought_items").isNull()).count())
print("===== 校验：also_viewed_items为null的行数 =====")
print(df_final.filter(F.col("also_viewed_items").isNull()).count())

# ---------- 释放缓存 ----------
inter_clean.unpersist()
meta_clean.unpersist()
df_final.unpersist() # 记得释放df_final缓存，防止内存占用过高



print("=" * 60)
print("【清洗效果复核】（Windows调试版，内存校验，不读磁盘文件）")
print("=" * 60)

# ========= 改动：直接复用内存中的df_final、meta_clean，不再read.parquet =========
df_check = df_final
meta_check = meta_clean

print(f"清洗后交互表行数: {df_check.count():,}")
print(f"清洗后meta表行数: {meta_check.count():,}")
print(f"清洗后独立用户数: {df_check.select('user_id').distinct().count():,}")
print(f"清洗后独立商品数: {df_check.select('item_id').distinct().count():,}")

# 关键校验：确认没有重复(user,item)（若开启了去重）
if DEDUP_USER_ITEM:
    dup_cnt = df_check.groupBy("user_id", "item_id").count().filter(F.col("count") > 1).count()
    print(f"(user,item)重复对数量: {dup_cnt} (应为0)")

# 确认没有非法评分
print(f"非法评分(不在1-5)数量: {df_check.filter(~F.col('rating').between(1,5)).count()}")
# 确认meta表字段
print(f"meta表字段: {meta_check.columns}")
