import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("AmazonBeauty_EDA_L2") \
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
from pyspark.sql import Window

print("=" * 70)
print("L2 行为与信号探查 — 评分、helpful、时间、文本、重复交互")
print("=" * 70)

# 性能优化：缓存交互表，避免多次全表扫描
inter_df.cache()
# 触发cache
_ = inter_df.count()

# ========== L2‑1 评分信号探查 ==========
print("\n【L2‑1 评分信号】")
inter_df.select(
    F.min("rating").alias("rating_min"),
    F.max("rating").alias("rating_max"),
    F.avg("rating").alias("rating_avg")
).show()

print("\n评分分布(各分数条数&占比)：")
total_inter = inter_df.count()
rating_dist = inter_df.groupBy("rating")\
    .agg(F.count("*").alias("cnt"))\
    .withColumn("ratio", F.round(F.col("cnt") / total_inter * 100, 2))\
    .orderBy("rating")
rating_dist.show()


# ========== L2‑2 helpful 投票数组探查 ==========
print("\n【L2‑2 helpful投票探查】")

print("helpful数组为空/不为空计数：")
inter_df.select(
    F.count(F.when(F.col("helpful").isNull(), True)).alias("helpful_null"),
    F.count(F.when(F.size(F.col("helpful")) == 0, True)).alias("helpful_empty_arr"),
    F.count(F.when(F.col("helpful").isNotNull() & (F.size(F.col("helpful"))==2), True)).alias("helpful_valid_2elem")
).show()

df_help = inter_df.filter(F.col("helpful").isNotNull())\
    .withColumn("help_useful", F.col("helpful")[0])\
    .withColumn("help_total", F.col("helpful")[1])

print("\nhelpful投票基础统计(有用数，总投票数)")
df_help.select(
    F.min("help_useful"),F.max("help_useful"),F.avg("help_useful"),
    F.min("help_total"),F.max("help_total"),F.avg("help_total")
).show()

df_help_rate = df_help.filter(F.col("help_total")>0)\
    .withColumn("help_rate", F.col("help_useful") / F.col("help_total"))
print("\nhelpful赞同率(有用/总投票)概要：")
df_help_rate.select(
    F.percentile_approx("help_rate", [0.25,0.5,0.75], 10000).alias("percentiles"),
    F.avg("help_rate").alias("avg_rate")
).show()


# ========== L2‑3 时间序列行为探查（新增相邻交互时间差检测） ==========
print("\n【L2‑3 交互时间时序探查】")
time_stat = inter_df.select(
    F.min("timestamp").alias("ts_min"),
    F.max("timestamp").alias("ts_max"),
    F.from_unixtime(F.min("timestamp")).alias("dt_min"),
    F.from_unixtime(F.max("timestamp")).alias("dt_max")
).collect()[0]
print(f"unix时间戳范围：{time_stat['ts_min']} ~ {time_stat['ts_max']}")
print(f"对应日期范围：{time_stat['dt_min']} ~ {time_stat['dt_max']}")

print("\n按年统计交互数量：")
inter_df.withColumn("year", F.year(F.from_unixtime("timestamp")))\
    .groupBy("year").count().orderBy("year").show()

# 新增：同一用户相邻评论时间差，检测1分钟内密集行为
print("\n检测：同一用户相邻交互间隔小于60秒的行为对数(疑似批量/刷评信号)")
w_user_time = Window.partitionBy("user_id").orderBy("timestamp")
short_interval_cnt = inter_df\
    .withColumn("prev_ts", F.lag("timestamp").over(w_user_time))\
    .withColumn("time_diff_sec", F.col("timestamp") - F.col("prev_ts"))\
    .filter(F.col("time_diff_sec") < 60)\
    .count()
print(f"相邻交互间隔<60s行为对数：{short_interval_cnt}")


# ========== L2‑4 文本字段探查 reviewText / summary ==========
print("\n【L2‑4 评论文本字段探查】")
df_textlen = inter_df\
    .withColumn("review_len", F.length(F.col("reviewText")))\
    .withColumn("summary_len", F.length(F.col("summary")))

print("reviewText、summary长度统计(空字符串也会被统计长度=0)")
df_textlen.select(
    F.percentile_approx("review_len", [0,0.25,0.5,0.75,1], 10000).alias("review_len_quantile"),
    F.percentile_approx("summary_len", [0,0.25,0.5,0.75,1], 10000).alias("summary_len_quantile")
).show()


# ========== L2‑5 user‑item重复交互探查：修复别名bug ==========
print("\n【L2‑5 user‑item重复交互对探查】")
# 修复：给count结果列命名 ui_record_cnt
user_item_cnt = inter_df.groupBy("user_id","item_id").agg(F.count("*").alias("ui_record_cnt"))

ui_stat = user_item_cnt.select(
    F.count(F.when(F.col("ui_record_cnt")==1, True)).alias("ui_once"),
    F.count(F.when(F.col("ui_record_cnt")>1, True)).alias("ui_multi"),
    F.sum(F.when(F.col("ui_record_cnt")>1, F.col("ui_record_cnt"))).alias("total_dup_rows")
).collect()[0]

print(f"仅交互1次的(user,item)对数：{ui_stat['ui_once']}")
print(f"发生多次交互的(user,item)对数：{ui_stat['ui_multi']}")
print(f"重复交互带来的总冗余行数：{ui_stat['total_dup_rows']}")

print("\n重复交互Top5样例(user_id,item_id,评论次数):")
user_item_cnt.filter(F.col("ui_record_cnt")>1).orderBy(F.col("ui_record_cnt").desc()).show(5)

# 释放cache
inter_df.unpersist()
print("\n==== L2 探查完成 ====")
