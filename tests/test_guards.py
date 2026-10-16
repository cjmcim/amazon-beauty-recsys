"""P1 数据守卫测试：守护最容易改坏、改坏后最致命的 4 条数据规则。

G1 clean 无重复对        —— clean_interaction 的 (user,item) 去重语义（保留最新）
G2 split 时间严格递增    —— 时序切分不重叠 + 冷用户过滤 + 三份无交集
G3 PIT 不泄露（核心）    —— train 窗口只看严格早于自己的历史；
                            val/test/负样本取 <= 样本时间的最近快照，绝不取未来
G4 类别编码复用          —— StringIndexer 只在 train 拟合、test 复用 transform；
                            apply_cat_codes 以 train 的 cat_map 编码 test（未见→-1）

设计原则：
- 全部用构造的小数据（几十行），不碰真实数据 / checkpoint，结果确定、秒级；
- 被测对象是真实业务函数，不是测试副本；
- 需要 Spark 的用例通过 session 级 fixture（tests/conftest.py）只起一次 Spark。
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pyspark.sql import functions as F

from src.recsys.config import RecallConfig, SplitConfig
from src.recsys.data.clean import clean_interaction
from src.recsys.data.split import run_split
from src.recsys.features.user_features import (
    attach_val_test_feat,
    build_user_dynamic,
    build_user_ts_feat,
)
from src.recsys.rank.features import apply_cat_codes, apply_recall_indexer
from src.recsys.rank.pit import attach_features_pit
from src.recsys.recall.engine import RecallBase

# ===========================================================================
# G1 clean 无重复对
# ===========================================================================
_INTER_SCHEMA = ("user_id string, item_id string, rating double, timestamp long, "
                 "helpful array<string>, reviewText string")


def _make_inter(spark):
    """含重复对 / 非法评分 / 非法评分 null 的构造交互表。"""
    return spark.createDataFrame(
        [
            ("u1", "i1", 5.0, 100, ["1", "1"], "txt1"),   # (u1,i1) 旧记录
            ("u1", "i1", 4.0, 200, ["1", "1"], "txt2"),   # (u1,i1) 新记录 → 应保留
            ("u1", "i2", 3.0, 150, ["1", "1"], "txt3"),
            ("u2", "i1", 5.0, 300, ["1", "1"], "txt4"),
            ("u3", "i3", 6.0, 400, ["1", "1"], "txt5"),   # 非法评分 6 → 过滤
            ("u3", "i4", None, 400, ["1", "1"], "txt6"),  # 非法评分 null → 过滤
        ],
        _INTER_SCHEMA,
    )


class TestCleanGuard:
    def test_clean_filters_invalid_and_dedups(self, spark):
        out = clean_interaction(_make_inter(spark), dedup_user_item=True)
        # 4 条合法（u1i1×2 + u1i2 + u2i1）→ 去重后 3 条
        assert out.count() == 3

    def test_clean_no_duplicate_pairs(self, spark):
        out = clean_interaction(_make_inter(spark), dedup_user_item=True)
        dup = (
            out.groupBy("user_id", "item_id").count()
            .filter(F.col("count") > 1).count()
        )
        assert dup == 0

    def test_clean_keeps_latest_timestamp(self, spark):
        out = clean_interaction(_make_inter(spark), dedup_user_item=True)
        rows = out.filter((F.col("user_id") == "u1") & (F.col("item_id") == "i1")).collect()
        assert len(rows) == 1
        assert rows[0]["timestamp"] == 200  # 去重语义：保留 timestamp 最新一条


# ===========================================================================
# G2 split 时间严格递增
# ===========================================================================
def _make_split_input(spark):
    """1000 条 (u,i) 全局唯一的交互：40 个用户均匀分布在整个时间轴
    （保证切分后 val/test 用户都 ⊆ train 用户、三份均非空、时间不重叠）；
    + 1 个只出现在 test 时段的冷用户（应被过滤）。

    注意：不能用「用户按时间块连续分布」的构造（如 k//5 分组），
    那样 val/test 时段的用户天然是 train 未出现的新用户，会被
    inner join 全部过滤导致 val/test 为空。
    """
    rows = [
        (f"u{k % 40}", f"i{k}", 4.0 + (k % 2), 1000 + k * 100)
        for k in range(1000)
    ]
    rows.append(("u_cold", "i_cold", 5.0, 200000))  # 冷用户：时间远超 test 时段
    return spark.createDataFrame(
        rows, "user_id string, item_id string, rating double, timestamp long"
    )


class TestSplitGuard:
    @pytest.fixture()
    def split_out(self, spark):
        cfg = SimpleNamespace(split=SplitConfig(
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        ))
        return run_split(_make_split_input(spark), cfg)

    def test_time_strictly_increasing(self, split_out):
        """train / val / test 时间严格递增（train 全部早于 val，val 全部早于 test）。"""
        t_max = split_out["train"].agg(F.max("timestamp")).first()[0]
        v_min = split_out["val"].agg(F.min("timestamp")).first()[0]
        assert t_max < v_min
        v_max = split_out["val"].agg(F.max("timestamp")).first()[0]
        te_min = split_out["test"].agg(F.min("timestamp")).first()[0]
        assert v_max < te_min

    def test_cold_user_filtered(self, split_out):
        """只出现在 test 时段的冷用户必须被过滤（val/test 用户 ⊆ train 用户）。"""
        assert split_out["test"].filter(F.col("user_id") == "u_cold").count() == 0
        assert split_out["val"].filter(F.col("user_id") == "u_cold").count() == 0

    def test_no_pair_overlap(self, split_out):
        """同一 (user,item) 交互不得同时出现在两份数据（切分不重叠）。"""
        pairs = {
            name: set(
                (r[0], r[1])
                for r in split_out[name].select("user_id", "item_id").distinct().collect()
            )
            for name in ("train", "val", "test")
        }
        assert not (pairs["train"] & pairs["val"])
        assert not (pairs["train"] & pairs["test"])
        assert not (pairs["val"] & pairs["test"])

    def test_val_test_users_subset_of_train(self, split_out):
        """val/test 的所有用户都必须出现在 train 中（防冷用户评估）。"""
        train_users = {
            r[0] for r in split_out["train"].select("user_id").distinct().collect()
        }
        for name in ("val", "test"):
            users = {
                r[0] for r in split_out[name].select("user_id").distinct().collect()
            }
            assert users <= train_users


# ===========================================================================
# G3 PIT 不泄露（核心）
# ===========================================================================
_USER_SCHEMA = ("user_id string, item_id string, timestamp long, rating double, "
                "reviewText string, brand string, main_category string")


def _make_train_pit(spark):
    """用户 u1 有 4 条严格递增的交互（t=10/20/30/40）。"""
    return spark.createDataFrame(
        [
            ("u1", "i1", 10, 5.0, "a", "B1", "C1"),
            ("u1", "i2", 20, 4.0, "b", "B2", "C2"),
            ("u1", "i3", 30, 3.0, "c", "B3", "C3"),
            ("u1", "i4", 40, 2.0, "d", "B4", "C4"),
        ],
        _USER_SCHEMA,
    )


def _make_item_feat(spark):
    """构造物品特征表：列须与真实 item_feat（build_item_feature 产出）一致，
    否则 attach_features_pit 的 fillna 硬编码列会报 UNRESOLVED_COLUMN。"""
    return spark.createDataFrame(
        [
            ("i2", "C1", "B1", 10, 4.5, 0.5, 8, 3.2),
            ("i3", "C9", "B9", 20, 3.0, 1.0, 5, 2.1),
            ("i9", "C1", "B1", 0, 0.0, 0.0, 0, 0.0),
        ],
        "item_id string, main_category string, brand string, "
        "item_train_cnt int, item_train_avg_rating double, item_train_rating_std double, "
        "item_train_pos_cnt int, item_train_avg_text_len double",
    )


class TestPitGuard:
    def test_train_window_excludes_current_row(self, spark):
        """train PIT 窗口 rowsBetween(unboundedPreceding,-1)：
        t=40 行只能看到前 3 条历史（不含自己的 rating=2 / item=i4）。"""
        dyn = build_user_dynamic(_make_train_pit(spark))
        rows = {
            r["timestamp"]: r
            for r in dyn.orderBy("timestamp").collect()
        }
        assert rows[10]["user_hist_cnt"] == 0      # 无历史
        assert rows[10]["user_avg_rating"] is None
        assert rows[20]["user_hist_cnt"] == 1
        assert rows[20]["user_avg_rating"] == 5.0  # 只看 t=10
        assert rows[30]["user_hist_cnt"] == 2
        assert rows[30]["user_avg_rating"] == 4.5
        assert rows[40]["user_hist_cnt"] == 3
        assert rows[40]["user_avg_rating"] == 4.0  # (5+4+3)/3，不含自己的 2
        assert rows[40]["user_pos_cnt"] == 2       # 前三条中 rating>=4 的条数
        assert sorted(rows[40]["user_hist_items"]) == ["i1", "i2", "i3"]  # 不含 i4

    def test_val_test_matches_nearest_snapshot_not_future(self, spark):
        """val 样本 label_ts=25 必须匹配 hist_ts=20 的最近快照（cnt=1），
        绝不能匹配 t=30 的未来快照（cnt=2）。"""
        train = _make_train_pit(spark)
        user_ts_feat, _ = build_user_ts_feat(build_user_dynamic(train))
        val_samples = spark.createDataFrame(
            [("u1", "i2", 25)], "user_id string, item_id string, timestamp long"
        )
        feat = attach_val_test_feat(val_samples, user_ts_feat, _make_item_feat(spark), "验证集")
        row = feat.first()
        assert row["user_hist_cnt"] == 1   # 匹配 hist_ts=20 的快照
        assert row["user_avg_rating"] == 5.0

    def test_rank_neg_sample_matches_nearest_snapshot(self, spark):
        """精排负样本（pit_time=25）同样取 <= pit_time 的最近快照（cnt=1）；
        无任何历史快照的用户（u_new）在 filter(pit_time > hist_ts) 时因 null
        比较被整行丢弃——真实流水线中 val/test 用户 ⊆ train 用户不会触发，
        这是 PIT 严格性的边界行为（宁可丢弃，绝不取到未来/未知特征）。"""
        train = _make_train_pit(spark)
        user_ts_feat, _ = build_user_ts_feat(build_user_dynamic(train))
        samples = spark.createDataFrame(
            [("u1", "i3", 25, 0), ("u_new", "i9", 100, 0)],
            "user_id string, item_id string, pit_time long, label int",
        )
        feat = attach_features_pit(samples, user_ts_feat, _make_item_feat(spark))
        rows = {r["user_id"]: r for r in feat.collect()}
        assert rows["u1"]["user_hist_cnt"] == 1    # 匹配 hist_ts=20，非未来 t=30
        assert "u_new" not in rows                 # 无历史快照 → 整行丢弃


# ===========================================================================
# G4 类别编码复用
# ===========================================================================
def _make_enc_train(spark):
    return spark.createDataFrame(
        [
            ("u1", "i1", 5.0, 100, "B1", "C1"),
            ("u2", "i1", 4.0, 200, "B1", "C1"),
            ("u2", "i2", 3.0, 300, "B2", "C2"),
        ],
        "user_id string, item_id string, rating double, timestamp long, "
        "brand string, main_category string",
    )


def _make_enc_val(spark):
    return spark.createDataFrame(
        [
            ("u1", "i3", 5.0, 400, "B1", "C1"),   # brand B1 在 train 出现过
            ("u3", "i4", 4.0, 500, "BX", "CX"),   # brand BX 未见过
        ],
        "user_id string, item_id string, rating double, timestamp long, "
        "brand string, main_category string",
    )


class TestEncodingGuard:
    def test_indexer_fit_on_train_only_and_reused(self, spark):
        """StringIndexer 只在 train 拟合；val/test 复用 transform：
        同一 brand 编码一致，未见 brand 进 unknown 桶（keep 语义）。"""
        train = _make_enc_train(spark)
        val = _make_enc_val(spark)
        cfg = RecallConfig(
            itemcf_top_k=5, hot_top_k=5, final_recall_k=10,
            min_cooc=1, eval_k_list=[5],
        )
        base = RecallBase(train, cfg)
        base.fit_indexers(train)

        train_enc = apply_recall_indexer(train, base.brand_model, base.cate_model)
        val_enc = apply_recall_indexer(val, base.brand_model, base.cate_model)

        train_b1 = train_enc.filter(F.col("brand") == "B1").first()["brand_id"]
        val_b1 = val_enc.filter(F.col("brand") == "B1").first()["brand_id"]
        assert train_b1 == val_b1  # 同一 brand 在 train/val 编码一致

        # keep 语义：未见类别映射到 labels.size（最后一个 unknown 桶）
        val_bx = val_enc.filter(F.col("brand") == "BX").first()["brand_id"]
        assert val_bx == len(base.brand_model.labels)

    def test_apply_cat_codes_reuses_train_map(self):
        """apply_cat_codes 以 train 的 cat_map 编码 test：
        同类别同 code；test 未见类别 → codes=-1（树模型当独立分支）。"""
        train_pdf = pd.DataFrame({
            "brand": pd.Categorical(["B1", "B2", "B1"]),
        })
        val_pdf = pd.DataFrame({
            "brand": pd.Categorical(["B1", "BX"]),
        })
        cat_map = apply_cat_codes(train_pdf, ["brand"])
        apply_cat_codes(val_pdf, ["brand"], cat_map)

        assert train_pdf["brand"].tolist() == [0, 1, 0]   # B1→0, B2→1
        assert val_pdf["brand"].tolist() == [0, -1]       # B1→0, BX 未见→-1
        assert cat_map == {"brand": ["B1", "B2"]}
