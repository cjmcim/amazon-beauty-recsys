"""第 2 步：阶段数据量采集单元测试（纯逻辑，无需 Spark）。

覆盖：
1. _df_count / _distinct_count：pandas / None / 非法列输入行为；
2. STAGE_VOLUME_COLLECTORS：pandas 伪产物下各 stage 的输出键齐全、值正确；
3. ExperimentManager.finalize：stage_volumes 并入 run_report.json 的 stages.rows；
   未传 stage_volumes 时行为不变（向后兼容）；resume 的 None 耗时不计入总时长。
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.recsys.experiment import ExperimentManager
from src.recsys.pipeline.runner import (
    STAGE_VOLUME_COLLECTORS,
    _df_count,
    _distinct_count,
    collect_stage_volume,
)


def _fake_inter(n=100, users=10, items=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "user_id": rng.integers(0, users, n),
        "item_id": rng.integers(0, items, n),
    })


# -------------------------------------------------------------- 基础计数函数
def test_df_count_pandas_and_none():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert _df_count(df) == 3
    assert _df_count(None) is None
    assert _df_count(123) is None


def test_distinct_count_pandas_and_none():
    df = pd.DataFrame({"user_id": [1, 1, 2, 3]})
    assert _distinct_count(df, "user_id") == 3
    assert _distinct_count(df, "no_such_col") is None
    assert _distinct_count(None, "user_id") is None


# ---------------------------------------------------------- 各 stage 采集器
def test_collect_clean_volume():
    arts = {"inter": _fake_inter(n=100, users=10, items=20),
            "meta": _fake_inter(n=25, users=5, items=15)}
    vol = collect_stage_volume("clean", {}, arts)
    assert vol == {"inter_rows": 100, "meta_rows": 25, "users": 10, "items": 20}


def test_collect_split_volume():
    # 显式构造确定性数据，保证 nunique 可预期
    arts = {
        "train": pd.DataFrame({"user_id": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]}),          # 9 用户
        "val": pd.DataFrame({"user_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),            # 10 用户
        "test": pd.DataFrame({"user_id": [0, 0, 0, 1, 2, 3]}),                       # 4 用户
    }
    vol = collect_stage_volume("split", {}, arts)
    assert vol == {"train_rows": 10, "val_rows": 10, "test_rows": 6,
                   "train_users": 9, "val_users": 10, "test_users": 4}


def test_collect_feature_volume():
    arts = {
        "item_feat": _fake_inter(n=20, users=20, items=20),
        "user_ts_feat": _fake_inter(n=10, users=10, items=10),
        "train_feat": _fake_inter(n=100, users=8, items=18),
        "val_feat": _fake_inter(n=10, users=5, items=12),
        "test_feat": _fake_inter(n=10, users=6, items=15),
    }
    vol = collect_stage_volume("feature", {}, arts)
    assert vol["item_feat_rows"] == 20
    assert vol["user_ts_feat_rows"] == 10
    assert vol["train_feat_rows"] == 100
    assert vol["val_feat_rows"] == 10
    assert vol["test_feat_rows"] == 10


def test_collect_recall_volume():
    arts = {
        "recall_candidate_df": _fake_inter(n=300, users=10, items=30),
        "val_recall_candidate": _fake_inter(n=150, users=5, items=20),
        "test_recall_candidate": _fake_inter(n=150, users=6, items=25),
    }
    vol = collect_stage_volume("recall", {}, arts)
    assert vol == {"recall_candidate_rows": 300, "val_recall_candidate_rows": 150,
                   "test_recall_candidate_rows": 150, "recall_users": 10}


def test_collect_rank_volume_with_sample_stats():
    arts = {
        "sample_stats": {
            "train": {"pos": 1000, "hard_neg": 3000, "random_neg": 1000, "total": 5000},
            "val": {"pos": 200, "hard_neg": 600, "random_neg": 200, "total": 1000},
        },
        "test_result": pd.DataFrame({"user_id": [1, 2, 3], "score": [0.1, 0.2, 0.3]}),
        "imp": pd.DataFrame({"feature": ["a", "b", "c", "d"]}),
    }
    vol = collect_stage_volume("rank", {}, arts)
    assert vol["train"]["pos"] == 1000
    assert vol["train"]["total"] == 5000
    assert vol["val"]["hard_neg"] == 600
    assert vol["test_result_rows"] == 3
    assert vol["feature_cnt"] == 4


def test_collect_rerank_volume():
    arts = {
        "df_reordered": _fake_inter(n=50, users=10, items=30),
        "df_origin_top": _fake_inter(n=50, users=10, items=30),
    }
    vol = collect_stage_volume("rerank", {}, arts)
    assert vol == {"output_rows": 50, "output_users": 10, "origin_rows": 50}


def test_collect_unknown_stage_returns_empty():
    assert collect_stage_volume("eda", {}, {}) == {}


# ------------------------------------------------------------ finalize 集成
class _FakePaths:
    def __init__(self, logs):
        self.logs = logs


class _FakeCfg:
    def __init__(self, logs):
        self.paths = _FakePaths(logs)

    def to_dict(self):
        return {"rank": {"hard_neg_ratio": 5}}


def test_finalize_with_stage_volumes(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    cfg = _FakeCfg(tmp_path / "logs")
    mgr.finalize(
        exp_name="exp01",
        cfg=cfg,
        elapsed={"clean": 185.93, "split": 30.5},
        stage_metrics={},
        command="main.py --exp exp01",
        config_path=None,
        started_at="t0",
        stage_volumes={"clean": {"inter_rows": 198502, "users": 22363}},
    )
    report = json.loads(
        (tmp_path / "experiments" / "exp01" / "run_report.json").read_text(encoding="utf-8")
    )
    assert report["stages"]["clean"]["elapsed_sec"] == 185.93
    assert report["stages"]["clean"]["rows"] == {"inter_rows": 198502, "users": 22363}
    # 无 rows 的 stage 只含耗时，保持兼容
    assert report["stages"]["split"] == {"elapsed_sec": 30.5}


def test_finalize_without_volumes_backward_compatible(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    cfg = _FakeCfg(tmp_path / "logs")
    mgr.finalize(
        exp_name="exp02", cfg=cfg,
        elapsed={"clean": 10.0}, stage_metrics={},
        command="main.py", config_path=None, started_at="t0",
    )
    report = json.loads(
        (tmp_path / "experiments" / "exp02" / "run_report.json").read_text(encoding="utf-8")
    )
    assert report["stages"]["clean"] == {"elapsed_sec": 10.0}


def test_finalize_elapsed_skips_none_for_resume(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    cfg = _FakeCfg(tmp_path / "logs")
    mgr.finalize(
        exp_name="exp03", cfg=cfg,
        elapsed={"clean": None, "split": 30.5},  # clean 为 resume 加载，无本次耗时
        stage_metrics={}, command="main.py", config_path=None, started_at="t0",
        stage_volumes={"clean": {"inter_rows": 100}},
    )
    report = json.loads(
        (tmp_path / "experiments" / "exp03" / "run_report.json").read_text(encoding="utf-8")
    )
    assert report["stages"]["clean"]["elapsed_sec"] is None
    assert report["stages"]["clean"]["rows"]["inter_rows"] == 100
    assert report["elapsed_sec"] == 30.5  # None 不计入总时长
