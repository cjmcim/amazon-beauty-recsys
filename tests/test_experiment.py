"""实验管理单元测试（纯逻辑，无需 Spark）。

覆盖：
1. deep_merge：标量覆盖 / 嵌套 dict 递归 / 列表整体替换 / 新增键 / 无副作用；
2. load_config_with_overlay：覆盖生效、未覆盖项继承、校验仍生效；
3. Config.to_dict：路径转 str、可 YAML 序列化；
4. ExperimentManager：目录创建、配置快照 round-trip、overview 追加与列对齐；
5. get_git_commit：始终返回非空字符串。
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.recsys.config import deep_merge, load_config, load_config_with_overlay
from src.recsys.experiment import OVERVIEW_COLUMNS, ExperimentManager, get_git_commit

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = str(REPO_ROOT / "config" / "config.yaml")


# ------------------------------------------------------------------ deep_merge
def test_deep_merge_scalar_overwrite():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    overlay = {"a": 2}
    assert deep_merge(base, overlay) == {"a": 2, "b": {"x": 1, "y": 2}}


def test_deep_merge_nested_dict_recursive():
    base = {"rank": {"hard_neg_ratio": 3, "lgb": {"learning_rate": 0.05, "num_leaves": 31}}}
    overlay = {"rank": {"hard_neg_ratio": 5, "lgb": {"learning_rate": 0.1}}}
    out = deep_merge(base, overlay)
    # 覆盖项更新、未覆盖的同级保留
    assert out["rank"] == {"hard_neg_ratio": 5, "lgb": {"learning_rate": 0.1, "num_leaves": 31}}


def test_deep_merge_list_whole_replace():
    base = {"recall": {"eval_k_list": [5, 10, 20]}}
    overlay = {"recall": {"eval_k_list": [1, 2]}}
    assert deep_merge(base, overlay)["recall"]["eval_k_list"] == [1, 2]


def test_deep_merge_new_key_no_side_effect():
    base = {"a": 1}
    overlay = {"new_key": {"nested": True}}
    out = deep_merge(base, overlay)
    assert out["new_key"] == {"nested": True}
    assert "new_key" not in base  # 不修改入参


# ---------------------------------------------------------- overlay 配置加载
def test_load_config_with_overlay(tmp_path):
    cfg = load_config(BASE_CONFIG)
    overlay = tmp_path / "exp.yaml"
    overlay.write_text("rank:\n  hard_neg_ratio: 9\n", encoding="utf-8")

    cfg2 = load_config_with_overlay(BASE_CONFIG, str(overlay))
    assert cfg2.rank.hard_neg_ratio == 9          # 覆盖生效
    assert cfg2.rank.random_neg_ratio == cfg.rank.random_neg_ratio  # 未覆盖项继承


def test_load_config_with_overlay_validation_still_applies(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("split:\n  train_ratio: 0.5\n  val_ratio: 0.2\n  test_ratio: 0.2\n",
                   encoding="utf-8")
    with pytest.raises(ValueError):
        load_config_with_overlay(BASE_CONFIG, str(bad))


def test_load_config_with_overlay_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config_with_overlay(BASE_CONFIG, str(tmp_path / "nope.yaml"))


# ------------------------------------------------------------- Config.to_dict
def test_config_to_dict_paths_to_str():
    cfg = load_config(BASE_CONFIG)
    d = cfg.to_dict()
    assert isinstance(d["paths"]["outputs"], str)
    assert d["rank"]["lgb_params"]["objective"] == "binary"
    # round trip：to_dict 结果必须可被 YAML 序列化（快照落盘用）
    dumped = yaml.safe_dump(d, allow_unicode=True)
    assert "objective: binary" in dumped


# ------------------------------------------------------------ ExperimentManager
def test_ensure_exp_dir(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    d = mgr.ensure_exp_dir("exp01")
    assert d.exists() and d.is_dir()


def test_save_config_snapshot_roundtrip(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    cfg = load_config(BASE_CONFIG)
    path = mgr.save_config_snapshot("exp01", cfg.to_dict())
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["rank"]["lgb_params"]["objective"] == "binary"
    assert isinstance(loaded["paths"]["outputs"], str)


def test_overview_append_and_column_alignment(tmp_path):
    mgr = ExperimentManager(tmp_path / "experiments")
    row1 = {"exp_id": "exp01", "created_at": "t1", "git_commit": "abc",
            "status": "done", "rank_auc": 0.6, "elapsed_sec": 10.0}
    row2 = {"exp_id": "exp02", "created_at": "t2", "git_commit": "def",
            "status": "done", "rank_auc": 0.7, "rank_Recall@5": 0.1,
            "rerank_diversity": 13.4, "elapsed_sec": 20.0}
    mgr.append_overview(row1)
    mgr.append_overview(row2)

    df = pd.read_csv(mgr.overview_file())
    assert len(df) == 2
    # 列严格对齐 OVERVIEW_COLUMNS（顺序即展示顺序）
    assert list(df.columns) == OVERVIEW_COLUMNS
    # 缺失列填空（exp01 没跑 rerank）
    assert pd.isna(df.loc[0, "rank_Recall@5"])
    assert df.loc[1, "rank_Recall@5"] == 0.1
    assert df.loc[1, "rerank_diversity"] == 13.4


def test_get_git_commit_returns_str():
    commit = get_git_commit(REPO_ROOT)
    assert isinstance(commit, str) and len(commit) > 0
