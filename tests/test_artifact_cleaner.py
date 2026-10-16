"""第 5 步产物治理测试：inspect 盘点 / clean 清理的纯逻辑验证。

全部使用 tmp_path 构造的临时目录（不碰真实 outputs/，不起 Spark）。
覆盖：inspect 状态判断、dry-run 不删、--yes 删除、final 产物保护、
未知目标报错、all 混用报错、实验中间产物清理且留痕保留。
"""

from __future__ import annotations

import json

import pytest

from src.recsys.artifact_cleaner import (
    clean_checkpoints,
    clean_experiments,
    inspect_outputs,
    parse_clean_targets,
    stage_complete,
)
from src.recsys.config import PathsConfig


@pytest.fixture()
def fake_project(tmp_path):
    """构造一个带 outputs/ + experiments/ 的假项目树。

    - clean  阶段：manifest 声明的 2 个产物都在 → 完整
    - split  阶段：manifest 声明 2 个产物但只落 1 个 → 不完整
    - feature 阶段：目录不存在 → 无产物
    - results / model / logs：各 1 个文件（最终产物，应受保护）
    - experiments/exp01：中间产物 + 留痕文件齐全
    """
    root = tmp_path
    outputs = root / "outputs"
    ckpt = outputs / "checkpoints"
    results = outputs / "results"
    model = outputs / "model"
    logs = outputs / "logs"
    eda_plots = outputs / "eda" / "plots"
    eda_tables = outputs / "eda" / "tables"
    for d in (ckpt, results, model, logs, eda_plots, eda_tables):
        d.mkdir(parents=True, exist_ok=True)

    clean = ckpt / "clean"
    clean.mkdir()
    (clean / "manifest.json").write_text(
        json.dumps({"inter": "spark_df", "stats": "json"}), encoding="utf-8"
    )
    (clean / "inter.parquet").mkdir()
    (clean / "inter.parquet" / "part-00000.parquet").write_bytes(b"x" * 1024)
    (clean / "stats.json").write_text("{}", encoding="utf-8")

    split = ckpt / "split"
    split.mkdir()
    (split / "manifest.json").write_text(
        json.dumps({"train": "spark_df", "val": "spark_df"}), encoding="utf-8"
    )
    (split / "train.parquet").mkdir()  # val.parquet 缺失 → 不完整

    (results / "reorder_final_result.csv").write_bytes(b"a,b\n1,2\n")
    (model / "lgb_rank_model.txt").write_text("booster", encoding="utf-8")
    (logs / "recsys.log").write_text("log line", encoding="utf-8")

    exp = root / "experiments" / "exp01"
    (exp / "checkpoints" / "x").mkdir(parents=True)
    (exp / "model").mkdir()
    (exp / "results").mkdir()
    (exp / "eda" / "plots").mkdir(parents=True)
    (exp / "config_snapshot.yaml").write_text("cfg: 1", encoding="utf-8")
    (exp / "meta.json").write_text("{}", encoding="utf-8")
    (exp / "run_report.json").write_text("{}", encoding="utf-8")
    (root / "experiments" / "exp01.yaml").write_text(
        "rank:\n  hard_neg_ratio: 5\n", encoding="utf-8"
    )
    (root / "experiments" / "overview.csv").write_text(
        "exp_id,created_at\nexp01,2026-10-14\n", encoding="utf-8"
    )

    paths = PathsConfig(
        inter_path=root / "raw.json",
        meta_path=root / "raw_meta.json",
        outputs=outputs,
        checkpoints=ckpt,
        logs=logs,
        model_dir=model,
        results=results,
        eda_plots=eda_plots,
        eda_tables=eda_tables,
    )
    return root, paths


# ---------------------------------------------------------------------------
# 盘点（inspect）
# ---------------------------------------------------------------------------
class TestInspect:
    def test_stage_status_detected(self, fake_project):
        """clean 完整 / split 不完整 / feature 不存在。"""
        _, paths = fake_project
        r = inspect_outputs(paths)
        st = r["checkpoint_stages"]
        assert st["clean"]["complete"] is True
        assert st["split"]["complete"] is False
        assert st["feature"]["exists"] is False

    def test_manifest_missing_artifact_is_incomplete(self, fake_project):
        """manifest 声明了但文件缺失 → stage_complete=False。"""
        _, paths = fake_project
        assert stage_complete(paths.checkpoints / "clean") is True
        assert stage_complete(paths.checkpoints / "split") is False

    def test_subdir_bytes_and_files(self, fake_project):
        """results / model / logs 字节数与文件清单正确。"""
        _, paths = fake_project
        r = inspect_outputs(paths)
        assert r["subdirs_bytes"]["results"] > 0
        assert r["subdirs_bytes"]["model"] > 0
        assert r["subdirs_bytes"]["logs"] > 0
        assert len(r["results_files"]) == 1
        assert r["results_files"][0]["name"] == "reorder_final_result.csv"

    def test_experiment_overview_count(self, fake_project):
        """overview.csv 的实验数 = 行数 - 1。"""
        _, paths = fake_project
        r = inspect_outputs(paths)
        assert r["experiments"]["count"] == 1
        assert len(r["experiments"]["dirs"]) == 1


# ---------------------------------------------------------------------------
# 清理（clean checkpoints）
# ---------------------------------------------------------------------------
class TestCleanCheckpoints:
    def test_dry_run_does_not_delete(self, fake_project):
        """默认 dry-run：打印清单，不删除任何文件。"""
        _, paths = fake_project
        r = clean_checkpoints(["clean"], paths.checkpoints, yes=False)
        assert r["dry_run"] is True
        assert r["deleted"] == []
        assert (paths.checkpoints / "clean").exists()

    def test_yes_deletes_target_only(self, fake_project):
        """--yes 删除指定阶段，其他阶段不受影响。"""
        _, paths = fake_project
        r = clean_checkpoints(["clean"], paths.checkpoints, yes=True)
        assert r["dry_run"] is False
        assert r["deleted"] == ["clean"]
        assert not (paths.checkpoints / "clean").exists()
        assert (paths.checkpoints / "split").exists()

    def test_all_deletes_all_and_protects_final(self, fake_project):
        """--clean all 清空 checkpoint，results/model/logs 永不自动删。"""
        _, paths = fake_project
        r = clean_checkpoints(["all"], paths.checkpoints, yes=True)
        assert not (paths.checkpoints / "clean").exists()
        assert not (paths.checkpoints / "split").exists()
        assert (paths.results / "reorder_final_result.csv").exists()
        assert (paths.model_dir / "lgb_rank_model.txt").exists()
        assert (paths.logs / "recsys.log").exists()
        assert len(r["deleted"]) == 2

    def test_clean_nonexistent_stage_is_noop(self, fake_project):
        """目标阶段不存在时 dry-run 不报错、不删除。"""
        _, paths = fake_project
        r = clean_checkpoints(["rerank"], paths.checkpoints, yes=True)
        assert r["dry_run"] is False
        assert r["deleted"] == []


# ---------------------------------------------------------------------------
# 清理（参数解析）
# ---------------------------------------------------------------------------
class TestParseCleanTargets:
    def test_parse_stage_list(self):
        assert parse_clean_targets("split,feature") == ["split", "feature"]

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="未知清理目标"):
            parse_clean_targets("nope")

    def test_all_mixed_with_stage_raises(self):
        with pytest.raises(ValueError, match="不能与其他目标混用"):
            parse_clean_targets("all,split")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            parse_clean_targets(" , ")


# ---------------------------------------------------------------------------
# 清理（实验中间产物）
# ---------------------------------------------------------------------------
class TestCleanExperiments:
    def test_clean_experiments_keeps_records(self, fake_project):
        """清理实验 checkpoints/model/results/eda，留痕文件全部保留。"""
        root, _ = fake_project
        r = clean_experiments(root / "experiments", yes=True)
        assert r["dry_run"] is False
        exp = root / "experiments" / "exp01"
        assert not (exp / "checkpoints").exists()
        assert not (exp / "model").exists()
        assert not (exp / "results").exists()
        assert not (exp / "eda").exists()
        # 留痕保留（可复现信息不删）
        assert (exp / "config_snapshot.yaml").exists()
        assert (exp / "meta.json").exists()
        assert (exp / "run_report.json").exists()
        assert (root / "experiments" / "overview.csv").exists()
        assert (root / "experiments" / "exp01.yaml").exists()

    def test_clean_experiments_dry_run(self, fake_project):
        """实验清理同样遵守 dry-run。"""
        root, _ = fake_project
        r = clean_experiments(root / "experiments", yes=False)
        assert r["dry_run"] is True
        exp = root / "experiments" / "exp01"
        assert (exp / "checkpoints").exists()
