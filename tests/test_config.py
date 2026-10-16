"""配置加载与校验测试（无需 Spark）。"""

from pathlib import Path

import pytest

from src.recsys.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    assert abs(cfg.split.train_ratio + cfg.split.val_ratio + cfg.split.test_ratio - 1) < 1e-6
    assert cfg.rank.lgb_params["objective"] == "binary"
    assert cfg.recall.final_recall_k >= cfg.recall.hot_top_k
    # 相对路径已解析为绝对路径
    assert cfg.paths.outputs.is_absolute()
    assert cfg.paths.results.is_absolute()


def test_validate_ratio_failure():
    import dataclasses

    base = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    split = dataclasses.replace(base.split, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
    bad = dataclasses.replace(base, split=split)
    with pytest.raises(ValueError, match="比例之和必须=1"):
        bad.validate()


def test_validate_rerank_ratio():
    import dataclasses

    base = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    rerank = dataclasses.replace(base.rerank, max_category_ratio=1.5)
    bad = dataclasses.replace(base, rerank=rerank)
    with pytest.raises(ValueError):
        bad.validate()
