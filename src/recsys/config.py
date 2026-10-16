"""全局配置加载模块。

将 config/config.yaml 解析为类型化配置对象（dataclass），
所有业务代码只通过 ``cfg`` 读取参数，不再出现硬编码路径/魔法数字。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
# 类型化配置对象
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    inter_path: Path
    meta_path: Path
    outputs: Path
    checkpoints: Path
    logs: Path
    model_dir: Path
    results: Path
    eda_plots: Path
    eda_tables: Path


@dataclass
class SparkConfig:
    app_name: str
    master: str
    driver_memory: str
    shuffle_partitions: int
    adaptive_enabled: bool
    hadoop_local_fs_fix: bool
    # TaskResultLost 加固：放宽结果收集上限 + 提高执行内存占比
    driver_max_result_size: str = "4g"
    memory_fraction: float = 0.8
    storage_fraction: float = 0.3


@dataclass
class DataCleanConfig:
    dedup_user_item: bool
    keep_missing_meta: bool


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class RecallConfig:
    itemcf_top_k: int
    hot_top_k: int
    final_recall_k: int
    min_cooc: int
    eval_k_list: List[int]


@dataclass
class RankConfig:
    hard_neg_ratio: int
    random_neg_ratio: int
    pos_rating_th: float
    eval_topk: List[int]
    num_boost_round: int
    early_stopping_rounds: int
    lgb_params: Dict[str, Any]


@dataclass
class RerankConfig:
    top_n_output: int
    hot_penalty_coeff: float
    max_category_ratio: float
    eval_topk: List[int]


@dataclass
class PipelineConfig:
    seed: int
    steps: List[str]
    checkpoint: bool
    resume: bool


@dataclass
class Config:
    paths: PathsConfig
    spark: SparkConfig
    data_clean: DataCleanConfig
    split: SplitConfig
    recall: RecallConfig
    rank: RankConfig
    rerank: RerankConfig
    pipeline: PipelineConfig

    def validate(self) -> None:
        """启动前校验配置合法性，尽早失败。"""
        s = self.split
        if abs(s.train_ratio + s.val_ratio + s.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"split 比例之和必须=1，当前为 {s.train_ratio + s.val_ratio + s.test_ratio}"
            )
        if not 0 < self.rerank.max_category_ratio <= 1:
            raise ValueError("rerank.max_category_ratio 必须在 (0, 1]")
        if self.rerank.top_n_output <= 0:
            raise ValueError("rerank.top_n_output 必须为正整数")

    def to_dict(self) -> dict:
        """把类型化配置转回普通 dict（Path 转 str），用于实验配置快照落盘。"""
        return _to_plain(dataclasses.asdict(self))


def _to_plain(obj: Any) -> Any:
    """递归把 dataclass 中的 Path / 容器转成可 JSON/YAML 序列化的普通对象。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# 加载入口
# ---------------------------------------------------------------------------
def _load_mapping(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return raw


def deep_merge(base: dict, overlay: dict) -> dict:
    """递归合并两个配置 dict。

    规则：
    - overlay 中的标量 / 列表直接整体覆盖 base 的同名键；
    - 两边都是 dict 时递归合并；
    - overlay 独有的键直接新增。
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_config(config_path: Path, raw: dict) -> Config:
    """从原始 yaml dict 构建类型化 Config（路径解析基于 config 文件定位项目根）。"""
    root = config_path.parent.parent  # 项目根目录

    paths_raw = raw["paths"]
    paths = PathsConfig(
        inter_path=Path(paths_raw["inter_path"]),
        meta_path=Path(paths_raw["meta_path"]),
        outputs=_resolve(root, paths_raw["outputs"]),
        checkpoints=_resolve(root, paths_raw["checkpoints"]),
        logs=_resolve(root, paths_raw["logs"]),
        model_dir=_resolve(root, paths_raw["model_dir"]),
        results=_resolve(root, paths_raw["results"]),
        eda_plots=_resolve(root, paths_raw["eda_plots"]),
        eda_tables=_resolve(root, paths_raw["eda_tables"]),
    )
    # 原始数据路径保持原样（用户本机路径，不做相对解析）

    spark_raw = raw["spark"]
    spark = SparkConfig(**spark_raw)

    clean_raw = raw["data_clean"]
    data_clean = DataCleanConfig(**clean_raw)

    split_raw = raw["split"]
    split = SplitConfig(**split_raw)

    recall_raw = raw["recall"]
    recall = RecallConfig(**recall_raw)

    rank_raw = raw["rank"]
    rank = RankConfig(
        hard_neg_ratio=rank_raw["hard_neg_ratio"],
        random_neg_ratio=rank_raw["random_neg_ratio"],
        pos_rating_th=rank_raw["pos_rating_th"],
        eval_topk=rank_raw["eval_topk"],
        num_boost_round=rank_raw["num_boost_round"],
        early_stopping_rounds=rank_raw["early_stopping_rounds"],
        lgb_params=rank_raw["lgb"],
    )

    rerank_raw = raw["rerank"]
    rerank = RerankConfig(**rerank_raw)

    pipeline_raw = raw["pipeline"]
    pipeline = PipelineConfig(**pipeline_raw)

    cfg = Config(
        paths=paths,
        spark=spark,
        data_clean=data_clean,
        split=split,
        recall=recall,
        rank=rank,
        rerank=rerank,
        pipeline=pipeline,
    )
    cfg.validate()
    return cfg


def load_config(config_path: str | Path) -> Config:
    """读取基线 yaml 配置并解析为 :class:`Config`。

    相对路径基于「项目根目录」（config 文件的上一级）解析，
    因此无论从哪个工作目录启动 ``python main.py``，路径都稳定。
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    raw = _load_mapping(config_path)
    return _build_config(config_path, raw)


def load_config_with_overlay(config_path: str | Path,
                             overlay_path: str | Path) -> Config:
    """读取基线配置并叠加实验配置（deep merge）后解析为 :class:`Config`。

    实验配置只需写要覆盖的项（如 ``rank: {hard_neg_ratio: 5}``），
    其余全部继承基线；合并结果与基线一样做启动校验。
    """
    config_path = Path(config_path).resolve()
    overlay_path = Path(overlay_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not overlay_path.exists():
        raise FileNotFoundError(f"实验配置不存在: {overlay_path}")
    base = _load_mapping(config_path)
    overlay = _load_mapping(overlay_path)
    return _build_config(config_path, deep_merge(base, overlay))


def _resolve(root: Path, p: str | Path) -> Path:
    """相对路径挂到项目根，绝对路径原样保留。"""
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def ensure_output_dirs(cfg: Config) -> None:
    """预先创建所有输出目录，避免运行时报 FileNotFoundError。"""
    for d in (
        cfg.paths.outputs,
        cfg.paths.checkpoints,
        cfg.paths.logs,
        cfg.paths.model_dir,
        cfg.paths.results,
        cfg.paths.eda_plots,
        cfg.paths.eda_tables,
    ):
        d.mkdir(parents=True, exist_ok=True)
