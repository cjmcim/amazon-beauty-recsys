"""流水线编排器 —— 工业化的核心骨架。

职责：
1. 定义 stage 依赖顺序（eda → clean → split → feature → recall → rank → rerank）；
2. 每个 stage 是一个纯函数：输入上游产物（dict），输出下游产物（dict）；
3. checkpoint 落盘：Spark DF → parquet / pandas → pickle / json / LightGBM → txt /
   Spark ML 模型 → 原生 save，支持断点续跑（--resume）；
4. 运行结束后统一释放缓存与 SparkSession。
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import StringIndexerModel

from src.recsys.io_utils import load_lgb_model, save_lgb_model
from src.recsys.config import Config
from src.recsys.data.clean import run_clean
from src.recsys.data.split import run_split
from src.recsys.eda.runner import run_eda
from src.recsys.features.feature_pipeline import run_feature
from src.recsys.rank.pipeline import run_rank
from src.recsys.recall.engine import run_recall
from src.recsys.rerank.reranker import run_rerank
from src.recsys.spark_utils import get_spark, stop_spark

logger = logging.getLogger("recsys.pipeline.runner")

STAGE_ORDER = ["eda", "clean", "split", "feature", "recall", "rank", "rerank"]

# ---------------------------------------------------------------------------
# Checkpoint：类型化落盘 / 读取
# ---------------------------------------------------------------------------
class CheckpointManager:
    """把 stage 产物按类型落盘到 outputs/checkpoints/<stage>/，可断点续跑。"""

    def __init__(self, root: Path, spark: SparkSession):
        self.root = Path(root)
        self.spark = spark

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def _kind(self, obj: Any) -> str:
        if isinstance(obj, DataFrame):
            return "spark_df"
        if isinstance(obj, pd.DataFrame):
            return "pandas"
        if isinstance(obj, lgb.Booster):
            return "booster"
        if isinstance(obj, StringIndexerModel):
            return "spark_model"
        if isinstance(obj, (dict, list, float, int, str, bool, type(None))):
            return "json"
        raise TypeError(f"暂不支持 checkpoint 的类型: {type(obj)}")

    def is_complete(self, stage: str) -> bool:
        """stage 全部产物已存在且可读。"""
        manifest = self.stage_dir(stage) / "manifest.json"
        if not manifest.exists():
            return False
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return False
        for name, kind in info.items():
            if not self._file(stage, name, kind).exists():
                return False
        return True

    def _file(self, stage: str, name: str, kind: str) -> Path:
        d = self.stage_dir(stage)
        if kind == "spark_df":
            return d / f"{name}.parquet"
        if kind == "pandas":
            return d / f"{name}.pkl"
        if kind == "booster":
            return d / f"{name}.txt"
        if kind == "spark_model":
            return d / f"{name}_model"
        return d / f"{name}.json"  # json 兜底

    def save(self, stage: str, artifacts: Dict[str, Any]) -> None:
        d = self.stage_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, str] = {}
        for name, obj in artifacts.items():
            kind = self._kind(obj)
            path = self._file(stage, name, kind)
            if kind == "spark_df":
                obj.write.mode("overwrite").parquet(str(path))
            elif kind == "pandas":
                obj.to_pickle(str(path))
            elif kind == "booster":
                save_lgb_model(obj, path)
            elif kind == "spark_model":
                obj.write().overwrite().save(str(path))
            else:
                path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            manifest[name] = kind
        (d / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("✔ checkpoint 已保存: %s", d)

    def load(self, stage: str) -> Dict[str, Any]:
        d = self.stage_dir(stage)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        out: Dict[str, Any] = {}
        for name, kind in manifest.items():
            path = self._file(stage, name, kind)
            if kind == "spark_df":
                out[name] = self.spark.read.parquet(str(path))
            elif kind == "pandas":
                out[name] = pd.read_pickle(str(path))
            elif kind == "booster":
                out[name] = load_lgb_model(path)
            elif kind == "spark_model":
                out[name] = StringIndexerModel.load(str(path))
            else:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("↩ checkpoint 已加载: %s", d)
        return out


# ---------------------------------------------------------------------------
# Stage 注册表：stage 名 → (依赖产物名列表, 执行函数)
# ---------------------------------------------------------------------------
def _stage_eda(cfg, ctx):
    return run_eda(get_spark(cfg.spark), cfg)


def _stage_clean(cfg, ctx):
    return run_clean(get_spark(cfg.spark), cfg)


def _stage_split(cfg, ctx):
    return run_split(ctx["inter"], cfg)


def _stage_feature(cfg, ctx):
    return run_feature(ctx["train"], ctx["val"], ctx["test"], ctx["meta"])


def _stage_recall(cfg, ctx):
    return run_recall(ctx["train"], ctx["val"], ctx["test"], cfg.recall)


def _stage_rank(cfg, ctx):
    return run_rank(
        cfg,
        train_inter=ctx["train_inter"], val_inter=ctx["val_inter"],
        test_inter=ctx["test_inter"],
        item_feat=ctx["item_feat"], user_ts_feat=ctx["user_ts_feat"],
        recall_candidate_df=ctx["recall_candidate_df"],
        val_recall_candidate=ctx["val_recall_candidate"],
        test_recall_candidate=ctx["test_recall_candidate"],
        brand_model=ctx["brand_model"], cate_model=ctx["cate_model"],
    )


def _stage_rerank(cfg, ctx):
    return run_rerank(ctx["test_result"], ctx["item_feat"], cfg.rerank,
                      cfg.paths.results)


STAGE_FUNCS = {
    "eda": _stage_eda,
    "clean": _stage_clean,
    "split": _stage_split,
    "feature": _stage_feature,
    "recall": _stage_recall,
    "rank": _stage_rank,
    "rerank": _stage_rerank,
}

# 每个 stage 需要上游提供的产物（用于依赖校验）
STAGE_DEPS = {
    "eda": [],
    "clean": [],
    "split": ["inter"],
    "feature": ["train", "val", "test", "meta"],
    "recall": ["train", "val", "test"],
    "rank": ["train_inter", "val_inter", "test_inter", "item_feat", "user_ts_feat",
             "recall_candidate_df", "val_recall_candidate", "test_recall_candidate",
             "brand_model", "cate_model"],
    "rerank": ["test_result", "item_feat"],
}


# ---------------------------------------------------------------------------
# Stage 数据量采集：每个 stage 结束后统计关键产物规模，写入运行报告 rows
# ---------------------------------------------------------------------------
def _df_count(obj):
    """统一取产物行数：Spark DataFrame → count()；pandas DataFrame → len；其他 → None。"""
    if isinstance(obj, DataFrame):
        return obj.count()
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    return None


def _distinct_count(obj, col: str):
    """统一取某列去重数（用户数/商品数）。"""
    if isinstance(obj, DataFrame):
        return obj.select(col).distinct().count()
    if isinstance(obj, pd.DataFrame):
        return int(obj[col].nunique()) if col in obj.columns else None
    return None


# stage 名 → (ctx, artifacts) -> dict（数据量指标名 → 数值）
STAGE_VOLUME_COLLECTORS = {
    "clean": lambda ctx, a: {
        "inter_rows": _df_count(a.get("inter")),
        "meta_rows": _df_count(a.get("meta")),
        "users": _distinct_count(a.get("inter"), "user_id"),
        "items": _distinct_count(a.get("inter"), "item_id"),
    },
    "split": lambda ctx, a: {
        "train_rows": _df_count(a.get("train")),
        "val_rows": _df_count(a.get("val")),
        "test_rows": _df_count(a.get("test")),
        "train_users": _distinct_count(a.get("train"), "user_id"),
        "val_users": _distinct_count(a.get("val"), "user_id"),
        "test_users": _distinct_count(a.get("test"), "user_id"),
    },
    "feature": lambda ctx, a: {
        "item_feat_rows": _df_count(a.get("item_feat")),
        "user_ts_feat_rows": _df_count(a.get("user_ts_feat")),
        "train_feat_rows": _df_count(a.get("train_feat")),
        "val_feat_rows": _df_count(a.get("val_feat")),
        "test_feat_rows": _df_count(a.get("test_feat")),
    },
    "recall": lambda ctx, a: {
        "recall_candidate_rows": _df_count(a.get("recall_candidate_df")),
        "val_recall_candidate_rows": _df_count(a.get("val_recall_candidate")),
        "test_recall_candidate_rows": _df_count(a.get("test_recall_candidate")),
        "recall_users": _distinct_count(a.get("recall_candidate_df"), "user_id"),
    },
    "rank": lambda ctx, a: {
        **a.get("sample_stats", {}),
        "test_result_rows": _df_count(a.get("test_result")),
        "feature_cnt": _df_count(a.get("imp")),
    },
    "rerank": lambda ctx, a: {
        "output_rows": _df_count(a.get("df_reordered")),
        "output_users": _distinct_count(a.get("df_reordered"), "user_id"),
        "origin_rows": _df_count(a.get("df_origin_top")),
    },
}


def collect_stage_volume(stage: str, ctx: Dict[str, Any],
                         artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """采集一个 stage 的关键数据量（产物行数 / 用户数等）；无采集器时返回空 dict。"""
    c = STAGE_VOLUME_COLLECTORS.get(stage)
    if c is None:
        return {}
    return c(ctx, artifacts)


def _pick_serializable(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """从 stage 产物中挑出可 JSON 序列化的指标（跳过 DataFrame/模型等）。"""
    out: Dict[str, Any] = {}
    for k, v in artifacts.items():
        if isinstance(v, (dict, list, float, int, str, bool)) or v is None:
            out[k] = v
    return out


def run_stages(cfg: Config, steps: List[str], exp_mgr=None,
               exp_name: str | None = None,
               config_path: str | Path | None = None,
               command: str | None = None) -> Dict[str, Any]:
    """执行流水线。

    Args:
        cfg: 全局配置
        steps: 有序 stage 名列表（STAGE_ORDER 的子集，按依赖顺序）
        exp_mgr: 实验管理器（None 表示普通模式，不做归档）
        exp_name: 实验名（实验模式归档用）
        config_path: 基线配置路径（meta 里记录 commit/hash 用）
        command: 完整命令行（meta 里记录用）
    """
    spark = get_spark(cfg.spark)
    ckpt = CheckpointManager(cfg.paths.checkpoints, spark)
    ctx: Dict[str, Any] = {}
    started_at = datetime.now().isoformat(timespec="seconds")
    stage_elapsed: Dict[str, float] = {}
    stage_metrics: Dict[str, Any] = {}
    stage_volumes: Dict[str, Dict] = {}

    # ---- 校验步骤合法性 ----
    # 注意：依赖检查不能在这里做（ctx 此时为空），必须放到逐 stage 执行前，
    # 那时前序 stage 的产物（或 resume 加载的 checkpoint）已经就绪。
    for s in steps:
        if s not in STAGE_ORDER:
            raise ValueError(f"未知 stage: {s}，可选: {STAGE_ORDER}")

    # ---- 执行 ----
    # resume 模式下：先按依赖顺序预加载所有已完成的 checkpoint。
    # 这样「全流程续跑」能跳过已算好的 stage，「只跑下游 stage」也能自动复用上游产物。
    loaded_stages: set[str] = set()
    if cfg.pipeline.resume:
        for s in STAGE_ORDER:
            if ckpt.is_complete(s):
                artifacts = ckpt.load(s)
                ctx.update(artifacts)
                loaded_stages.add(s)
                # resume 加载的 stage 也统计数据量（耗时记为 None：非本次计算）
                stage_elapsed[s] = None
                vol = collect_stage_volume(s, ctx, artifacts)
                if vol:
                    stage_volumes[s] = vol
                    logger.info("📊 stage '%s' 数据量(自checkpoint): %s", s, vol)
                logger.info("✔ stage '%s' 使用已落盘 checkpoint（跳过重算）", s)

    for stage in steps:
        if stage in loaded_stages:
            continue
        # 依赖检查：前序 stage 已执行（或 resume 已从 checkpoint 加载），
        # ctx 此时应包含全部上游产物
        missing = [d for d in STAGE_DEPS[stage] if d not in ctx]
        if missing:
            raise RuntimeError(
                f"stage '{stage}' 缺少上游产物 {missing}。"
                f"请从更早的 stage 开始运行，或使用 --resume 加载已落盘的 checkpoint。"
            )
        logger.info("开始执行 stage: %s", stage)
        t0 = time.perf_counter()
        artifacts = STAGE_FUNCS[stage](cfg, ctx)
        stage_elapsed[stage] = round(time.perf_counter() - t0, 2)
        vol = collect_stage_volume(stage, ctx, artifacts)
        if vol:
            stage_volumes[stage] = vol
            logger.info("📊 stage '%s' 数据量: %s", stage, vol)
        if exp_mgr is not None:
            sm = _pick_serializable(artifacts)
            if sm:
                stage_metrics[stage] = sm
        if cfg.pipeline.checkpoint:
            ckpt.save(stage, artifacts)
        ctx.update(artifacts)

    # ---- 实验归档（在释放 Spark 缓存之前：数据量统计依赖缓存命中） ----
    if exp_mgr is not None:
        exp_mgr.finalize(
            exp_name=exp_name,
            cfg=cfg,
            elapsed=stage_elapsed,
            stage_metrics=stage_metrics,
            command=command or " ".join(sys.argv),
            config_path=Path(config_path) if config_path else None,
            started_at=started_at,
            stage_volumes=stage_volumes,
        )

    # ---- 资源清理（legacy 结尾的释放逻辑） ----
    logger.info("全链路结束，释放 Spark 缓存与内存…")
    for name, obj in ctx.items():
        if isinstance(obj, DataFrame):
            try:
                obj.unpersist()
            except Exception:
                pass
    gc.collect()
    stop_spark()
    return ctx


def main(config_path: str, steps: List[str], resume: bool,
         checkpoint: bool | None, exp_name: str | None = None) -> None:
    """CLI 入口逻辑（供 main.py 调用）。

    Args:
        exp_name: 实验名。非 None 时进入实验模式：
            基线 config + experiments/<exp_name>.yaml 覆盖合并；
            产物路径（checkpoint/model/results/eda）切换到实验目录，实现实验隔离。
    """
    from src.recsys.config import ensure_output_dirs, load_config, load_config_with_overlay
    from src.recsys.logging_utils import setup_logging

    cfg = load_config(config_path)
    # CLI 参数覆盖配置（resume / checkpoint）
    if resume:
        cfg.pipeline.resume = True
    if checkpoint is not None:
        cfg.pipeline.checkpoint = checkpoint
    if steps == ["all"] or steps is None:
        steps = cfg.pipeline.steps

    # ---- 实验模式：加载覆盖配置 + 切换产物路径 ----
    exp_mgr = None
    if exp_name:
        from src.recsys.experiment import ExperimentManager

        project_root = Path(config_path).resolve().parent.parent
        exp_root = project_root / "experiments"
        overlay = exp_root / f"{exp_name}.yaml"
        if not overlay.exists():
            raise SystemExit(
                f"实验配置不存在: {overlay}。请先创建 experiments/{exp_name}.yaml"
            )
        cfg = load_config_with_overlay(config_path, overlay)
        exp_mgr = ExperimentManager(exp_root)
        exp_mgr.ensure_exp_dir(exp_name)
        d = exp_mgr.exp_dir(exp_name)
        # checkpoint 隔离：实验各自独立中间产物，互不污染（可复现的根基）
        cfg.paths.checkpoints = d / "checkpoints"
        cfg.paths.model_dir = d / "model"
        cfg.paths.results = d / "results"
        cfg.paths.eda_plots = d / "eda" / "plots"
        cfg.paths.eda_tables = d / "eda" / "tables"
        logger = setup_logging(cfg.paths.logs)  # 日志仍写全局 outputs/logs
        logger.info("实验模式: %s → 产物写入 %s（checkpoint 独立隔离）", exp_name, d)

    ensure_output_dirs(cfg)
    logger = setup_logging(cfg.paths.logs)
    logger.info("配置加载完成: %s", config_path)
    if exp_mgr is not None:
        logger.info("实验覆盖配置: %s", exp_root / f"{exp_name}.yaml")
    logger.info("执行 stages: %s | checkpoint=%s resume=%s",
                steps, cfg.pipeline.checkpoint, cfg.pipeline.resume)

    ctx = run_stages(cfg, steps, exp_mgr=exp_mgr, exp_name=exp_name,
                     config_path=config_path)
    if exp_mgr is not None:
        logger.info("实验归档完成: %s", exp_mgr.exp_dir(exp_name))

    # 摘要输出
    if "metrics" in ctx and ctx.get("metrics"):
        logger.info("精排指标: %s", ctx["metrics"])
    if "auc" in ctx and ctx.get("auc") is not None:
        logger.info("精排 Candidate AUC: %.4f", ctx["auc"])
    if "df_reordered" in ctx and ctx.get("df_reordered") is not None:
        logger.info("重排结果已写入: %s/reorder_final_result.csv", cfg.paths.results)
    elif "test_result" in ctx:
        # rank 完成但 rerank 未执行（如 --steps rank）——不再打印假日志
        logger.info("精排完成；rerank 未执行。运行: python main.py --steps rerank --resume 生成最终结果")
    logger.info("🎉 流水线执行完成")
