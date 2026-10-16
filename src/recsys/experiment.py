"""轻量实验管理（自研版，零外部依赖）。

目标：把「改配置 → 跑 → 结果被覆盖」变成「写实验配置 → 跑 → 自动留痕 → 可对比可复现」。

目录约定（均在项目根 /experiments 下）：
    experiments/
    ├── <exp_name>.yaml          # 实验配置：只写要覆盖基线的参数
    ├── <exp_name>/              # 运行后自动生成
    │   ├── config_snapshot.yaml # 合并后的完整配置快照（复现的钥匙）
    │   ├── meta.json            # git commit / 时间 / 命令 / 基线配置 hash
    │   ├── run_report.json      # 各 stage 耗时 + 数据量（行数/用户数）+ 可序列化指标
    │   ├── checkpoints/         # 本实验独立的中间产物（实验隔离，互不污染）
    │   ├── model/  results/  eda/
    └── overview.csv             # 全局对比表：每个实验一行

用法（见 main.py）：
    python main.py --exp exp01           # 基线 + exp01.yaml 覆盖，实验模式运行
    python main.py --list-exp            # 列出所有实验
    python main.py --compare             # 打印实验对比表
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# overview.csv 固定列（缺失留空，顺序即展示顺序）
OVERVIEW_COLUMNS = [
    "exp_id", "created_at", "git_commit", "status",
    "rank_auc",
    "rank_Recall@5", "rank_Recall@10", "rank_Recall@20",
    "rank_NDCG@5", "rank_NDCG@10", "rank_NDCG@20",
    "rerank_Recall@5", "rerank_NDCG@5", "rerank_diversity",
    "elapsed_sec",
]

_RANK_K = ["Recall@5", "Recall@10", "Recall@20",
           "NDCG@5", "NDCG@10", "NDCG@20"]


def get_git_commit(repo_root: Path) -> str:
    """取当前 git HEAD 短信息；无 git 环境时返回 'unknown'。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class ExperimentManager:
    """实验目录管理与归档写入。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 路径
    def config_file(self, exp_name: str) -> Path:
        return self.root / f"{exp_name}.yaml"

    def exp_dir(self, exp_name: str) -> Path:
        return self.root / exp_name

    def overview_file(self) -> Path:
        return self.root / "overview.csv"

    def ensure_exp_dir(self, exp_name: str) -> Path:
        d = self.exp_dir(exp_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------- 归档写入
    def save_config_snapshot(self, exp_name: str, cfg_dict: dict) -> Path:
        """合并后的完整配置快照（YAML），这是「复现」的钥匙。"""
        path = self.ensure_exp_dir(exp_name) / "config_snapshot.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
        return path

    def save_meta(self, exp_name: str, meta: dict) -> Path:
        path = self.ensure_exp_dir(exp_name) / "meta.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def save_run_report(self, exp_name: str, report: dict) -> Path:
        path = self.ensure_exp_dir(exp_name) / "run_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def append_overview(self, row: Dict[str, Any]) -> None:
        """追加一行到 overview.csv；文件不存在时自动建表头。"""
        new_df = pd.DataFrame([row])
        new_df = new_df.reindex(columns=OVERVIEW_COLUMNS)
        out = self.overview_file()
        if out.exists():
            old = pd.read_csv(out)
            merged = pd.concat([old, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_csv(out, index=False, encoding="utf-8")

    # -------------------------------------------------------------- 查询展示
    def _load_overview(self) -> Optional[pd.DataFrame]:
        out = self.overview_file()
        if not out.exists():
            return None
        df = pd.read_csv(out)
        if df.empty:
            return None
        return df

    def print_experiments(self) -> None:
        """列出所有实验（简洁摘要）。"""
        df = self._load_overview()
        if df is None:
            print(f"暂无实验记录（{self.overview_file()} 不存在）。")
            return
        cols = [c for c in OVERVIEW_COLUMNS if c in df.columns]
        print(f"共 {len(df)} 个实验，记录文件: {self.overview_file()}")
        print(df[cols].to_string(index=False))

    def print_compare(self) -> None:
        """打印实验对比表（每个实验一行，横向对比指标）。"""
        df = self._load_overview()
        if df is None:
            print("暂无实验记录，先跑一个实验（python main.py --exp <name>）再对比。")
            return
        cols = [c for c in OVERVIEW_COLUMNS if c in df.columns]
        print(f"实验对比表（{len(df)} 个实验）:")
        print(df[cols].to_string(index=False))

    # -------------------------------------------------------------- 收尾归档
    def finalize(
        self,
        exp_name: str,
        cfg,
        elapsed: Dict[str, float],
        stage_metrics: Dict[str, Any],
        command: str,
        config_path: Optional[Path] = None,
        started_at: Optional[str] = None,
        stage_volumes: Optional[Dict[str, Dict]] = None,
    ) -> Path:
        """流水线正常结束后统一归档：配置快照 / meta / 运行报告 / overview 行。

        必须在 Spark 资源释放之前调用（部分数据量统计依赖缓存命中）。

        Args:
            elapsed: stage 名 → 耗时秒数（resume 加载的 stage 为 None，表示非本次计算）
            stage_metrics: stage 名 → 产物中可序列化的指标
            stage_volumes: stage 名 → 关键数据量（行数/用户数等），写入 run_report 的 rows
        """
        exp_dir = self.ensure_exp_dir(exp_name)
        finished_at = datetime.now().isoformat(timespec="seconds")
        started_at = started_at or finished_at

        # 1) 配置快照
        self.save_config_snapshot(exp_name, cfg.to_dict())

        # 2) meta 信息
        repo_root = Path(config_path).resolve().parent.parent if config_path else self.root.parent
        meta = {
            "exp_id": exp_name,
            "created_at": finished_at,
            "git_commit": get_git_commit(repo_root),
            "command": command,
            "python_version": sys.version.split()[0],
            "base_config_sha256": _file_sha256(config_path) if config_path else None,
            "log_file": str(cfg.paths.logs / "recsys.log"),
        }
        self.save_meta(exp_name, meta)

        # 3) 运行报告（stage 耗时 + 数据量 + 各 stage 可序列化指标）
        report = {
            "exp_id": exp_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": round(sum(v for v in elapsed.values() if v is not None), 2),
            "stages": self._build_stage_report(elapsed, stage_volumes),
            "stage_metrics": stage_metrics,
        }
        self.save_run_report(exp_name, report)

        # 4) overview 对比表行
        row = self._build_overview_row(exp_name, meta, report, stage_metrics)
        self.append_overview(row)
        return exp_dir

    @staticmethod
    def _build_stage_report(elapsed: Dict[str, float],
                            stage_volumes: Optional[Dict[str, Dict]]) -> Dict[str, Any]:
        """组装 stages 字段：耗时 + 数据量（rows）。"""
        out: Dict[str, Any] = {}
        for k, v in elapsed.items():
            entry: Dict[str, Any] = {"elapsed_sec": v}
            if stage_volumes and stage_volumes.get(k):
                entry["rows"] = stage_volumes[k]
            out[k] = entry
        return out

    def _build_overview_row(self, exp_name: str, meta: dict, report: dict,
                            stage_metrics: Dict[str, Any]) -> Dict[str, Any]:
        rank_sm = stage_metrics.get("rank", {})
        rerank_sm = stage_metrics.get("rerank", {})
        rerank_reorder = rerank_sm.get("metrics", {}).get("reorder", {})
        row: Dict[str, Any] = {
            "exp_id": exp_name,
            "created_at": meta["created_at"],
            "git_commit": meta["git_commit"],
            "status": "done",
            "rank_auc": rank_sm.get("auc"),
        }
        for k in _RANK_K:
            row[f"rank_{k}"] = rank_sm.get("metrics", {}).get(k)
        row["rerank_Recall@5"] = rerank_reorder.get("Recall@5")
        row["rerank_NDCG@5"] = rerank_reorder.get("NDCG@5")
        row["rerank_diversity"] = rerank_sm.get("diversity", {}).get("reorder")
        row["elapsed_sec"] = report["elapsed_sec"]
        return row
