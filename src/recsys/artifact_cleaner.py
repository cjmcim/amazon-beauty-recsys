"""产物治理：盘点（--inspect）与安全清理（--clean）。

工业项目里「产物」分两类：
- 可重建的中间产物（checkpoint，由各 stage 落盘，`--resume` 可复用）→ 可安全清理；
- 最终产物（results / model / logs）→ 永不自动删。

设计原则（已与用户确认）：
1. 只清理 checkpoint 中间产物；results / model / logs 永不自动删；
2. 清理默认 dry-run：只打印"将删除什么"，加 --yes 才真正执行；
3. 纯文件系统操作，不依赖 Spark（秒级），不触发流水线；
4. 实验目录可单独清理中间产物（checkpoints/model/results/eda），
   保留 config_snapshot.yaml / meta.json / run_report.json / overview.csv / *.yaml。

用法（main.py 入口）：
    python main.py --inspect                     # 盘点：各产物大小 / checkpoint 完整状态
    python main.py --clean all --dry-run         # 查看将删除内容（默认行为，不真删）
    python main.py --clean split,feature --yes   # 删除指定阶段 checkpoint
    python main.py --clean experiments --yes     # 清理实验中间产物（保留留痕文件）
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.recsys.config import PathsConfig
from src.recsys.pipeline.runner import STAGE_ORDER

# checkpoint 落盘格式 → 文件后缀（与 pipeline/runner.py CheckpointManager 规则一致）
_KIND_SUFFIX = {
    "spark_df": ".parquet",
    "pandas": ".pkl",
    "booster": ".txt",
    "spark_model": "_model",
    "json": ".json",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _dir_bytes(path: Path) -> int:
    """目录总字节数（含子目录）；不存在返回 0。"""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _file_count(path: Path) -> int:
    """目录下文件数（含子目录）；不存在返回 0。"""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def _human_size(n: int) -> str:
    """字节数 → 人类可读（B / KB / MB / GB）。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _mtime_str(path: Path) -> str:
    """目录最后修改时间（YYYY-MM-DD HH:MM）；不存在返回 '-'。"""
    if not path.exists():
        return "-"
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# checkpoint 完整状态校验（与 CheckpointManager 同一套规则，纯文件校验）
# ---------------------------------------------------------------------------
def _artifact_file(stage_dir: Path, name: str, kind: str) -> Path:
    suffix = _KIND_SUFFIX.get(kind, ".json")
    return stage_dir / f"{name}{suffix}"


def stage_complete(stage_dir: Path) -> bool:
    """manifest.json 存在且其声明的每个产物文件都存在 → 完整。"""
    manifest = stage_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    for name, kind in info.items():
        if not _artifact_file(stage_dir, name, kind).exists():
            return False
    return True


def stage_artifact_count(stage_dir: Path) -> int:
    """manifest 声明的产物数（含缺失项）；无 manifest 返回 0。"""
    manifest = stage_dir / "manifest.json"
    if not manifest.exists():
        return 0
    try:
        return len(json.loads(manifest.read_text(encoding="utf-8")))
    except Exception:
        return 0


def inspect_stage(stage_dir: Path) -> Dict[str, Any]:
    """单个 stage checkpoint 的状态。"""
    exists = stage_dir.exists()
    return {
        "exists": exists,
        "complete": stage_complete(stage_dir) if exists else False,
        "bytes": _dir_bytes(stage_dir),
        "files": _file_count(stage_dir),
        "artifacts": stage_artifact_count(stage_dir),
        "mtime": _mtime_str(stage_dir),
    }


# ---------------------------------------------------------------------------
# 盘点（--inspect）
# ---------------------------------------------------------------------------
def _list_files(path: Path) -> List[Dict[str, Any]]:
    """目录下顶层文件清单（name + bytes）。"""
    if not path.exists():
        return []
    out = []
    for f in sorted(path.iterdir(), key=lambda p: p.name):
        if f.is_file():
            out.append({"name": f.name, "bytes": f.stat().st_size})
        else:
            out.append({"name": f.name + "/", "bytes": _dir_bytes(f)})
    return out


def inspect_outputs(paths: PathsConfig) -> Dict[str, Any]:
    """盘点全部产物，返回结构化结果（同时打印人类可读报告）。

    Args:
        paths: 普通模式（基线）的路径配置，outputs 须位于项目根下。
    """
    root = paths.outputs.parent  # 项目根（outputs/ 在根下）
    exp_root = root / "experiments"

    stages: Dict[str, Dict[str, Any]] = {}
    for s in STAGE_ORDER:
        stages[s] = inspect_stage(paths.checkpoints / s)

    result: Dict[str, Any] = {
        "project_root": str(root),
        "outputs_total_bytes": _dir_bytes(paths.outputs),
        "subdirs_bytes": {
            "checkpoints": _dir_bytes(paths.checkpoints),
            "results": _dir_bytes(paths.results),
            "model": _dir_bytes(paths.model_dir),
            "eda": _dir_bytes(paths.eda_plots) + _dir_bytes(paths.eda_tables),
            "logs": _dir_bytes(paths.logs),
        },
        "checkpoint_stages": stages,
        "results_files": _list_files(paths.results),
        "model_files": _list_files(paths.model_dir),
        "eda_plots_files": _list_files(paths.eda_plots),
        "eda_tables_files": _list_files(paths.eda_tables),
        "logs_files": _list_files(paths.logs),
    }

    # experiments 概览
    exp_count = 0
    exp_dirs: List[Dict[str, Any]] = []
    if exp_root.exists():
        overview = exp_root / "overview.csv"
        if overview.exists():
            exp_count = max(0, sum(1 for _ in overview.open(encoding="utf-8")) - 1)
        for d in sorted(exp_root.iterdir()):
            if d.is_dir():
                exp_dirs.append({"name": d.name, "bytes": _dir_bytes(d)})
    result["experiments"] = {"count": exp_count, "dirs": exp_dirs, "root": str(exp_root)}

    _print_inspect(result)
    return result


def _print_inspect(r: Dict[str, Any]) -> None:
    """打印人类可读盘点报告（中文、对齐、小白友好）。"""
    line = "=" * 46
    print(f"\n{line}\n产物盘点  {r['project_root']}\n{line}")

    sub = r["subdirs_bytes"]
    print(f"\noutputs/ 总大小: {_human_size(r['outputs_total_bytes'])}")

    print(f"\n[1] checkpoints/ 中间产物（可重建，可清理）: {_human_size(sub['checkpoints'])}")
    mark = {"ok": "✔ 完整", "bad": "✘ 不完整", "none": "- 无产物"}
    for s, st in r["checkpoint_stages"].items():
        if not st["exists"]:
            m = mark["none"]
        elif st["complete"]:
            m = mark["ok"]
        else:
            m = mark["bad"]
        print(f"    {s:<8} {_human_size(st['bytes']):>10}  {m}  "
              f"{st['mtime']}   {st['artifacts']} 个产物")

    def _dump(section: str, files: List[Dict[str, Any]], bytes_: int) -> None:
        print(f"\n[{section}] 最终产物（永不自动删）: {_human_size(bytes_)}")
        if not files:
            print("    （空）")
        for f in files:
            print(f"    {f['name']:<32} {_human_size(f['bytes']):>10}")

    _dump("2 results/", r["results_files"], sub["results"])
    _dump("3 model/", r["model_files"], sub["model"])
    print(f"\n[4] eda/ 图表与统计: {_human_size(sub['eda'])}  "
          f"(plots {len(r['eda_plots_files'])} 项 / tables {len(r['eda_tables_files'])} 项)")
    _dump("5 logs/", r["logs_files"], sub["logs"])

    exp = r["experiments"]
    print(f"\n[6] experiments/ 实验记录: {exp['count']} 个实验（overview.csv）")
    for d in exp["dirs"]:
        print(f"    {d['name']:<24} {_human_size(d['bytes']):>10}")

    print(f"\n{line}\n清理入口（默认只打印，--yes 才执行）:\n"
          f"    python main.py --clean all --dry-run\n"
          f"    python main.py --clean split,feature --yes\n"
          f"    python main.py --clean experiments --yes\n{line}\n")


# ---------------------------------------------------------------------------
# 清理（--clean）
# ---------------------------------------------------------------------------
def parse_clean_targets(value: str) -> List[str]:
    """解析 --clean 参数：逗号分隔的阶段名 / all / experiments。

    规则：元素必须是合法 stage 或 all / experiments；
    all 表示全部 checkpoint，不能与其他目标混用。
    """
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("--clean 参数为空，示例: --clean split,feature / --clean all")
    if "all" in items and len(items) > 1:
        raise ValueError("--clean all 表示清理全部 checkpoint，不能与其他目标混用")
    for it in items:
        if it not in STAGE_ORDER and it not in ("all", "experiments"):
            raise ValueError(
                f"未知清理目标: {it}。可选: 阶段 {STAGE_ORDER} / all / experiments"
            )
    return items


def clean_checkpoints(stages: List[str], checkpoints_root: Path,
                      yes: bool = False) -> Dict[str, Any]:
    """清理指定 stage 的 checkpoint 目录。

    Args:
        stages: stage 名列表（"all" 会在此前展开为 STAGE_ORDER）。
        checkpoints_root: outputs/checkpoints 目录。
        yes: False 只打印将删除内容（dry-run）；True 真正删除。

    Returns:
        {"dry_run": bool, "deleted": [相对路径], "freed_bytes": int}
    """
    if "all" in stages:
        stages = list(STAGE_ORDER)
    targets = [checkpoints_root / s for s in stages if (checkpoints_root / s).exists()]

    freed = sum(_dir_bytes(t) for t in targets)
    print(f"\n清理 checkpoint（{len(targets)} 个阶段目录，释放 {_human_size(freed)}）:")
    for t in targets:
        print(f"    删除  {t.relative_to(checkpoints_root)}  "
              f"({_human_size(_dir_bytes(t))}, {_file_count(t)} 个文件)")
    if not targets:
        print("    无已存在的 checkpoint，无需清理。")

    if not yes:
        print("    [dry-run] 未删除任何文件；确认后加 --yes 执行。")
        return {"dry_run": True, "deleted": [], "freed_bytes": freed}

    deleted = []
    for t in targets:
        shutil.rmtree(t)
        deleted.append(str(t.relative_to(checkpoints_root)))
    print(f"✔ 已删除 {len(deleted)} 个 checkpoint 阶段目录。")
    return {"dry_run": False, "deleted": deleted, "freed_bytes": freed}


def clean_experiments(exp_root: Path, yes: bool = False) -> Dict[str, Any]:
    """清理所有实验的中间产物（checkpoints/model/results/eda）。

    保留留痕文件：*.yaml（实验配置）、config_snapshot.yaml、meta.json、
    run_report.json、overview.csv —— 实验记录与可复现信息不删。
    """
    if not exp_root.exists():
        print(f"\nexperiments/ 不存在（{exp_root}），无需清理。")
        return {"dry_run": not yes, "deleted": [], "freed_bytes": 0}

    targets: List[Path] = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        for sub in ("checkpoints", "model", "results", "eda"):
            p = d / sub
            if p.exists():
                targets.append(p)

    freed = sum(_dir_bytes(t) for t in targets)
    print(f"\n清理实验中间产物（{len(targets)} 个目录，释放 {_human_size(freed)}）:")
    for t in targets:
        print(f"    删除  {t.relative_to(exp_root)}  "
              f"({_human_size(_dir_bytes(t))}, {_file_count(t)} 个文件)")
    print("    保留  config_snapshot.yaml / meta.json / run_report.json / overview.csv / *.yaml")
    if not targets:
        print("    无实验中间产物，无需清理。")

    if not yes:
        print("    [dry-run] 未删除任何文件；确认后加 --yes 执行。")
        return {"dry_run": True, "deleted": [], "freed_bytes": freed}

    deleted = []
    for t in targets:
        shutil.rmtree(t)
        deleted.append(str(t.relative_to(exp_root)))
    print(f"✔ 已清理 {len(deleted)} 个实验中间产物目录。")
    return {"dry_run": False, "deleted": deleted, "freed_bytes": freed}
