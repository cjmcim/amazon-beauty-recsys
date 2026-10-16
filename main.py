"""Amazon Beauty 离线推荐系统 - 统一入口。

用法（在项目根目录执行）：
    python main.py                                # 全流程
    python main.py --steps eda                    # 只跑 EDA
    python main.py --steps clean,split            # 指定阶段
    python main.py --steps rank,rerank --resume   # 复用已有 checkpoint
    python main.py --config config/config.yaml --no-checkpoint
    # 实验模式（自研实验管理）：
    python main.py --exp exp01                    # 基线 + experiments/exp01.yaml 覆盖
    python main.py --list-exp                     # 列出所有实验
    python main.py --compare                      # 打印实验对比表
    # 产物治理：
    python main.py --inspect                      # 盘点产物（大小 / checkpoint 完整状态）
    python main.py --clean all --dry-run          # 查看将删除内容（不真删）
    python main.py --clean split,feature --yes    # 删除指定阶段 checkpoint
    python main.py --clean experiments --yes      # 清理实验中间产物（保留留痕）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.recsys.pipeline.runner import STAGE_ORDER, main as run_pipeline

DEFAULT_CONFIG = "config/config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amazon Beauty 离线推荐系统")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--steps", default="all",
                        help="要执行的 stage，逗号分隔；all 表示全部（默认）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已存在 checkpoint 的 stage，直接加载")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="关闭中间产物落盘")
    parser.add_argument("--exp", default=None,
                        help="实验名（experiments/<name>.yaml），开启实验模式："
                             "基线配置 + 实验覆盖，产物归档到 experiments/<name>/")
    parser.add_argument("--list-exp", action="store_true",
                        help="列出所有实验（不运行流水线）")
    parser.add_argument("--compare", action="store_true",
                        help="打印实验对比表 overview.csv（不运行流水线）")
    parser.add_argument("--inspect", action="store_true",
                        help="盘点产物：各目录大小 / checkpoint 完整状态（不运行流水线）")
    parser.add_argument("--clean", default=None,
                        help="清理 checkpoint 中间产物：逗号分隔的阶段名（clean,split,feature…）"
                             "或 all（全部）或 experiments（实验中间产物）。"
                             "默认 dry-run 只打印将删除内容，加 --yes 才执行")
    parser.add_argument("--dry-run", action="store_true",
                        help="--clean 时显式声明只打印不删除（默认行为，可省略）")
    parser.add_argument("--yes", action="store_true",
                        help="--clean 时跳过 dry-run 确认，真正执行删除")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 查询模式：不跑流水线 ----
    if args.list_exp or args.compare:
        from src.recsys.experiment import ExperimentManager

        exp_root = Path(args.config).resolve().parent.parent / "experiments"
        mgr = ExperimentManager(exp_root)
        if args.list_exp:
            mgr.print_experiments()
        if args.compare:
            mgr.print_compare()
        return

    # ---- 产物治理模式：盘点 / 清理（不跑流水线，不起 Spark） ----
    if args.inspect or args.clean:
        from src.recsys.artifact_cleaner import (
            clean_checkpoints,
            clean_experiments,
            inspect_outputs,
            parse_clean_targets,
        )
        from src.recsys.config import load_config

        cfg = load_config(args.config)
        if args.inspect:
            inspect_outputs(cfg.paths)
        if args.clean:
            targets = parse_clean_targets(args.clean)
            project_root = Path(args.config).resolve().parent.parent
            exp_root = project_root / "experiments"
            for t in targets:
                if t == "experiments":
                    clean_experiments(exp_root, args.yes)
                else:
                    stages = STAGE_ORDER if t == "all" else [t]
                    clean_checkpoints(stages, cfg.paths.checkpoints, args.yes)
        return

    steps = args.steps.split(",")
    if steps != ["all"]:  # "all" 在 runner 里展开为配置中的完整 stage 列表
        for s in steps:
            if s not in STAGE_ORDER:
                raise SystemExit(f"未知 stage: {s}，可选: {', '.join(STAGE_ORDER)}")
    run_pipeline(
        config_path=args.config,
        steps=steps,
        resume=args.resume,
        checkpoint=False if args.no_checkpoint else None,
        exp_name=args.exp,
    )


if __name__ == "__main__":
    main()
