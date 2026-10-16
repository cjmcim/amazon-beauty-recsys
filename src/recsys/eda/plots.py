"""EDA 图表工具：把 Spark 聚合结果转 pandas 后画图存盘（Agg 无头后端）。

统一使用英文标签，避免 Windows 默认字体缺字导致方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无图形界面环境
import matplotlib.pyplot as plt
import pandas as pd


def save_bar(pdf: pd.DataFrame, x_col: str, y_col: str, title: str,
             out_dir: str | Path, filename: str,
             xlabel: str = "", ylabel: str = "count", figsize=(8, 5),
             rot: int = 45) -> None:
    """柱状图（按 y 降序已排好的数据直接传入）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(pdf[x_col].astype(str), pdf[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rot)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=120)
    plt.close(fig)


def save_hist(series: pd.Series, title: str, out_dir: str | Path, filename: str,
              bins: int = 50, log_scale: bool = False) -> None:
    """直方图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=bins, log=log_scale)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=120)
    plt.close(fig)
