"""日志工具：统一输出到控制台 + 滚动文件，替代散落的 print。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "recsys"
_CONFIGURED = False


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """初始化根日志器（幂等）。

    - 控制台：INFO 级别，方便观察进度
    - 文件：outputs/logs/recsys.log，滚动保留 5 个文件，每个 20MB

    返回项目统一的 logger（模块内用 ``logging.getLogger("recsys")`` 即可拿到）。
    """
    global _CONFIGURED
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:  # 重复初始化只清空 handler，避免日志翻倍
        logger.handlers.clear()

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "recsys.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取子模块 logger：``get_logger(__name__)`` 会在日志里显示模块路径。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
