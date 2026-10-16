"""SparkSession 工厂：全项目共享同一个 SparkSession（单例）。

原 Jupyter 脚本每个文件各自 ``SparkSession.builder...getOrCreate()``，
迁移后统一由本模块创建、复用、销毁，避免资源重复占用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

from src.recsys.config import SparkConfig
from src.recsys.logging_utils import get_logger

logger = get_logger("spark_utils")

_spark: SparkSession | None = None

# Windows 下 PySpark 写本地文件（parquet/checkpoint）需要 winutils.exe，
# 否则 Hadoop 报 HADOOP_HOME and hadoop.home.dir are unset。
# 工程自包含方案：winutils 随项目放在 tools/winutils/bin/，代码自动探测并注入，
# 用户无需配置任何系统环境变量。
_WINUTILS_DIR = Path(__file__).resolve().parents[2] / "tools" / "winutils"


def _configure_winutils(builder: Any) -> Any:
    """Windows 下把 winutils 路径注入 Spark（仅 Windows 生效，其他系统跳过）。"""
    if os.name != "nt":
        return builder
    winutils_dir = _WINUTILS_DIR
    winutils_bin = winutils_dir / "bin"
    if not (winutils_bin / "winutils.exe").exists():
        logger.warning("未找到 winutils: %s（Windows 下写 parquet 可能报 HADOOP_HOME unset）",
                       winutils_bin / "winutils.exe")
        return builder

    # 1) HADOOP_HOME：让 Hadoop 找到 winutils.exe（写文件时设置权限）
    os.environ["HADOOP_HOME"] = str(winutils_dir)

    # 2) 把 hadoop.dll 所在目录加进 PATH：JVM 的默认 java.library.path 包含 PATH，
    #    这样才能加载 hadoop.dll（否则报 UnsatisfiedLinkError: NativeIO$Windows.access0）
    win_bin_str = str(winutils_bin)
    if win_bin_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = win_bin_str + os.pathsep + os.environ.get("PATH", "")

    # 3) Spark/Hadoop 配置属性（双保险，指向 winutils 所在目录）
    # 4) Windows 本地模式经典修复：强制 driver 绑定回环地址。
    #    否则 driver 绑定外部 IP（如校园网 10.x.x.x），DHCP 变化/防火墙拦截
    #    会导致 executor 拉取任务结果 Connection refused → TaskResultLost。
    return (
        builder
        .config("spark.hadoop.hadoop.home.dir", str(winutils_dir))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )


def get_spark(cfg: SparkConfig) -> SparkSession:
    """返回全局唯一的 SparkSession（懒加载）。"""
    global _spark
    if _spark is not None:
        return _spark

    builder = (
        SparkSession.builder
        .appName(cfg.app_name)
        .master(cfg.master)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.driver.maxResultSize", cfg.driver_max_result_size)
        .config("spark.memory.fraction", str(cfg.memory_fraction))
        .config("spark.memory.storageFraction", str(cfg.storage_fraction))
    )
    if cfg.adaptive_enabled:
        builder = builder.config("spark.sql.adaptive.enabled", "true")

    if cfg.hadoop_local_fs_fix:
        # Windows 无 winutils 时的规避：使用本地文件系统实现并关闭权限校验
        # 注意：spark.sql.autoBroadcastJoinThreshold 已废弃，统一用 adaptive 版本
        builder = (
            builder
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
            .config("spark.hadoop.fs.file.permissions.enabled", "false")
            .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
        )

    builder = _configure_winutils(builder)

    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession 初始化完成: app=%s master=%s", cfg.app_name, cfg.master)
    return _spark


def stop_spark() -> None:
    """结束会话（可选调用；正常跑完流水线建议调用以释放资源）。"""
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
        logger.info("SparkSession 已停止")
