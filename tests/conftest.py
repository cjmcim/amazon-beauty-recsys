"""pytest 共享 fixture：session 级 SparkSession + 守卫测试基础设施。

数据守卫测试（tests/test_guards.py）需要 Spark 对小数据做窗口/join/聚合计算。
整个测试会话只启动一次 Spark（约 10~20 秒），用后销毁；纯逻辑测试不受影响。
"""

from __future__ import annotations

import os
import sys

import pytest

from src.recsys.config import SparkConfig
from src.recsys.spark_utils import get_spark, stop_spark


@pytest.fixture(scope="session")
def spark():
    """session 级 SparkSession（复用项目 spark_utils 的 winutils 注入与单例管理）。

    必须显式设置 PYSPARK_PYTHON：pyspark 未设置时默认用 ``python3`` 命令按 PATH
    解析，本机 PATH 首位是其他 Python 解释器（缺 pyspark），会导致 Python worker
    启动失败（PythonWorkerFactory / ServerSocket.accept 超时）。
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable  # worker 与 driver 同解释器
    cfg = SparkConfig(
        app_name="recsys-tests",
        master="local[1]",
        driver_memory="1g",
        shuffle_partitions=2,
        adaptive_enabled=False,
        hadoop_local_fs_fix=True,
    )
    s = get_spark(cfg)
    yield s
    stop_spark()
