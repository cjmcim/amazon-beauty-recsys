"""pytest 根 conftest：把项目根目录加入 sys.path，保证 `pytest tests/` 可直接运行。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
