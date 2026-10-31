"""pytest 配置：确保项目根目录在模块搜索路径里。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
