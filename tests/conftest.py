"""pytest 配置：把项目根加入 sys.path，使 tests 能 import 各包。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
