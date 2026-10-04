import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Hồ sơ học sinh khi chạy test được ghi vào thư mục tạm, không đụng tới dữ liệu thật
os.environ.setdefault("XUONG_PROFILES_DIR", tempfile.mkdtemp(prefix="xuong-test-profiles-"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

from core.knowledge import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load()
