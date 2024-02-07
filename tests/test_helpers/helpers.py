import sys
import pytest

skip_windows_due_to_parallel = pytest.mark.skipif(sys.platform.startswith("win"), reason="Parallel simulation not supported on windows due to fork")
