import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_import_research_modules():
    modules = [
        "src",
        "src.utils",
        "src.indicators",
        "src.market_structure",
        "src.signals",
        "src.risk_manager",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None
