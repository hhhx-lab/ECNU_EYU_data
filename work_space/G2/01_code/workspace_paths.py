"""Bridge script-local imports to the project's dependency-free path helper."""
import importlib.util
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "PROJECT_RULES.md").is_file())
_spec = importlib.util.spec_from_file_location("_ecnu_project_paths", _root / "tools/project_paths.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
get_path = _module.get_path
resolve_legacy_path = _module.resolve_legacy_path
