"""Make the adjacent G2 scripts importable during file-based test loading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
