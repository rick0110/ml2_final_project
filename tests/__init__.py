"""Shared test utilities and fixtures."""
import sys
from pathlib import Path

# Make sure the src package is importable when running tests from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
