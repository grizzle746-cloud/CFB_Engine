# tests/conftest.py
import os
import sys
import importlib.util

# Project root (one level up from tests/)
ROOT = os.path.dirname(os.path.dirname(__file__))
ENGINE_PATH = os.path.join(ROOT, "engine.py")

# Load engine.py as module "engine" and ensure it is in sys.modules
# BEFORE executing (important for dataclasses / __module__ lookups).
spec = importlib.util.spec_from_file_location("engine", ENGINE_PATH)
engine_mod = importlib.util.module_from_spec(spec)
sys.modules["engine"] = engine_mod          # <-- insert first
assert spec.loader is not None
spec.loader.exec_module(engine_mod)         # <-- then execute
