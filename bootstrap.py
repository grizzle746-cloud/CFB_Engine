# bootstrap.py
import importlib
import sys
import os

# Add engine folder to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import sports_engine
    importlib.reload(sports_engine)
    print("Sports engine loaded successfully.")
except Exception as e:
    print(f"Error loading sports engine: {e}")
