"""Pytest config: make the DetV2 package importable without installation."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
