# This conftest.py at the root ensures pytest adds the project root to sys.path,
# allowing `from src.detector import FaceDetector` to resolve correctly.
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
