"""
This file is used to set the relative path to the root of the project.
"""

import importlib.util
spec = importlib.util.find_spec("NN-for-pulse-retrieval")
if spec is None or spec.origin == "namespace":
    import sys
    from pathlib import Path
    core_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(core_folder))