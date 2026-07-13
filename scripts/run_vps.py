#!/usr/bin/env python3
"""Thin path-based entry point. Equivalent: `python3 -m pythongpu.processing.feature_extraction`.

Inserts the repo root on sys.path so `import pythongpu` resolves even when
this file is launched as `python3 scripts/run_vps.py` (which otherwise only puts
scripts/ on the path, not the repo root).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pythongpu.processing.feature_extraction import main

if __name__ == "__main__":
    main()
