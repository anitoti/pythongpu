#!/usr/bin/env python3
"""Thin path-based entry point. Equivalent: `python3 -m pythongpu.networks.fmri_loader`.

Inserts the repo root on sys.path so `import pythongpu` resolves even when
this file is launched as `python3 scripts/run_correlate_fmri.py` (which otherwise only puts
scripts/ on the path, not the repo root).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pythongpu.networks.fmri_loader import main

if __name__ == "__main__":
    main()
