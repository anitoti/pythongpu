#!/usr/bin/env python3
"""Path-based entry point for inspecting derivative .npz files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pythongpu.pipeline.inspect_derivative_npzs import main


if __name__ == "__main__":
    main()
