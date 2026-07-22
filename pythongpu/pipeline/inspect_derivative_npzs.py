import argparse
import re
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = next(
    (
        candidate
        for candidate in (
            _REPO_ROOT / "data" / "derivatives",
            _REPO_ROOT.parent / "data" / "derivatives",
        )
        if candidate.exists()
    ),
    _REPO_ROOT / "data" / "derivatives",
)


def _k_from_name(path: Path):
    match = re.search(r"_K([0-9]+(?:\.[0-9]+)?)\.npz$", path.name)
    return float(match.group(1)) if match else None


def _nan_count(array) -> int:
    data = np.asarray(array, dtype=float)
    return int(np.isnan(data).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely inspect every .npz under data/derivatives/ and report gamma_sign stats."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory to search recursively for .npz files.",
    )
    args = parser.parse_args()

    npz_files = sorted(args.root.rglob("*.npz"))
    print(f"root: {args.root}")
    print(f"npz files: {len(npz_files)}")

    one_element_nan_files = []
    k05_one_element_nan = []

    for path in npz_files:
        try:
            with np.load(path, allow_pickle=True) as f:
                gamma_sign = f["gamma_sign"]
                nan_count = _nan_count(gamma_sign)
                shape = gamma_sign.shape
                size = gamma_sign.size
                print(
                    f"{path.relative_to(args.root)}: "
                    f"shape={shape}, size={size}, nan_count={nan_count}"
                )
                if shape == (1,) and nan_count == 1:
                    one_element_nan_files.append(path)
                    if _k_from_name(path) == 0.5:
                        k05_one_element_nan.append(path)
        except KeyError:
            print(f"{path.relative_to(args.root)}: missing gamma_sign")
        except Exception as exc:
            print(f"{path.relative_to(args.root)}: ERROR {type(exc).__name__}: {exc}")

    print("\nsummary")
    print(f"1-element NaN gamma_sign files: {len(one_element_nan_files)}")
    print(f"K=0.5 files among them: {len(k05_one_element_nan)}")
    if len(one_element_nan_files) == 1 and k05_one_element_nan:
        print("K=0.5 is the only file with a 1-element NaN gamma_sign array.")
    else:
        print("K=0.5 is not the only file with a 1-element NaN gamma_sign array.")


if __name__ == "__main__":
    main()
