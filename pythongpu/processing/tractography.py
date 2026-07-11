#!/usr/bin/env python3
"""
MRtrix3-based structural connectome pipeline (methods plan item 7).

Preprocesses HCP diffusion data, fits constrained spherical deconvolution
(CSD), runs probabilistic tractography, and maps the resulting streamlines
onto a parcellation image via tck2connectome to produce a weighted
structural adjacency matrix -- the DTI-side counterpart to the functional
connectome produced by parcellation.py + causation_entropy.py.

HCP's T1w/Diffusion/data.nii.gz is already minimally preprocessed by HCP's
own pipeline (motion, eddy-current, EPI-susceptibility, and gradient-
nonlinearity corrected; Glasser et al. 2013). This script therefore starts
from response-function estimation, not from raw denoising -- rerunning
dwidenoise/eddy on HCP's own corrected output is neither necessary nor
standard practice.

Requires MRtrix3 on PATH (mrconvert, dwi2response, dwi2fod, tckgen,
tck2connectome). Anatomically-constrained tractography (ACT) is used if
--five_tt is supplied (a 5-tissue-type segmentation built from the
subject's T1w volume via 5ttgen); otherwise tractography falls back to
mask-constrained seeding using nodif_brain_mask.nii.gz.

Expected inputs under --diffusion-dir (HCP T1w/Diffusion/ layout):
    data.nii.gz             -- preprocessed 4D DWI volume
    bvals, bvecs            -- FSL-format gradient table
    nodif_brain_mask.nii.gz -- brain mask

--atlas must be a NIfTI label volume in the same space as the DWI data
(e.g. a Ward parcellation label image), not a parcellated timeseries CSV
-- tck2connectome reads voxel labels, not per-TR signal tables.

Run:
    python3 -m pythongpu.processing.tractography \\
        --diffusion-dir data/raw/nifti/100307/T1w/Diffusion \\
        --atlas data/processed/100307/atlas_240.nii.gz \\
        --out_csv data/processed/100307/structural_adjacency.csv \\
        --n_streamlines 10000000 --n_jobs 8 \\
        --log-file data/processed/100307/tractography.log
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_TOOLS = ["mrconvert", "dwi2response", "dwi2fod", "tckgen", "tck2connectome"]


def check_mrtrix_installed() -> None:
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "MRtrix3 is not installed or not on PATH -- missing: "
            + ", ".join(missing)
            + ". Install via `sudo apt install mrtrix3` or "
              "`conda install -c mrtrix3 mrtrix3`."
        )


def run(cmd: list[str], log: logging.Logger) -> None:
    log.info("$ " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log.info(f"  ({elapsed:.1f}s, exit={result.returncode})")
    if result.stdout.strip():
        log.debug(result.stdout)
    if result.returncode != 0:
        log.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def build_connectome(
    diffusion_dir: Path,
    atlas: Path,
    out_csv: Path,
    n_streamlines: int,
    n_jobs: int,
    work_dir: Path,
    five_tt: Path | None,
    log: logging.Logger,
) -> None:
    dwi = diffusion_dir / "data.nii.gz"
    bvecs = diffusion_dir / "bvecs"
    bvals = diffusion_dir / "bvals"
    mask = diffusion_dir / "nodif_brain_mask.nii.gz"

    for required in (dwi, bvecs, bvals, mask):
        if not required.exists():
            raise FileNotFoundError(f"missing required input: {required}")

    work_dir.mkdir(parents=True, exist_ok=True)
    dwi_mif = work_dir / "dwi.mif"
    fod = work_dir / "wmfod.mif"
    tracks = work_dir / "tracks.tck"

    # 1. Import DWI + gradient table into MRtrix's .mif format
    run([
        "mrconvert", str(dwi), str(dwi_mif),
        "-fslgrad", str(bvecs), str(bvals),
        "-nthreads", str(n_jobs), "-force",
    ], log)

    # 2. Response function estimation. dhollander handles both single- and
    #    multi-shell data unsupervised -- appropriate for HCP's multishell
    #    (b=1000/2000/3000) protocol without assuming a shell scheme.
    response_wm = work_dir / "response_wm.txt"
    response_gm = work_dir / "response_gm.txt"
    response_csf = work_dir / "response_csf.txt"
    run([
        "dwi2response", "dhollander", str(dwi_mif),
        str(response_wm), str(response_gm), str(response_csf),
        "-mask", str(mask), "-nthreads", str(n_jobs), "-force",
    ], log)

    # 3. Multi-shell multi-tissue constrained spherical deconvolution -> FODs
    run([
        "dwi2fod", "msmt_csd", str(dwi_mif),
        str(response_wm), str(fod),
        str(response_gm), str(work_dir / "gm.mif"),
        str(response_csf), str(work_dir / "csf.mif"),
        "-mask", str(mask), "-nthreads", str(n_jobs), "-force",
    ], log)

    # 4. Probabilistic tractography (iFOD2). ACT if a 5TT segmentation is
    #    given, otherwise mask-constrained whole-brain seeding.
    tckgen_cmd = [
        "tckgen", str(fod), str(tracks),
        "-algorithm", "iFOD2",
        "-select", str(n_streamlines),
        "-nthreads", str(n_jobs), "-force",
    ]
    if five_tt is not None:
        tckgen_cmd += [
            "-act", str(five_tt), "-backtrack", "-crop_at_gmwmi",
            "-seed_dynamic", str(fod),
        ]
    else:
        tckgen_cmd += ["-seed_image", str(mask), "-mask", str(mask)]
    run(tckgen_cmd, log)

    # 5. Map streamlines onto the parcellation -> weighted adjacency matrix
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run([
        "tck2connectome", str(tracks), str(atlas), str(out_csv),
        "-symmetric", "-zero_diagonal",
        "-nthreads", str(n_jobs), "-force",
    ], log)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--diffusion-dir", required=True, type=Path,
                     help="HCP T1w/Diffusion/ directory (data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz).")
    ap.add_argument("--atlas", required=True, type=Path,
                     help="Parcellation label volume (NIfTI), not the timeseries CSV.")
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--n_streamlines", type=int, default=10_000_000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--five_tt", type=Path, default=None,
                     help="Optional 5TT segmentation for anatomically-constrained tractography.")
    ap.add_argument("--work-dir", type=Path, default=None,
                     help="Intermediate .mif/.tck files (default: alongside out_csv).")
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    log = logging.getLogger("tractography")
    log.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)

    check_mrtrix_installed()

    if not args.atlas.exists():
        raise FileNotFoundError(
            f"atlas image not found: {args.atlas}. tck2connectome needs a "
            f"NIfTI label volume, not a parcellated_timeseries.csv."
        )

    work_dir = args.work_dir or (args.out_csv.parent / f".tractography_work_{args.out_csv.stem}")

    t0 = time.time()
    log.info(
        f"diffusion_dir={args.diffusion_dir}  atlas={args.atlas}  "
        f"n_streamlines={args.n_streamlines}  n_jobs={args.n_jobs}  "
        f"act={'yes' if args.five_tt else 'no'}"
    )
    build_connectome(
        args.diffusion_dir, args.atlas, args.out_csv, args.n_streamlines,
        args.n_jobs, work_dir, args.five_tt, log,
    )
    elapsed_min = (time.time() - t0) / 60
    log.info(f"structural adjacency matrix saved -> {args.out_csv}  ({elapsed_min:.1f} min)")


if __name__ == "__main__":
    main()
