# ACRES HPC Deployment Templates

Slurm sbatch templates for distributing the DTI-coupled Lorenz-63 basin
sweep across GPU nodes of the ACRES cluster, then compiling the per-coupling
basin frames into an animation.

| Template | Stage | Resources |
|----------|-------|-----------|
| [`sweep_array.slurm`](sweep_array.slurm) | Job array: one coupling coefficient K per element | GPU node (`--gres=gpu:1`), `--mem=32G` |
| [`compile_animation.slurm`](compile_animation.slurm) | Post-processing: frames → 12 fps MP4 via ffmpeg | CPU only (`short`), `--mem=8G` |

## Coupling-coefficient mapping

`sweep_array.slurm` maps `SLURM_ARRAY_TASK_ID` to a coupling coefficient K by
an affine transform over the closed interval `[K_MIN, K_MAX]`
(default `[0.45, 0.65]`):

```
K(a) = K_MIN + a * (K_MAX - K_MIN) / (M - 1)
```

where `a` is the task index and `M = SLURM_ARRAY_TASK_COUNT`. The array upper
bound selects the sampling density: `--array=0-20` yields 21 coefficients at
`delta-K = 0.01`; endpoints are inclusive.

## Two-stage submission

The compilation stage is released only when **all** array elements exit
cleanly, enforced with an `afterok` dependency:

```bash
ARRAY_ID=$(sbatch --parsable --array=0-20 pythongpu/hpc/sweep_array.slurm)
sbatch --dependency=afterok:${ARRAY_ID} pythongpu/hpc/compile_animation.slurm
```

Override defaults at submit time without editing the templates:

```bash
sbatch --array=0-40 \
       --export=ALL,K_MIN=0.45,K_MAX=0.65,GRID_N=361 \
       pythongpu/hpc/sweep_array.slurm
```

## Tunable variables

Both templates read the following (via `--export` or shell environment); each
has an in-file default:

| Variable | Default | Meaning |
|----------|---------|---------|
| `K_MIN`, `K_MAX` | `0.45`, `0.65` | Coupling-coefficient interval endpoints |
| `GRID_N` | `361` | Initial-condition grid points per axis |
| `K_CLUSTERS` | `auto` | Basin count (`auto` = VPS consensus selection) |
| `DTI_PATH` | `data/DTI_A.mat` | Structural connectivity matrix |
| `RESULTS_ROOT` | `data/coupling_sweep` | Per-task output root and frame gallery |
| `FRAME_RATE` | `12` | Output animation rate (frames/second) |

The Slurm resource directives themselves — `--ntasks`, `--gres=gpu:1`,
`--mem`, `--cpus-per-task` — are declared in each template header for flexible
scaling and may be overridden on the `sbatch` command line.

## Output layout

```
data/coupling_sweep/
├── task_0000_K0.4500/        # full per-node morphometry for one K
│   ├── basin_map_kmeans.png
│   ├── basin_boundary.png
│   ├── boxcount_loglog.png
│   └── basin_data.npz
├── task_0001_K0.4600/
│   └── ...
├── frames/                   # flat, K-ordered gallery consumed by ffmpeg
│   ├── frame_0000.png
│   └── frame_0001.png
└── coupling_sweep.mp4        # 12 fps compilation
```

## Notes

- **Partition names:** `gpu` and `short` match the partitions already used by
  the repository's existing submission scripts. If the ACRES partitions carry
  different identifiers, update the `#SBATCH --partition` lines (query with
  `sinfo -s`).
- **ffmpeg availability:** the base environment on this host has no
  system ffmpeg; `compile_animation.slurm` loads a cluster `FFmpeg` module. If
  no module is available, fall back to the bundled `cv2.VideoWriter(mp4v)`
  muxer in [`../pipeline/animate_coupling.py`](../pipeline/animate_coupling.py).
- **Module/venv:** templates load `Python/3.10.4-GCCcore-11.3.0` and activate
  `fmri_env`, matching `pythongpu/pipeline/slurm_submit.sh`.
