#!/usr/bin/env bash
# =============================================================================
#  submit_clv_sweep.sh — authoritative sbatch submitter for the CLV sweeps
#
#  Why this wrapper exists
#  -----------------------
#  The 15-minute timeouts (jobs 4555034 / 4555035) came from #SBATCH directives
#  being placed BELOW the first executable line of run_clv_sweep.sh /
#  run_clv_null_sweep.sh, where sbatch silently ignores them. The job therefore
#  inherited the cluster DEFAULT partition + walltime (the 15-min short/dev cap)
#  instead of the 8-hour gpu request in the header.
#
#  sbatch COMMAND-LINE flags always take precedence over in-script #SBATCH
#  directives, so passing -p / -t / --qos here guarantees the intended
#  partition, walltime and QoS regardless of the inner script's header. (The
#  inner scripts have also been fixed; this wrapper is the belt-and-braces path
#  the task asked for.)
#
#  Usage
#  -----
#    scripts/submit_clv_sweep.sh                 # main sweep, gpu partition
#    scripts/submit_clv_sweep.sh null            # null-model sweep
#    scripts/submit_clv_sweep.sh both            # submit both, back to back
#
#    # Optional 2nd positional arg = partition (wins over $PARTITION / gpu default):
#    scripts/submit_clv_sweep.sh main general    # submit the main sweep to 'general'
#
#    # CPU partitions (e.g. general) have no GPUs: drop the GPU request with an
#    # empty GRES so --gres is omitted (SLURM rejects --gres=gpu:1 on such nodes).
#    # NB: the inner run_clv_*.sh still carry `#SBATCH --gres=gpu:1`; for a true
#    # CPU run remove that directive too, else it re-requests a GPU.
#    GRES= scripts/submit_clv_sweep.sh main general
#
#    # Any default is env-overridable:
#    QOS=short WALLTIME=04:00:00 scripts/submit_clv_sweep.sh
#
#  IMPORTANT — confirm QOS before you rely on it
#  ---------------------------------------------
#  'normal' is the SLURM out-of-the-box default QoS name, but ACRES may use a
#  different one. An invalid --qos makes sbatch REJECT the job outright, so
#  confirm the valid name first:
#      sacctmgr show user "$USER" withassoc format=User,Account,Partition,QOS,DefaultQOS
#      scontrol show partition gpu        # check DefaultTime / MaxTime / QoS
#  then set QOS=<that name>. Setting QOS= (empty) omits the flag and falls back
#  to the cluster default QoS (always valid, but may re-impose a walltime cap).
# =============================================================================
set -euo pipefail

MODE="${1:-main}"

# --- Overridable submission parameters -----
# Precedence for PARTITION: 2nd positional arg > $PARTITION env var > 'gpu'.
PARTITION="${2:-${PARTITION:-gpu}}"
WALLTIME="${WALLTIME:-08:00:00}"
QOS="${QOS-normal}"           # <-- VERIFY against sacctmgr output; set QOS= to omit
                              # (no-colon form so an explicit empty QOS= is kept,
                              #  not replaced by the default, and omits the flag)
# GRES: no-colon form (like QOS) so an explicit empty GRES= is kept, not replaced
# by the default — required for CPU partitions (e.g. general) that have no GPUs.
GRES="${GRES-gpu:1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

submit_one() {
  local script="$1" jobname="$2" logdir="$3"
  if [ ! -f "$script" ]; then
    echo "ERROR: inner script not found: $script" >&2
    exit 1
  fi
  # Slurm opens the --output file before the job body runs, so the directory
  # must already exist at submit time.
  mkdir -p "$logdir"

  # Assemble sbatch flags. --gres and --qos are appended only when non-empty, so
  # an empty override cleanly omits the flag instead of forcing gpu:1 (which
  # SLURM rejects on CPU partitions) or passing an invalid QoS.
  local -a flags=(
    --job-name="$jobname"
    --partition="$PARTITION"
    --time="$WALLTIME"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --chdir="$REPO_ROOT"
    --output="${logdir}/slurm-%j.out"
  )
  [ -n "$GRES" ] && flags+=(--gres="$GRES")
  [ -n "$QOS" ]  && flags+=(--qos="$QOS")

  echo "+ sbatch ${flags[*]} $script"
  sbatch "${flags[@]}" "$script"
}

case "$MODE" in
  main)
    submit_one scripts/run_clv_sweep.sh      clv-sweep      output/clv_results
    ;;
  null)
    submit_one scripts/run_clv_null_sweep.sh clv-null-sweep output/clv_null_results
    ;;
  both)
    submit_one scripts/run_clv_sweep.sh      clv-sweep      output/clv_results
    submit_one scripts/run_clv_null_sweep.sh clv-null-sweep output/clv_null_results
    ;;
  *)
    echo "usage: $0 [main|null|both] [partition]" >&2
    exit 2
    ;;
esac

cat <<'EOF'

Submitted. Verify each job picked up the intended limits BEFORE walking away:

  squeue -u "$USER" -o '%.12i %.14j %.9P %.8T %.10M %.10l %.6q %R'
  #   the %.10l column is the TimeLimit, %.9P the partition, %.6q the QoS

  scontrol show job <JOBID> | grep -E 'Partition=|TimeLimit=|QOS=|JobState='

A correctly-submitted job shows  Partition=gpu  TimeLimit=08:00:00  and your QOS.
If TimeLimit still reads 00:15:00, the partition/QoS is capping it — re-check
`scontrol show partition gpu` (MaxTime) and the QoS MaxWall in sacctmgr.
EOF
