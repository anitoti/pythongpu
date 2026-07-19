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
#    scripts/submit_clv_sweep.sh            # main sweep       (run_clv_sweep.sh)
#    scripts/submit_clv_sweep.sh null       # null-model sweep (run_clv_null_sweep.sh)
#    scripts/submit_clv_sweep.sh both       # submit both, back to back
#
#    # Override any default at submit time via environment variables:
#    QOS=short PARTITION=gpu WALLTIME=04:00:00 scripts/submit_clv_sweep.sh
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

# --- Overridable submission parameters (env vars win over these defaults) -----
PARTITION="${PARTITION:-gpu}"
WALLTIME="${WALLTIME:-08:00:00}"
QOS="${QOS-normal}"           # <-- VERIFY against sacctmgr output; set QOS= to omit
                              # (no-colon form so an explicit empty QOS= is kept,
                              #  not replaced by the default, and omits the flag)
GRES="${GRES:-gpu:1}"
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

  # Assemble sbatch flags. --qos is appended only when QOS is non-empty so an
  # empty QOS cleanly falls back to the cluster default instead of erroring.
  local -a flags=(
    --job-name="$jobname"
    --partition="$PARTITION"
    --time="$WALLTIME"
    --gres="$GRES"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --output="${logdir}/slurm-%j.out"
  )
  [ -n "$QOS" ] && flags+=(--qos="$QOS")

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
    echo "usage: $0 [main|null|both]" >&2
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
