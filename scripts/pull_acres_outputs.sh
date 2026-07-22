#!/usr/bin/env bash
# =============================================================================
#  pull_acres_outputs.sh — rsync simulation outputs from ACRES down to this
#  machine. Outputs are deliberately untracked by git (large binaries, the
#  5MB hard rule), so this is the sync path Git can't cover.
#
#  Usage
#  -----
#    scripts/pull_acres_outputs.sh                  # pull everything (see PATHS below)
#    scripts/pull_acres_outputs.sh derivatives       # pull just data/derivatives/
#    scripts/pull_acres_outputs.sh clv               # pull just clv_results_july20/ + output/clv_null_results/
#    scripts/pull_acres_outputs.sh --dry-run         # show what would transfer, change nothing
#
#    # Any default is env-overridable:
#    ACRES_HOST=user@acres-head0.clarkson.edu scripts/pull_acres_outputs.sh
#    ACRES_ROOT=~/pythongpu scripts/pull_acres_outputs.sh
# =============================================================================
set -euo pipefail

ACRES_HOST="${ACRES_HOST:-atotilca@acres-head0.clarkson.edu}"
ACRES_ROOT="${ACRES_ROOT:-~/pythongpu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=""
MODE="all"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run" ;;
    derivatives|clv|all) MODE="$arg" ;;
    *) echo "usage: $0 [all|derivatives|clv] [--dry-run]" >&2; exit 2 ;;
  esac
done

# Common excludes: SLURM logs, egg-info, and anything else that's noise on
# the remote side but should never land here (mirrors .git/info/exclude).
RSYNC_FLAGS=(-avz --progress --prune-empty-dirs
  --exclude 'slurm-*.out' --exclude '*.egg-info' --exclude 'logs/'
  $DRY_RUN)

pull_one() {
  local remote_subpath="$1" local_subpath="$2"
  mkdir -p "$local_subpath"
  echo "+ rsync ${remote_subpath} -> ${local_subpath}"
  rsync "${RSYNC_FLAGS[@]}" \
    "${ACRES_HOST}:${ACRES_ROOT}/${remote_subpath}/" \
    "${local_subpath}/"
}

case "$MODE" in
  derivatives)
    pull_one "data/derivatives" "data/derivatives"
    ;;
  clv)
    pull_one "clv_results_july20"        "clv_results_july20"
    pull_one "output/clv_null_results"   "output/clv_null_results"
    ;;
  all)
    pull_one "data/derivatives"          "data/derivatives"
    pull_one "clv_results_july20"        "clv_results_july20"
    pull_one "output/clv_null_results"   "output/clv_null_results"
    ;;
esac

echo
echo "Done${DRY_RUN:+ (dry run — nothing was actually transferred)}."
