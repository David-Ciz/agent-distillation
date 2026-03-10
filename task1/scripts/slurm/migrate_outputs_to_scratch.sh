#!/usr/bin/env bash
# =============================================================================
# migrate_outputs_to_scratch.sh — One-time cleanup: move model checkpoints and
# final adapters from ~/agent-distillation/task1/outputs/ to scratch.
#
# Run this ONCE on LUMI to fix the home quota issue.
# After this, all future training jobs write directly to scratch (SCRATCH_OUTPUT_DIR).
#
# Usage (run on LUMI login node, not via sbatch):
#   bash task1/scripts/slurm/migrate_outputs_to_scratch.sh
#   bash task1/scripts/slurm/migrate_outputs_to_scratch.sh --dry-run
#
# What it does:
#   1. Moves *-lora-checkpoints/ and *-full-finetune-checkpoints/ to scratch
#      (intermediate checkpoints — large, safe to move or delete)
#   2. Moves *-lora-final/ and *-full-finetune-final/ to scratch
#      (final adapters — still needed for eval, so moved not deleted)
#   3. Leaves logs/, evaluations/, analysis/ in home (small, fine to keep)
#   4. Leaves symlinks in the old locations so existing path references still work
#
# After running:
#   du -sh ~/agent-distillation/task1/outputs/   # should be tiny
#   du -sh /scratch/project_465002758/$USER/agent-distillation/task1/outputs/
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be moved."
fi

HOME_OUTPUTS="${HOME}/agent-distillation/task1/outputs"
SCRATCH_OUTPUTS="/scratch/project_465002758/${USER}/agent-distillation/task1/outputs"

echo "=================================================="
echo "Migrate model outputs: home → scratch"
echo "  From: ${HOME_OUTPUTS}"
echo "  To:   ${SCRATCH_OUTPUTS}"
echo "=================================================="

if ! $DRY_RUN; then
    mkdir -p "${SCRATCH_OUTPUTS}"
fi

moved=0
skipped=0

for dir in "${HOME_OUTPUTS}"/*/; do
    [[ -d "$dir" ]] || continue

    name="$(basename "$dir")"

    # Only move model directories — leave logs, evaluations, analysis in place
    case "$name" in
        *-lora-checkpoints|*-full-finetune-checkpoints|\
        *-lora-final|*-full-finetune-final)
            ;;
        *)
            echo "  SKIP (not a model dir): $name"
            skipped=$((skipped + 1))
            continue
            ;;
    esac

    dest="${SCRATCH_OUTPUTS}/${name}"

    if [[ -e "$dest" ]]; then
        echo "  SKIP (already on scratch): $name"
        skipped=$((skipped + 1))
        continue
    fi

    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  MOVE [$size]  $dir  →  $dest"

    if ! $DRY_RUN; then
        mv "$dir" "$dest"
        # Leave a symlink so any hardcoded paths still resolve
        ln -s "$dest" "$dir"
        echo "        ↳ symlink left at $dir"
    fi

    moved=$((moved + 1))
done

echo ""
echo "=================================================="
echo "Done.  Moved: $moved   Skipped: $skipped"
if $DRY_RUN; then
    echo "(dry run — nothing actually moved)"
else
    echo ""
    echo "Home outputs dir size now:"
    du -sh "${HOME_OUTPUTS}" 2>/dev/null || true
    echo ""
    echo "Scratch outputs dir size:"
    du -sh "${SCRATCH_OUTPUTS}" 2>/dev/null || true
fi
echo "=================================================="

