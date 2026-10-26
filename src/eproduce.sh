#!/usr/bin/env bash
# The one documented command the assignment brief asks for:
# clean checkout -> every number and figure in the write-up.
#
#   conda env create -f environment.yml
#   conda activate fgzsd-hicl-repro
#   bash reproduce.sh
#
# Each stage below is skip-if-already-done: if checkpoints/baseline.pt or
# checkpoints/hicl.pt already report finishing --epochs, src/train.py
# evaluates and exits instead of retraining (see "Resumability" in
# src/train.py). So re-running this script after an interruption, or after
# cloning the repo fresh, both do the right thing.
#
# Expect this to take hours on a single GPU the first time (15 epochs each
# for the baseline and the hicl run, plus the one-off dataset/hierarchy
# download) -- that is expected and allowed by the brief ("a reproduction
# that trains has to train").
set -euo pipefail

EPOCHS="${EPOCHS:-15}"
DATA_DIR="${DATA_DIR:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

echo "== 1/4: baseline (cross-entropy only) =="
python -m src.train --run baseline --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --epochs "$EPOCHS"

echo "== 2/4: treatment (cross-entropy + MSA-hicl) =="
python -m src.train --run hicl --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --epochs "$EPOCHS" --hicl-weight 0.1

echo "== 3/4: results.json (the headline number) =="
python -m src.evaluate --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --results-path results/results.json --epochs "$EPOCHS"

echo "== 4/4: t-SNE figure (the qualitative claim) =="
python -m src.tsne_figure --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --out figures/tsne_comparison.png

echo "Done. See results/results.json and figures/tsne_comparison.png"
