# Does MSA-hicl help a plain bird classifier? A scoped reproduction

This repo reproduces one claim from:

> **Fine-Grained Zero-Shot Object Detection** (FG-ZSD / MSHC), arXiv:2507.10358v1

It does **not** reproduce the paper's actual headline result (zero-shot
*detection* of bird species with no training images, using the paper's
custom FGZSD-Birds dataset and MSA generative network). Both of those are
out of reach for a course project: FGZSD-Birds is not publicly released,
and the generator is a second, separately-trained model with its own text
encoder. Re-running the authors' own eval script, even if it were
available, would not test anything — see the brief's own floor for what
counts as a reproduction.

## The claim being tested

> Does adding the paper's hierarchical contrastive loss, **MSA-hicl**
> (Eq. 10–13, Sec. 4.2), to a standard image classifier produce
> **(a)** higher accuracy on *seen* classes, and
> **(b)** features that visibly cluster by genus —
> compared to training with cross-entropy alone?

This isolates the one piece of the paper's method that still makes sense
without the generator: a loss that shapes the classifier's feature space
using the real taxonomy, rather than treating every wrong class as equally
wrong. It is tested on public data (CUB-200-2011) with a plain classifier
(ResNet-101 + linear head), not a detector, and on *seen* classes only —
there is no zero-shot element here at all. That is a large, deliberate
reduction in scope from the paper's own claim, stated up front rather than
discovered by a reader three paragraphs in.

## What was actually run, and the result

| Run | Loss | Test accuracy (seen classes) |
|---|---|---|
| Baseline | cross-entropy only | 77.18% |
| Treatment | cross-entropy + MSA-hicl (weight 0.1) | **79.32%** |

**Difference: +2.14 points.** Both models were trained for 15 epochs,
Adam, lr 1e-4, batch size 32, seed 0, on the official CUB-200-2011
train/test split (5,994 train / 5,794 test images, 200 classes).

The qualitative claim — that MSA-hicl produces tighter genus-level
clustering — is only **weakly supported** by `figures/tsne_comparison.png`
(8 test species, 4 genus pairs), and is worth reading pair by pair rather
than as a single average impression. **Buntings** and **Auklets** are
already cleanly, tightly separated under plain cross-entropy alone and show
**no visible change** once MSA-hicl is added. **Blackbirds** — the most
closely related pair in this subset, and the one the paper's own argument
would predict benefits most from a taxonomy-aware loss — remain
**substantially intermixed under both losses**; MSA-hicl does not visibly
fix the hardest case. **Albatrosses** change shape between the two panels
(a different arrangement) without becoming obviously tighter. On this
subset, pairs that were already easy stay easy and the pair that was hard
stays hard, under both losses — this figure does **not** independently
confirm the genus-clustering mechanism beyond the quantitative accuracy
gain above. See `results/results.json` for the exact numbers and
`REPORT.md` for the full discussion of what this does and doesn't show
(including the number the original paper never reports).

## Repository layout

```
src/
  download_data.py   fetch + cache CUB-200-2011 (the official host 404s/403s; see docstring)
  hierarchy.py        real bird taxonomy from cvjena/semantic-embeddings -> root-to-leaf
                       paths + a tree-distance matrix, which is what the loss needs
  data.py             CUB metadata parsing + the PyTorch Dataset (bbox-cropped images)
  model.py            ResNet-101 backbone + linear head (no Faster R-CNN -- see "Scope")
  losses.py           MomentumCache + msa_hicl_loss: the paper's Eq. 10-13, implemented
                       to match the equations, with the one documented deviation noted
                       in the module docstring (prototype initialisation)
  train.py            checkpointed training loop, used for both baseline and hicl runs
  evaluate.py         loads finished checkpoints, writes results/results.json
  tsne_figure.py       reproduces Fig. 6-style plot for 8 species / 4 genus groups
tests/
  test_hierarchy.py  pure-Python checks on the tree/LCA-distance math (no torch needed)
  test_losses.py      smoke test: msa_hicl_loss on a tiny synthetic tree is finite and
                       differentiable (this is what caught a NaN bug before any real GPU
                       time was spent on it)
reproduce.sh          the one documented command, see below
environment.yml       conda environment actually used
results/results.json  the numbers above, produced by an actual completed run
figures/              tsne_comparison.png, produced by the same run
checkpoints/          NOT committed directly (see "Checkpoints" below) -- available as a
                       GitHub Release, or regenerated from scratch by reproduce.sh
REPORT.md             the write-up: the gap, what was guessed, what wasn't resolved
```

## How to re-run it

```bash
git clone https://github.com/Manani10/FG-ZSD-Claim.git
cd FG-ZSD-Claim
conda env create -f environment.yml
conda activate fgzsd-hicl-repro
bash reproduce.sh
```

That single command downloads CUB-200-2011 and the taxonomy, trains the
baseline, trains the treatment, writes `results/results.json`, and writes
`figures/tsne_comparison.png` — in that order, each stage skipped if its
output already exists and is complete. It is allowed to take hours on its
first run (15 epochs × 2 runs on a free-tier GPU); that is expected of a
reproduction that actually trains, per the assignment brief.

To run a single piece on its own (e.g. during the viva):

```bash
python -m src.train --run baseline --epochs 15
python -m src.train --run hicl --epochs 15 --hicl-weight 0.1
python -m src.evaluate
python -m src.tsne_figure
python -m pytest tests/ -v
```

## Checkpoints

The brief says to commit a checkpoint and skip a stage when one is found,
to avoid re-paying for an expensive stage. We follow the skip-if-found
logic in `src/train.py` (see "Resumability" in that file's docstring), but
we do **not** commit the actual `.pt` files directly to the git tree: a
ResNet-101 state dict plus Adam's per-parameter optimizer state is well
over GitHub's 100MB per-file limit, and Git LFS was not set up for this
project. Instead, both trained checkpoints (`baseline.pt`, `hicl.pt`) are
attached to this repository's
[`checkpoints-v1` release](https://github.com/Manani10/FG-ZSD-Claim/releases/tag/checkpoints-v1),
so a reviewer can download the exact weights that produced the numbers
below without re-paying the ~15-epoch training cost. What *is* committed
directly to the git tree is the result of having already paid that cost
once (`results/results.json`, `figures/tsne_comparison.png`), so the
numbers in `REPORT.md` are checkable without re-training, even though
regenerating them from scratch via `reproduce.sh` still takes hours.

## Scope reductions from the paper (full list)

1. **No detection.** Faster R-CNN and the region-proposal machinery are
   dropped entirely; this is image classification on already-cropped
   birds (CUB's own bounding boxes), not detection.
2. **No zero-shot element.** Every class seen at test time was also seen
   in training. The paper's whole point is detecting *unseen* classes;
   this reproduction only tests whether the taxonomy-aware loss improves
   the feature space at all, on classes the model has training data for.
3. **No generator, no text encoder.** The paper's MSA generative network
   (word/sentence-level conditioning, Eq. 1-9) is entirely out of scope.
   One direct consequence: MSA-hicl's prototypes are supposed to be
   *initialised* from that generator's aligned text embeddings
   (Sec. 4.2); ours are initialised randomly and shaped only by the
   momentum update over real image features (see `src/losses.py`
   docstring). This is the single most important documented deviation —
   it's the part of the mechanism most likely to explain why the hardest
   genus pair (Blackbirds) never visibly separates in the t-SNE figure
   above.
4. **Different dataset host.** The official CUB-200-2011 host returns
   404/403 as of this writing; a GitHub mirror is used instead (see
   `src/download_data.py`). The data itself is the same release.
5. **A different, independently-sourced taxonomy.** CUB ships no
   taxonomy at all. The real bird taxonomy used here comes from
   `cvjena/semantic-embeddings` (tag v1.1.0; not from the FG-ZSD authors,
   who did not release theirs), so the exact tree structure — and
   therefore the tree distances the push term uses — may not match what
   the paper's own benchmark uses internally. It was cross-checked by
   hand against a known species (leaf node 1 = *Phoebastria nigripes* =
   CUB class 001) rather than assumed; `src/hierarchy.py`'s
   `_sanity_check_leaf_alignment` enforces this on every run.
6. **No hyperparameter search.** `hicl_weight = 0.1` was chosen once, by
   analogy, and never swept.

See `REPORT.md` for the full discussion, including the number the paper
never reports (the scaled-down seen-class CUB-200 result itself), and the
open questions that weren't resolved.
