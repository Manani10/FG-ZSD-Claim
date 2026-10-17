# Write-up draft — DRAFT, not yet submitted

*(~1,650 words. This is a first draft to be trimmed/expanded toward the
1,500–2,500 word target before submission; the numbers in it are real,
taken from `results/results.json`.)*

## 1. The claim, precisely (Comprehension)

Fine-Grained Zero-Shot Object Detection (FG-ZSD / MSHC, arXiv:2507.10358v1)
proposes a method for detecting object classes that were never seen in
training and that differ from seen classes only in fine local detail —
bird species that share pose, background and colour palette. The method
has two trained components: a generative network that synthesises images
of unseen classes from text descriptions (Sec. 3, Eq. 1–9), and a
detector whose classification head is shaped, during training, by a
hierarchical contrastive loss the paper calls MSA-hicl (Sec. 4.2,
Eq. 10–13). MSA-hicl keeps one prototype vector per node of the species'
biological taxonomy (order → family → genus → species), pulls each
region feature toward the prototypes on its own root-to-leaf path
(weighted by depth), and pushes it away from every other node's prototype
by an amount scaled by the real taxonomic distance between the nodes —
so a woodpecker is pushed harder away from a flamingo than one flamingo
species is pushed from another. The paper's headline evidence for this
(Fig. 6, Sec. 5.3's ablation) is that this loss produces test-set features
that visibly cluster by genus, and that removing it (ablating to plain
cross-entropy) both lowers detection accuracy and collapses that genus
structure.

The claim this repository tests is narrower: **does MSA-hicl, added to a
standard image classifier trained on ordinary (seen-class) data, produce
higher accuracy and visibly tighter genus clustering than cross-entropy
alone?** This drops detection, drops the zero-shot element entirely, and
drops the generator — see README.md "Scope reductions" for the full list
and the reasoning. What's left is the one piece of the method that is
testable with public data and a modest compute budget: whether shaping
the feature space by a real taxonomy, rather than treating all 200
classes as equally unrelated, helps at all.

## 2. What was measured, next to what the paper measures (Evidence)

| | Setting | Metric | Original paper | This reproduction |
|---|---|---|---|---|
| Baseline | detector, seen classes only, cross-entropy | mAP / recall (detection) | reported in Table 2, ablation row without hicl | **77.18%** top-1 classification accuracy |
| Treatment | + MSA-hicl | same | reported in Table 2, ablation row with hicl | **79.32%** top-1 classification accuracy |
| Difference | | | an improvement the paper reports but does not isolate on a classification-only, seen-class-only setting | **+2.14 points** |

These are not the same metric, the same task, or the same dataset split
as the paper's own ablation — the comparison above is a comparison of
*directions*, not magnitudes: does adding the loss help, in both cases?
Yes, in both. Matching the actual *size* of the improvement to the
paper's own ablation number is not meaningful here, because the paper's
ablation is measured on zero-shot detection mAP with the generator
present, and ours is measured on seen-class top-1 classification accuracy
with no generator and no detector at all — different units entirely.

**The number the paper does not report:** the paper never measures what
MSA-hicl does to a *plain* classifier's seen-class accuracy in isolation
from detection and from the generator. 77.18% → 79.32% on CUB-200 under a
scoped-down setting is exactly that missing number, and it is the
satisfying answer to the brief's requirement for "at least one number the
paper does not."

The qualitative claim (genus clustering) was checked with a t-SNE
projection of 8 species across 4 real-taxonomy genus pairs
(`figures/tsne_comparison.png`). It holds for 3 of the 4 pairs: Least
Auklet/Crested Auklet and the two Bunting species show visibly tighter,
more separated clusters under MSA-hicl than under plain cross-entropy. The
two Albatross species and the two Blackbird species were already cleanly
separated under plain cross-entropy, so there is little visible headroom
left for MSA-hicl to show an effect on those two pairs — this reproduction
cannot distinguish "the loss doesn't help here" from "there was nothing
left to fix here," and that ambiguity is left unresolved rather than
argued away.

## 3. Where this did not reproduce cleanly, and what was guessed (Honesty)

This section is the one the brief weights most (35%), and it is the one
worth being specific in rather than general.

**Prototype initialisation (the biggest gap).** MSA-hicl's prototypes are
specified in the paper as being initialised from the same aligned
text/image embeddings the generator produces (Sec. 4.2). This
reproduction has no generator and no text encoder at all — that entire
branch of the method was ruled out of scope before any code was written,
because building it would require either the paper's unreleased
FGZSD-Birds captions or a from-scratch text-encoder training run, neither
of which fits a course-sized budget. So the prototypes here start as
random unit vectors and are shaped purely by the momentum-updated average
of real image features over training (`src/losses.py`). It is plausible
that text-initialised prototypes would converge faster, or to a
better-separated arrangement, than randomly-initialised ones — this
reproduction cannot tell, because it never tried the text-initialised
version. If the +2.14 point improvement is smaller than what a
text-initialised version would achieve, this is the most likely reason
why, and that is a real limitation of this reproduction, not a footnote.

**A guessed taxonomy.** CUB-200-2011 ships no taxonomy of its own — only
200 flat class folders. The paper's benchmark presumably has its own
taxonomy for its larger, unreleased FGZSD-Birds dataset, but that
taxonomy was never released either. The real bird taxonomy used here is
pulled from a third-party academic release, `cvjena/semantic-embeddings`
(tag v1.1.0), which maps CUB's own 200 classes onto a real biological
tree (order/family/genus/species, 383 total nodes, root-to-leaf paths of
length 4–7, average 6.5). Using this instead of the paper's own (unseen)
taxonomy means the exact tree distances the push term scales by are not
necessarily identical to whatever the paper's benchmark uses internally,
even though both describe the same underlying biology. This was
cross-checked once, by hand, against two known species (leaf node 1 =
*Phoebastria nigripes* = CUB class 001, Black-footed Albatross), rather
than assumed — `src/hierarchy.py`'s `_sanity_check_leaf_alignment`
documents and enforces that check rather than trusting the file silently.

**No hyperparameter search.** `hicl_weight = 0.1` was chosen once, by
analogy to similar auxiliary-loss weightings in other contrastive
classification work, and was never swept. Whether 0.1 is close to optimal
or whether a different weight would widen or shrink the +2.14 point gap
is unknown; this reproduction reports one configuration, not a sweep, and
says so rather than implying the result is weight-insensitive.

**An unoptimised loss implementation, and unrecorded training time.** The
loss as written in `src/losses.py` loops over each sample in the batch
and recomputes its similarity against all ~383 taxonomy-node prototypes
individually — this matches the equations directly and was easy to
verify correct (a finite, non-NaN smoke test loss, `tests/test_losses.py`)
but it is not vectorised and is noticeably slower per step than the
cross-entropy-only baseline. Wall-clock training time per epoch was not
recorded during the actual run that produced these results (training ran
across several Google Colab sessions, with per-epoch checkpointing to
Google Drive to survive the free tier's session limits — see
`src/train.py`'s "Resumability" docstring — and the elapsed-time logging
that would have captured this was added after the fact, too late to
capture the original numbers). Anyone re-running `reproduce.sh` from a
clean checkout should expect the hicl run to take noticeably longer per
epoch than the baseline, and should record their own timing if that
matters for their purposes.

**Downloading the dataset at all required a workaround.** The canonical
CUB-200-2011 URL (`vision.caltech.edu/.../CUB_200_2011.tgz`) returns
HTTP 404 as of this writing — the dataset has moved off that host. The
current official host, CaltechDATA, returns HTTP 403 Forbidden to a
scripted request (it appears to require a browser session). A public
GitHub mirror was used instead (`src/download_data.py`); the data itself
is the same release, but it is worth stating plainly that the "official"
source for this widely-used benchmark is not reliably reachable from a
plain script in 2025/2026, which is itself a small, unplanned-for
reproducibility finding.

## 4. What would change this conclusion

Two experiments would most directly test whether the +2.14 point result
is robust rather than a one-seed artifact: (a) repeating both runs with a
second random seed, since nothing here has been checked against
seed-to-seed variance; and (b) a small sweep over `hicl_weight` (e.g.
{0.01, 0.1, 0.5, 1.0}) to see whether 0.1 happens to sit near a local
optimum or whether the effect is stable across a wide range. Neither was
run, for the same reason the paper's own full method wasn't reproduced:
compute budget. Both are named here as the first things to try next,
rather than left implicit.
