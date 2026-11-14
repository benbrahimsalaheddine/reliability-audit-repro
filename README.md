# Reliability Is Not Fidelity — reproduction code

Supplementary code for the submission *Reliability Is Not Fidelity: Auditing the
Human References of LLM Judges*. The scripts here recompute the quantities the
paper reports: per-panel reliabilities and fidelity, the panel-specific-variance
test and its calibration, the certificate–fidelity inversions, the ceilings, and
the nine-judge comparison.

This archive contains only code and the small derived outputs the paper
produced. It does **not** redistribute any third-party dataset. Each dataset is
public; the paths and preprocessing needed to place it are given below.

## Requirements

```
pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `pandas`, `scipy`. Python 3.10 or later.

## Shared constants

All standardized reruns use the same constants, defined once in
`std/repro_config.py`:

- `SEED = 0` — every result starts its own `numpy.random.default_rng(0)`.
- `N = 5000` — replications for rate/power simulations.
- `B = 10000` — bootstrap draws.

Point estimates are deterministic under a fixed seed; interval widths depend on
`B`.

## Datasets (not included — place them under `./data/`)

The scripts read datasets from `./data/`. Download each public release and
arrange it as shown. No raw dataset is shipped in this archive.

- **SummEval** (Fabbri et al., 2021): the aligned model-annotation file
  `model_annotations.aligned.jsonl` → `./data/summeval/model_annotations.aligned.jsonl`.
- **D3Code** (Davani et al., 2024): the rater-level release with
  `d3-ratings.csv`, `d3-raters.csv`, `d3-items.csv` → `./data/d3repo/dataset/`.
- **CREHate** (Lee et al., 2024): `CREHate_raw_annotations.csv` (and the other
  released annotation files) → `./data/crehate/data/`.
- **G-Eval** judge scores (Liu et al., 2023): the released per-item scores →
  `./data/geval/results/`.

Preprocessing is the datasets' own released rater-level format; the scripts do
their own filtering (for SummEval, the 793 rows that survive de-duplication —
see `std/S16_items_kept_std.py`). No manual relabeling is applied.

## Repository layout and script → result map

### `std/` — standardized reruns (seed 0, N, B from `repro_config.py`)

| Script | Produces | Output file |
| --- | --- | --- |
| `table2_summeval_std.py` | SummEval per-dimension reliabilities (rho_1, rho_k), fidelity, tau^2 (main SummEval table) | `RES_table2_summeval_std.txt`, `RESULTS_std_table2_summeval_B10000_v01.txt` |
| `irep_std.py` | IRep reliabilities across seeds | `RESULTS_irep_B10000_seeds0to3_v01.txt` |
| `W1_false_positive_std.py`, `inc_W1.py` | false-positive rate of the per-panel test | `RES_W1_false_positive_std.txt`, `RES_W1_std.txt` |
| `calib_std.py`, `inc_calib.py` | item vs nested-document calibration (FPR at corr 0 / 0.3) | `RES_calib_std_0.txt`, `RES_calib_std_0.3.txt`, `RES_calib_std_N5000_*.txt` |
| `robust_sim_std.py`, `inc_robust.py` | Size and power of the test under the reported setting | `RES_robust_sim_std.txt` |
| `care_summeval.py`, `care_3judges.py` | CARE baseline comparison | `RES_care_summeval.txt`, `RES_care_3judges.txt` |
| `d3code_recount_std.py` | D3Code certificate–fidelity inversions (recount) | `RES_d3code_recount_std.txt` |
| `jigsaw_d3pair_std.py` | Jigsaw / D3 panel pairing | `RES_jigsaw_d3pair_std.txt` |
| `nine_judges_traits_halo.py` | Nine-judge expert/crowd gap, with traits and halo variants | `RES_nine_judges_traits_halo.txt` |
| `summeval_traits.py` | SummEval traits/halo decomposition | `RES_summeval_traits_halo.txt` |
| `tau2_judges_experts.py` | tau^2 of judges vs experts | `RES_tau2_judges_experts.txt` |
| `S16_items_kept_std.py` | Clean-row bookkeeping for SummEval | `RES_S16_items_kept_std.txt` |
| `flesch_std.py` | Flesch readability control (style/verbosity) | — |
| `geval_gap_ci.py`, `geval_gap_ci_clean.py` | G-Eval ceiling-gap confidence intervals | — |
| `repro_config.py` | Shared constants | — |
| `go.sh` | Resumable launcher for the long bootstraps | writes to `ckpt/` |

`ckpt/` holds small bootstrap checkpoints (`W1.jsonl`, `calib_0.0.jsonl`,
`calib_0.3.jsonl`, `robust.jsonl`) so the tables regenerate without rerunning
every draw.

### `judges/` — judges treated as panels

| Script | Produces | Output file |
| --- | --- | --- |
| `judges_as_panels_v121.py` | The audit applied with the judges themselves as panels | `RESULTS_judges_v121.txt` |

`RESULTS_sim_general_10000.txt` is the general two-panel simulation at B = 10000.

### `tbl/` — table builders

| Script | Produces | Output file |
| --- | --- | --- |
| `tables_panels_v121.py` | Per-panel and per-pair summary tables | `table_panels_v121.csv`, `table_pairs_v121.csv`, `RESULTS_tables_v121.txt` |

### `w1/` — false-positive rate (independent rerun)

| Script | Produces | Output file |
| --- | --- | --- |
| `assumption_W1_false_positive_v121.py`, `run.py` | false-positive rate (independent rerun) | `RESULTS_rerun.txt` |

### `irepseed/` — IRep seed sensitivity

| Script | Produces | Output file |
| --- | --- | --- |
| `irep_param.py`, `flags_param.py`, `variants.py` | IRep reliabilities and flags across seeds | `b10000_0..3.txt`, `seeds500.txt`, `seeds3000.txt`, `flags_seeds_B1000.txt` |

### Root scripts

| Script | Produces |
| --- | --- |
| `sim_general_case_v113.py` | Shared simulation module (imported by the calibration scripts) |
| `nested_sim_v114.py` | Nested-document calibration simulation |
| `seeds_calib_v121.py` | Seed sensitivity of the calibration and the bounds |
| `d3code_bonferroni_v113.py` | D3Code inversions surviving a Bonferroni correction |
| `d3code_equal_significant_v113.py` | D3Code inversion counts (two-sided / one-sided) |
| `assumption_geval_recompute_v121.py` | G-Eval recomputation (pooled correlations) |
| `tetrad_calib_d3_v114.py`, `tetrad_d3code_v114.py` | Independence / tetrad checks on D3Code |

## How to run

From the archive root, with the datasets placed under `./data/`. Most scripts
take the dataset file as their first argument; the top of each file shows the
exact arguments it expects. For example:

```
# the SummEval table (pass the aligned annotation file):
python3 std/table2_summeval_std.py ./data/summeval/model_annotations.aligned.jsonl

# the long bootstraps (resumable launcher):
bash std/go.sh        # writes checkpoints to std/ckpt/

# panel / pair tables:
python3 tbl/tables_panels_v121.py
```

Each `std/` script prints to stdout and writes its `RES_*.txt`. The `RES_*.txt`
and `RESULTS_*.txt` files already in this archive are the outputs from our own
run; a reader can compare against them. If a script is launched without its
argument it exits with an error rather than using a hidden default, so the
dataset path is always explicit.

## Scope and honesty note

The scripts recompute the numbers the paper reports, under seed 0 and the stated
`N` and `B`. Interval widths move slightly with `B`; point estimates do not. A
few results depend on a dataset that is public but not redistributed here, so a
reader must download it first as described above. We do not claim a single
command reproduces every figure and table end to end; the map above says which
script produces which quantity.
