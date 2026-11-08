# Realizable Adversarial Robustness of NetFlow Intrusion Detectors

Code and result summaries for the paper
*Realizable Adversarial Robustness of NetFlow Intrusion Detectors: Constraints, Leakage, and Operating Points*
(Nanlei Chen, Xiaoyu Ke, Yifan Hui, Xuyan Lu, Huangze Fang, Qiulong Lin; under submission).

The repository evaluates flow-based intrusion detectors (Random Forest, XGBoost, MLP) on three NetFlow v3 datasets under
attacks that an attacker can actually realize, and compares augmentation-based defenses.

## Data

The datasets are **not** redistributed here. Download the NetFlow v3 versions of
NF-UNSW-NB15-v3, NF-CSE-CIC-IDS2018-v3 and NF-ToN-IoT-v3 from the University of Queensland
(see Luay et al., arXiv:2503.04404) and place the original zip files in `01_数据/raw/`.

## Pipeline

Run all scripts from the repository root, in this order. Each script documents its inputs and outputs in its header.

| Step | Script | Purpose |
|---|---|---|
| 1 | `04_代码/01_check_constraints.py` | Verify candidate protocol constraints on all flows |
| 2 | `04_代码/03_prepare_data.py` | Sample ~1.5 M flows per dataset, drop identifiers |
| 3 | `04_代码/12_formal_train.py <dataset> --seed s [--variant RAW|noTTL]` | Train RF / XGBoost / MLP per seed |
| 4 | `04_代码/08_projection_check.py` | Unit tests of the projection operator on real flows |
| 5 | `04_代码/11_adaptive_search.py <dataset> --seed s --variant RAW` | Action-space attacks, four operating points |
| 6 | `04_代码/09_constrained_attack.py <dataset> --seed s` | Unconstrained vs. repaired vs. action-space attacks |
| 7 | `04_代码/10_ablation.py actions <dataset> --seed s` | Action ablation (TTL / window / flags) |
| 8 | `04_代码/14_adv_training.py <dataset> --seed s` | Defenses D0, D1, D1′, D2, D3 |
| 9 | `04_代码/13_aggregate.py`, `04_代码/15_aggregate_at.py` | Mean ± std over seeds |

Shared modules: `nfv3_constraints.py` (constraint checks, splits) and `nfv3_projection.py` (attacker actions and projection).
Seeds 0, 1, 2 control the split, training subsample, model initialisation and attack sampling.

## Results

`03_结果/` contains the per-seed and aggregated CSV files behind every table and figure in the paper.

## Environment

CPU only. See `requirements.txt` for package versions.

## License

MIT (code). The datasets remain under their original licenses.
