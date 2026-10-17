# Refresh2Track three ablations

Status: RUNNING. Only completed full 21-scene / 3316-frame seed-zero evaluations appear as results. Pending branches have no final scores.

| Method | Status | Input dimensions (effective) | Feature BPS | Serialized BPS | Optional UPDATE % | mAP | AMOTA | MOTA | IDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4 Full (reference) | COMPLETE | 2324 | 16501.531966224367 | 27258.582629674307 | 5.851959225661648 | 0.36513092109991985 | 0.37533100220281657 | 0.301300270001421 | 249.0 |
| w/o Explicit Delta | RUNNING | 1556 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| w/o Receiver-State Conditioning | RUNNING | 787 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| w/o Communication Loss | COMPLETE | 2324 | 109791.03739445114 | 120548.08805790107 | 98.57617132703638 | 0.3621972811723704 | 0.3602163290347968 | 0.29074889867841414 | 304.0 |

Feature payload uses nominal 10 Hz × mean(52N + 780U); serialized BPS uses actual packet length. Optional UPDATE includes optional decisions only. MOTA/IDS/FP/FN use the inherited evaluator operating point.

Official validation participates in checkpoint selection, early stopping and reporting; these scores do not constitute an independent test. Only seed 0 is used, without claims about cross-seed robustness or statistical significance.

Explicit delta is computable from current and last received features: its ablation measures the value of an explicit representation, rather than new mathematical information. Receiver-state comparisons require joint assessment of communication and accuracy. Communication-loss outcomes, including extreme UPDATE/REUSE ratios, are recorded without threshold or regularization changes.

Frozen V4 reference checkpoint SHA256: `0a15216b74b3fc2ebce0570e8aae436c156421b9c7119ec45755527d6cfa4a5f`. Source result: `/mnt/d/CoopTrack_Audit/experiments/v4_ablations/reference_v4_full/final_validation.json`.

Evidence and branch paths are recorded in `run_manifest.json`; the source audit is `v4_preflight.md` and the input checks are `input_mask_tests.json`.
