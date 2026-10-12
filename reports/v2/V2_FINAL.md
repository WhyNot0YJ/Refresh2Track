# V2 formal experiment

**V2_FORMAL_EXPERIMENT_COMPLETE**


Seed 0 shows low-rate numerical improvements, with mixed higher-rate results. Transformer has no consistent cost-aware advantage over MLP. Cross-seed robustness is untested.

## Reviewed interpretation

The frozen source, model, configuration, manifest and raw measurements remain unchanged. The original generator assigned verdict labels without evaluating deltas; those original labels and its report-stage RUNNING snapshot are preserved in final_summary.json under post_run_interpretation.generator_labels_before_review.

- MLP B1/B2 improve both mAP and AMOTA against the nearest measured-byte Periodic references, with 1.50%/2.58% lower actual cost. MLP B2 is nevertheless dominated in cost+mAP+AMOTA by P12 and MLP B1; nearest-pair improvement is not a global frontier claim.
- Transformer B1 improves AMOTA and reduces mAP versus nearest P12 at 2.82% higher cost; it improves both metrics versus upper P10 at 2.80% lower cost. MLP B1 has lower cost and higher mAP/AMOTA than Transformer B1.
- Transformer B2 improves both metrics versus both adjacent brackets. It uses 8.45% fewer bytes than historical P4, and 12.49% more than P6; neither pair is an equal-rate match within 5%.
- Transformer B3 has lower mAP than nearest P2 and lower AMOTA than lower-bracket P4. Transformer B4 has lower mAP than both P2 and P1 while AMOTA is higher. All unfavorable comparisons and rate gaps remain in the tables.
- At the same nominal B, Transformer uses 12.41%,30.02%,44.41%,50.34% more actual bytes than MLP. B2 alone improves both metrics; B1/B3/B4 worsen both. A consistent equal-cost architectural advantage is not established.
- In the complete seed-0 cost+mAP+AMOTA table, MLP B1 and Transformer B2 add numerical frontier points. MLP B1 has the highest AMOTA; Transformer B2 has the highest mAP. These observations do not establish cross-budget or cross-seed superiority.
- All eight adaptive mean costs are below their nominal budgets. This is a mean constraint, not a hard per-frame cap.
- Historical P1 retains its original serializer and measured 12161.645355850422 B/frame. Historical P1/P2/P4 use inherited inference RNG placement, so their reused comparisons remain descriptive.
- No cross-seed, replicated-improvement or significance claim is supported.

46 official train scenes / 7362 samples; fixed 12 epochs per scheduler; epoch-12 final checkpoints. Official val: 21 scenes / 3316 samples.
The fixed primary seed is 0. All costs are actual serialized bytes; no validation checkpoint selection or model adjustment is applied.

| Method | Budget | Evidence | Actual B/frame | P95 B/frame | B/s (10Hz) | mAP | AMOTA | AMOTP | MOTA | IDS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | N/A | inherited_official_val_reference | 12161.645356 | 21043.000000 | 121616.453559 | 0.361107 | 0.353343 | 1.323493 | 0.287303 | 316.000000 |
| P2 | N/A | inherited_official_val_reference | 7100.773824 | 14782.000000 | 71007.738239 | 0.362819 | 0.352596 | 1.323629 | 0.287303 | 366.000000 |
| P4 | N/A | inherited_official_val_reference | 4559.422799 | 10330.000000 | 45594.227986 | 0.360797 | 0.354025 | 1.325711 | 0.283004 | 372.000000 |
| INIT_ONLY | N/A | new_formal_v2 | 2137.094692 | 3801.000000 | 21370.946924 | 0.351785 | 0.331364 | 1.333205 | 0.266591 | 296.000000 |
| P10 | N/A | new_formal_v2 | 3042.468637 | 6261.500000 | 30424.686369 | 0.361066 | 0.357876 | 1.320195 | 0.288511 | 279.000000 |
| P12 | N/A | new_formal_v2 | 2876.165862 | 5676.250000 | 28761.658625 | 0.362203 | 0.361646 | 1.323265 | 0.287942 | 277.000000 |
| P16 | N/A | new_formal_v2 | 2670.816043 | 5220.500000 | 26708.160434 | 0.360143 | 0.360109 | 1.324434 | 0.292348 | 280.000000 |
| P24 | N/A | new_formal_v2 | 2463.819662 | 4595.750000 | 24638.196622 | 0.359570 | 0.357263 | 1.323016 | 0.291495 | 297.000000 |
| P32 | N/A | new_formal_v2 | 2362.438480 | 4238.000000 | 23624.384801 | 0.359524 | 0.357565 | 1.323500 | 0.285207 | 256.000000 |
| P6 | N/A | new_formal_v2 | 3710.737636 | 8420.000000 | 37107.376357 | 0.363033 | 0.361545 | 1.320432 | 0.284994 | 274.000000 |
| P8 | N/A | new_formal_v2 | 3295.333534 | 7335.000000 | 32953.335344 | 0.360608 | 0.357244 | 1.322270 | 0.286450 | 277.000000 |
| mlp | B1 | new_formal_v2 | 2630.828106 | 4825.000000 | 26308.281062 | 0.362306 | 0.369316 | 1.322707 | 0.298956 | 274.000000 |
| mlp | B2 | new_formal_v2 | 3210.417973 | 6200.000000 | 32104.179735 | 0.361867 | 0.358024 | 1.321121 | 0.286663 | 263.000000 |
| mlp | B3 | new_formal_v2 | 4065.218938 | 7776.750000 | 40652.189385 | 0.362019 | 0.363621 | 1.319018 | 0.287267 | 283.000000 |
| mlp | B4 | new_formal_v2 | 5029.163450 | 9576.000000 | 50291.634499 | 0.362941 | 0.360350 | 1.320765 | 0.291246 | 320.000000 |
| transformer | B1 | new_formal_v2 | 2957.317853 | 4374.000000 | 29573.178528 | 0.361828 | 0.366951 | 1.323104 | 0.291140 | 229.000000 |
| transformer | B2 | new_formal_v2 | 4174.127262 | 6058.000000 | 41741.272618 | 0.363251 | 0.364243 | 1.321886 | 0.285953 | 284.000000 |
| transformer | B3 | new_formal_v2 | 5870.556695 | 8675.250000 | 58705.566948 | 0.361564 | 0.353208 | 1.321840 | 0.284958 | 317.000000 |
| transformer | B4 | new_formal_v2 | 7560.870326 | 11310.500000 | 75608.703257 | 0.360639 | 0.357989 | 1.322362 | 0.290358 | 320.000000 |

Every adaptive budget is compared with the nearest actual-byte Periodic point and both adjacent brackets. Missing brackets are N/A; every comparison is retained.

| Adaptive | Budget | Role | Periodic | Adaptive bytes | Periodic bytes | Relative gap | ΔmAP | ΔAMOTA | Rate match |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| mlp | B1 | nearest | P16 | 2630.828106 | 2670.816043 | 0.014972 | 0.002164 | 0.009207 | True |
| mlp | B1 | lower_bracket | P24 | 2630.828106 | 2463.819662 | 0.067784 | 0.002736 | 0.012053 | False |
| mlp | B1 | upper_bracket | P16 | 2630.828106 | 2670.816043 | 0.014972 | 0.002164 | 0.009207 | True |
| mlp | B2 | nearest | P8 | 3210.417973 | 3295.333534 | 0.025768 | 0.001260 | 0.000780 | True |
| mlp | B2 | lower_bracket | P10 | 3210.417973 | 3042.468637 | 0.055202 | 0.000802 | 0.000148 | False |
| mlp | B2 | upper_bracket | P8 | 3210.417973 | 3295.333534 | 0.025768 | 0.001260 | 0.000780 | True |
| mlp | B3 | nearest | P6 | 4065.218938 | 3710.737636 | 0.095529 | -0.001014 | 0.002075 | False |
| mlp | B3 | lower_bracket | P6 | 4065.218938 | 3710.737636 | 0.095529 | -0.001014 | 0.002075 | False |
| mlp | B3 | upper_bracket | P4 | 4065.218938 | 4559.422799 | 0.108392 | 0.001222 | 0.009596 | False |
| mlp | B4 | nearest | P4 | 5029.163450 | 4559.422799 | 0.103026 | 0.002144 | 0.006325 | False |
| mlp | B4 | lower_bracket | P4 | 5029.163450 | 4559.422799 | 0.103026 | 0.002144 | 0.006325 | False |
| mlp | B4 | upper_bracket | P2 | 5029.163450 | 7100.773824 | 0.291744 | 0.000121 | 0.007755 | False |
| transformer | B1 | nearest | P12 | 2957.317853 | 2876.165862 | 0.028215 | -0.000375 | 0.005305 | True |
| transformer | B1 | lower_bracket | P12 | 2957.317853 | 2876.165862 | 0.028215 | -0.000375 | 0.005305 | True |
| transformer | B1 | upper_bracket | P10 | 2957.317853 | 3042.468637 | 0.027987 | 0.000762 | 0.009075 | True |
| transformer | B2 | nearest | P4 | 4174.127262 | 4559.422799 | 0.084505 | 0.002454 | 0.010218 | False |
| transformer | B2 | lower_bracket | P6 | 4174.127262 | 3710.737636 | 0.124878 | 0.000218 | 0.002698 | False |
| transformer | B2 | upper_bracket | P4 | 4174.127262 | 4559.422799 | 0.084505 | 0.002454 | 0.010218 | False |
| transformer | B3 | nearest | P2 | 5870.556695 | 7100.773824 | 0.173251 | -0.001255 | 0.000613 | False |
| transformer | B3 | lower_bracket | P4 | 5870.556695 | 4559.422799 | 0.287566 | 0.000768 | -0.000817 | False |
| transformer | B3 | upper_bracket | P2 | 5870.556695 | 7100.773824 | 0.173251 | -0.001255 | 0.000613 | False |
| transformer | B4 | nearest | P2 | 7560.870326 | 7100.773824 | 0.064795 | -0.002180 | 0.005394 | False |
| transformer | B4 | lower_bracket | P2 | 7560.870326 | 7100.773824 | 0.064795 | -0.002180 | 0.005394 | False |
| transformer | B4 | upper_bracket | P1 | 7560.870326 | 12161.645356 | 0.378302 | -0.000468 | 0.004646 | False |

Full communication accounting, staleness summaries, six buckets and ten-bin optional p(update) histograms appear below; full machine-readable evidence is linked at the end.
Historical P1 retains its original measured 12161.645355850422 B/frame. Its serializer differs from new V2, and inherited P1/P2/P4 RNG placement differs.

Seed 0 cannot establish statistical robustness. Favorable single-seed observations remain descriptive; unfavorable observations remain in the same tables.

## Transformer versus MLP at every nominal budget

Same nominal B is not the same actual communication cost. Deltas below are Transformer minus MLP; all four actual-cost differences exceed 5%.

| Budget | Transformer B/frame | MLP B/frame | Cost increase | delta mAP | delta AMOTA |
| --- | --- | --- | --- | --- | --- |
| B1 | 2957.317853 | 2630.828106 | 12.41% | -0.000479 | -0.002364 |
| B2 | 4174.127262 | 3210.417973 | 30.02% | 0.001383 | 0.006219 |
| B3 | 5870.556695 | 4065.218938 | 44.41% | -0.000455 | -0.010413 |
| B4 | 7560.870326 | 5029.163450 | 50.34% | -0.002301 | -0.002361 |

## Training and final protocol verification

Each model completed 88,344 sample visits across 12 full epochs; every scene received each budget three times, with 22,086 samples per budget. Native base parameters and BN remained frozen; FP32, fresh seed 0, true batch 4 with legal tails, AdamW peak 1e-4 / decay 1e-4 / clip 5, one warmup epoch and fixed 12-epoch cosine horizon to 1e-6.

MLP performed 23,439 optimizer steps: 23,388 nonzero-gradient and 51 finite zero-gradient batches. Transformer performed 23,439: 23,394 and 45. Each skipped 14 cold/scene-tail batches. Independent nonzero original-task gradient checks passed; zero-gradient batches remain disclosed.

Gate threshold 0.5, forced UPDATE only for missing receiver cache, no TTL/age cap/Top-K fill. No internal dev, validation during training, checkpoint selection or early stopping.

Final manifest SHA256: dd753608c90b75f8b7a316bb6207403c6d27bbd2f0eebb2a9878760c69968923. Frozen at 2026-10-08T05:58:36.290849+00:00, 20.285 seconds before the first new official-val run.

| Model | Final checkpoint | SHA256 |
| --- | --- | --- |
| mlp | [final.pth](../../experiments/v2/checkpoints/mlp/final.pth) | d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c |
| transformer | [final.pth](../../experiments/v2/checkpoints/transformer/final.pth) | 58bda699962f479a7556e75c6ad3b0389d6a909ca7b798e5eece2ea7f2fc65a3 |

### All training epochs

| Model | Epoch | Frames | Scenes | Native batch counts 4/3/2/1 | Task loss | Rate loss | Actual B/frame | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 1 | 7362 | 46 | 1685/107/76/149 | 1.512405 | 0.016396 | 2654.208775 | 3212.779466 |
| mlp | 2 | 7362 | 46 | 1748/56/83/36 | 1.418720 | 0.002670 | 4583.124830 | 3216.364850 |
| mlp | 3 | 7362 | 46 | 1672/89/70/267 | 1.411943 | 0.003066 | 4729.652948 | 3348.895002 |
| mlp | 4 | 7362 | 46 | 1720/2/190/96 | 1.394346 | 0.001581 | 4185.602418 | 3307.979041 |
| mlp | 5 | 7362 | 46 | 1767/11/117/27 | 1.400948 | 0.001914 | 4142.269084 | 3216.711898 |
| mlp | 6 | 7362 | 46 | 1800/12/23/80 | 1.378887 | 0.001255 | 3969.571448 | 3125.007406 |
| mlp | 7 | 7362 | 46 | 1775/36/42/70 | 1.388432 | 0.001402 | 3756.824912 | 3099.574796 |
| mlp | 8 | 7362 | 46 | 1768/77/6/47 | 1.370187 | 0.000887 | 3643.670877 | 3147.342466 |
| mlp | 9 | 7362 | 46 | 1766/1/124/47 | 1.385706 | 0.001805 | 3819.970796 | 3163.631673 |
| mlp | 10 | 7362 | 46 | 1768/22/54/116 | 1.373632 | 0.001268 | 3398.503803 | 3080.323213 |
| mlp | 11 | 7362 | 46 | 1781/42/12/88 | 1.383631 | 0.001037 | 3316.075115 | 3083.728484 |
| mlp | 12 | 7362 | 46 | 1762/19/110/37 | 1.367971 | 0.000910 | 3730.867291 | 3191.476949 |
| transformer | 1 | 7362 | 46 | 1685/107/76/149 | 1.449399 | 0.005758 | 4264.005026 | 3447.400308 |
| transformer | 2 | 7362 | 46 | 1748/56/83/36 | 1.376132 | 0.000807 | 5627.680657 | 3544.788256 |
| transformer | 3 | 7362 | 46 | 1672/89/70/267 | 1.382720 | 0.001183 | 5559.131350 | 3429.399347 |
| transformer | 4 | 7362 | 46 | 1720/2/190/96 | 1.369813 | 0.000662 | 5673.238930 | 3384.927587 |
| transformer | 5 | 7362 | 46 | 1767/11/117/27 | 1.386946 | 0.000961 | 5559.767047 | 3380.951533 |
| transformer | 6 | 7362 | 46 | 1800/12/23/80 | 1.375936 | 0.000565 | 6097.990356 | 3380.680921 |
| transformer | 7 | 7362 | 46 | 1775/36/42/70 | 1.377864 | 0.000662 | 5701.315539 | 3404.047000 |
| transformer | 8 | 7362 | 46 | 1768/77/6/47 | 1.365568 | 0.000416 | 5973.605678 | 3380.356833 |
| transformer | 9 | 7362 | 46 | 1766/1/124/47 | 1.375253 | 0.000854 | 5664.127275 | 3415.760992 |
| transformer | 10 | 7362 | 46 | 1768/22/54/116 | 1.363714 | 0.000449 | 6072.880331 | 3402.610672 |
| transformer | 11 | 7362 | 46 | 1781/42/12/88 | 1.379381 | 0.000589 | 5776.115865 | 3382.289680 |
| transformer | 12 | 7362 | 46 | 1762/19/110/37 | 1.360398 | 0.000549 | 5498.316354 | 3347.177621 |

## Communication and staleness diagnostics

Each new run has 44,329 query records, 1,557 forced initializations and 42,772 optional queries. Ratios are query-weighted. Latent bytes include init and optional UPDATE. Ages are measured before the decision; no-cache initialization is excluded. Unrecorded historical P1/P2/P4 fields remain N/A.

| Run | Mandatory B/frame | Latent B/frame | Hard UPDATE | Forced init | Optional UPDATE | Age mean | Age median | Age p95 | Age max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| P2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| P4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| INIT_ONLY | 1770.852232 | 366.242461 | 0.035124 | 0.035124 | 0.000000 | 52.061769 | 36.000000 | 157.000000 | 248.000000 |
| P10 | 1770.852232 | 1271.616405 | 0.121952 | 0.035124 | 0.089989 | 5.295895 | 5.000000 | 10.000000 | 10.000000 |
| P12 | 1770.852232 | 1105.313631 | 0.106003 | 0.035124 | 0.073459 | 6.204386 | 6.000000 | 12.000000 | 12.000000 |
| P16 | 1770.852232 | 899.963812 | 0.086309 | 0.035124 | 0.053049 | 8.006780 | 8.000000 | 16.000000 | 16.000000 |
| P24 | 1770.852232 | 692.967431 | 0.066458 | 0.035124 | 0.032475 | 11.393295 | 11.000000 | 23.000000 | 24.000000 |
| P32 | 1770.852232 | 591.586248 | 0.056735 | 0.035124 | 0.022398 | 14.645890 | 14.000000 | 30.000000 | 32.000000 |
| P6 | 1770.852232 | 1939.885404 | 0.186041 | 0.035124 | 0.156411 | 3.424764 | 3.000000 | 6.000000 | 6.000000 |
| P8 | 1770.852232 | 1524.481303 | 0.146202 | 0.035124 | 0.115122 | 4.367764 | 4.000000 | 8.000000 | 8.000000 |
| mlp B1 | 1770.852232 | 859.975875 | 0.082474 | 0.035124 | 0.049074 | 25.153629 | 12.000000 | 108.000000 | 248.000000 |
| mlp B2 | 1770.852232 | 1439.565742 | 0.138059 | 0.035124 | 0.106682 | 13.734078 | 7.000000 | 51.000000 | 138.000000 |
| mlp B3 | 1770.852232 | 2294.366707 | 0.220037 | 0.035124 | 0.191644 | 9.275087 | 5.000000 | 33.000000 | 112.000000 |
| mlp B4 | 1770.852232 | 3258.311218 | 0.312482 | 0.035124 | 0.287454 | 6.859067 | 4.000000 | 25.000000 | 82.000000 |
| transformer B1 | 1770.852232 | 1186.465621 | 0.113786 | 0.035124 | 0.081525 | 23.065417 | 12.000000 | 96.000000 | 195.000000 |
| transformer B2 | 1770.852232 | 2403.275030 | 0.230481 | 0.035124 | 0.202469 | 10.952983 | 6.000000 | 39.000000 | 129.000000 |
| transformer B3 | 1770.852232 | 4099.704463 | 0.393174 | 0.035124 | 0.371084 | 5.411835 | 3.000000 | 19.000000 | 71.000000 |
| transformer B4 | 1770.852232 | 5790.018094 | 0.555280 | 0.035124 | 0.539091 | 3.269522 | 1.000000 | 12.000000 | 46.000000 |

### Ten-bin optional p(update) histogram

Counts in [0.0,0.1), ..., [0.9,1.0]; forced init excluded. Each row sums to 42,772. Periodic policies have no learned probability.

| Run | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp B1 | 24951 | 5200 | 3449 | 2938 | 4135 | 1966 | 133 | 0 | 0 | 0 |
| mlp B2 | 15427 | 5002 | 3931 | 4242 | 9607 | 4348 | 215 | 0 | 0 | 0 |
| mlp B3 | 5900 | 3536 | 3460 | 4861 | 16818 | 7873 | 322 | 2 | 0 | 0 |
| mlp B4 | 2206 | 1467 | 2311 | 4038 | 20455 | 11719 | 564 | 12 | 0 | 0 |
| transformer B1 | 23584 | 3084 | 2696 | 3375 | 6546 | 3458 | 28 | 1 | 0 | 0 |
| transformer B2 | 12387 | 2623 | 2729 | 4467 | 11906 | 8517 | 142 | 1 | 0 | 0 |
| transformer B3 | 4254 | 1586 | 1850 | 3584 | 15626 | 15585 | 287 | 0 | 0 | 0 |
| transformer B4 | 1105 | 636 | 975 | 2203 | 14795 | 22649 | 407 | 2 | 0 | 0 |

### All six age buckets for every adaptive budget

All optional queries are retained. Count, UPDATE count, UPDATE ratio and mean probability are recorded before the decision.

| Run | Age | Count | UPDATE count | UPDATE ratio | Mean p(update) |
| --- | --- | --- | --- | --- | --- |
| mlp B1 | 0-1 | 3004 | 260 | 0.086551 | 0.139235 |
| mlp B1 | 2-3 | 4892 | 278 | 0.056827 | 0.132181 |
| mlp B1 | 4-7 | 7629 | 372 | 0.048761 | 0.139534 |
| mlp B1 | 8-15 | 9808 | 600 | 0.061175 | 0.159498 |
| mlp B1 | 16-31 | 8097 | 404 | 0.049895 | 0.150578 |
| mlp B1 | 32+ | 9342 | 185 | 0.019803 | 0.134597 |
| mlp B2 | 0-1 | 5336 | 1446 | 0.270990 | 0.312040 |
| mlp B2 | 2-3 | 6957 | 725 | 0.104212 | 0.259762 |
| mlp B2 | 4-7 | 9442 | 961 | 0.101779 | 0.254977 |
| mlp B2 | 8-15 | 9541 | 825 | 0.086469 | 0.240032 |
| mlp B2 | 16-31 | 6678 | 432 | 0.064690 | 0.210283 |
| mlp B2 | 32+ | 4818 | 174 | 0.036115 | 0.194390 |
| mlp B3 | 0-1 | 8810 | 4078 | 0.462883 | 0.433598 |
| mlp B3 | 2-3 | 8257 | 1409 | 0.170643 | 0.368625 |
| mlp B3 | 4-7 | 9666 | 1227 | 0.126940 | 0.355548 |
| mlp B3 | 8-15 | 8483 | 960 | 0.113168 | 0.334671 |
| mlp B3 | 16-31 | 5139 | 377 | 0.073361 | 0.306111 |
| mlp B3 | 32+ | 2417 | 146 | 0.060405 | 0.308521 |
| mlp B4 | 0-1 | 12809 | 7890 | 0.615973 | 0.488452 |
| mlp B4 | 2-3 | 8537 | 1607 | 0.188239 | 0.425817 |
| mlp B4 | 4-7 | 9297 | 1516 | 0.163063 | 0.414519 |
| mlp B4 | 8-15 | 6924 | 845 | 0.122039 | 0.390390 |
| mlp B4 | 16-31 | 4096 | 337 | 0.082275 | 0.379946 |
| mlp B4 | 32+ | 1109 | 100 | 0.090171 | 0.347820 |
| transformer B1 | 0-1 | 4403 | 1795 | 0.407677 | 0.318173 |
| transformer B1 | 2-3 | 4587 | 398 | 0.086767 | 0.181684 |
| transformer B1 | 4-7 | 6759 | 300 | 0.044385 | 0.151795 |
| transformer B1 | 8-15 | 9320 | 362 | 0.038841 | 0.162018 |
| transformer B1 | 16-31 | 8775 | 427 | 0.048661 | 0.180054 |
| transformer B1 | 32+ | 8928 | 205 | 0.022961 | 0.139805 |
| transformer B2 | 0-1 | 9357 | 5740 | 0.613444 | 0.440336 |
| transformer B2 | 2-3 | 6327 | 709 | 0.112059 | 0.289395 |
| transformer B2 | 4-7 | 8779 | 732 | 0.083381 | 0.281842 |
| transformer B2 | 8-15 | 9326 | 871 | 0.093395 | 0.277783 |
| transformer B2 | 16-31 | 5824 | 474 | 0.081387 | 0.243267 |
| transformer B2 | 32+ | 3159 | 134 | 0.042418 | 0.201064 |
| transformer B3 | 0-1 | 16245 | 11300 | 0.695599 | 0.495525 |
| transformer B3 | 2-3 | 8338 | 1900 | 0.227872 | 0.400022 |
| transformer B3 | 4-7 | 8540 | 1456 | 0.170492 | 0.376921 |
| transformer B3 | 8-15 | 6275 | 858 | 0.136733 | 0.349868 |
| transformer B3 | 16-31 | 2768 | 305 | 0.110188 | 0.329991 |
| transformer B3 | 32+ | 606 | 53 | 0.087459 | 0.297579 |
| transformer B4 | 0-1 | 23256 | 18198 | 0.782508 | 0.518610 |
| transformer B4 | 2-3 | 8262 | 2525 | 0.305616 | 0.449198 |
| transformer B4 | 4-7 | 6591 | 1522 | 0.230921 | 0.428662 |
| transformer B4 | 8-15 | 3449 | 652 | 0.189040 | 0.408978 |
| transformer B4 | 16-31 | 1106 | 145 | 0.131103 | 0.394208 |
| transformer B4 | 32+ | 108 | 16 | 0.148148 | 0.415100 |

## Evidence and cleanup

- [Full summary](../../experiments/v2/metrics/final_summary.json), [final protocol manifest](../../experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json), [source freeze](../../experiments/v2/metrics/source_freeze.json), [final run state](../../experiments/v2/metrics/run_state.json).
- [Periodic CSV](../../experiments/v2/metrics/periodic_official_val.csv), [adaptive CSV](../../experiments/v2/metrics/v2_official_val.csv), [MLP train JSON](../../experiments/v2/metrics/mlp_train.json), [Transformer train JSON](../../experiments/v2/metrics/transformer_train.json).
- Raw result, full per-frame packet/age/probability evidence and losslessly compressed predictions: experiments/v2/metrics/official_val/<run>/result.json, experiments/v2/communication/official_val/<run>.json, experiments/v2/predictions/official_val/<run>/.
- [Reset and cleanup evidence](CLEANUP_RESET.md), [train log](../../logs/v2/train.log), [eval/cleanup log](../../logs/v2/eval.log). All 16 results, identities, 3316 unique tokens/run, packet formula, ten bins, six buckets and 48 gzip integrity checks passed independent audit.
- Native process exited code 0. Current run_state includes report and cleanup-final; no experiment writer remains. The original report-stage runtime snapshot is preserved as provenance in the full summary.
- Old dev/selection/checkpoint/log/report artifacts were deleted. V1 results/reports/native code/data/caches/backbone checkpoints/container image were preserved.
- Both redundant latest.pth files were removed; only two final.pth files remain. No experiment epoch/temp/partial/writing/.eval_hook fragments remain.
- Two current-run V2 bytecode caches remain: scripts/__pycache__/v2_run.cpython-311.pyc and v2_eval.cpython-311.pyc. Automatic approval review rejected their deletion, including a literal-path attempt; the only reason returned was blocked by policy.

## Figures

![official_val_bytes_vs_amota](../../experiments/v2/figures/official_val_bytes_vs_amota.png)

![official_val_bytes_vs_map](../../experiments/v2/figures/official_val_bytes_vs_map.png)

![formal_mlp_histogram](../../experiments/v2/figures/formal_mlp_histogram.png)

![formal_mlp_staleness](../../experiments/v2/figures/formal_mlp_staleness.png)

![formal_transformer_histogram](../../experiments/v2/figures/formal_transformer_histogram.png)

![formal_transformer_staleness](../../experiments/v2/figures/formal_transformer_staleness.png)

