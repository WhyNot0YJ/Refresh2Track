# V4 results and research status

**COMPLETED**; completed training epochs: 24; best epoch: 12.

The table includes only actual completed evaluations. Pending cells are not results. Full official val is21scenes/3316samples; the three requested seeds measure inference stability of one training seed.

| Method | Inference seed | mAP | AMOTA | AMOTP | MOTA | IDS | Payload B/s | Serialized B/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | 0 | 0.365131 | 0.375331 | 1.323209 | 0.301300 | 249.000000 | 16501.531966 | 27258.582630 |
| P16 | 0 | 0.360143 | 0.360109 | 1.324434 | 0.292348 | 280.000000 | 15951.109771 | 26708.160434 |
| V4 | 2025 | 0.362754 | 0.365196 | 1.326658 | 0.292277 | 271.000000 | 16501.531966 | 27258.582630 |
| P16 | 2025 | 0.358024 | 0.356050 | 1.323621 | 0.287125 | 241.000000 | 15951.109771 | 26708.160434 |
| V4 | 2026 | 0.363162 | 0.368841 | 1.324988 | 0.295971 | 269.000000 | 16501.531966 | 27258.582630 |
| P16 | 2026 | 0.360242 | 0.361166 | 1.322897 | 0.294302 | 230.000000 | 15951.109771 | 26708.160434 |

Historical references (unchanged; pairing and protocol caveats are stored with each source SHA in `metrics/reference_results.json`):

| Method | Seed | mAP | AMOTA | Reconstructed payload B/s | Serialized B/s |
|---|---:|---:|---:|---:|---:|
| FULL_FP32 | 0 | 0.362958 | 0.362202 | 417623.027744 | 423558.455971 |
| FULL_INT8 | 0 | 0.361107 | 0.353343 | 111223.546441 | 121616.453559 |
| P16 | 0 | 0.360143 | 0.360109 | 15951.109771 | 26708.160434 |
| P32 | 0 | 0.359524 | 0.357565 | 12867.334138 | 23624.384801 |
| INIT_ONLY | 0 | 0.351785 | 0.331364 | 10613.896261 | 21370.946924 |
| V2_MLP_B1 | 0 | 0.362306 | 0.369316 | 15551.230398 | 26308.281062 |

Paired differences:

```json
{
  "pairs": [
    {
      "seed": 0,
      "pairing_verified": true,
      "AMOTA_difference": 0.015222280230387297,
      "mAP_difference": 0.00498803139290771,
      "feature_payload_BPS_difference": 550.4221954161658,
      "feature_payload_cost_ratio": 1.034506827632989
    },
    {
      "seed": 2025,
      "pairing_verified": true,
      "AMOTA_difference": 0.009145786299470327,
      "mAP_difference": 0.0047298000906786375,
      "feature_payload_BPS_difference": 550.4221954161658,
      "feature_payload_cost_ratio": 1.034506827632989
    },
    {
      "seed": 2026,
      "pairing_verified": true,
      "AMOTA_difference": 0.0076755186017199994,
      "mAP_difference": 0.0029199780996166935,
      "feature_payload_BPS_difference": 550.4221954161658,
      "feature_payload_cost_ratio": 1.034506827632989
    }
  ],
  "three_seed_mean_and_sample_std": {
    "AMOTA_difference": {
      "mean": 0.010681195043859207,
      "sample_standard_deviation": 0.004000814014577615
    },
    "mAP_difference": {
      "mean": 0.004212603194401014,
      "sample_standard_deviation": 0.0011268675956439427
    },
    "feature_payload_BPS_difference": {
      "mean": 550.4221954161658,
      "sample_standard_deviation": 0.0
    }
  }
}
```

Cost ratios above are descriptive. Nearby cost uses a ±10% payload convention fixed in this reporting code before training; raw ratios are always reported. Unequal payload rates do not establish superiority at the same cost. No statistical significance or cross-training-seed stability is claimed.

1. Budget removal: the separate embedding and excess-budget penalty are removed; the only objective is native task plus0.05×optional communication. Preflight status: PASS.
2. Learned optional UPDATE and actual payload: latest completed training optional rate=0.053561; payload=15304.966042B/s. Final inference values are in the table; pending training values must not be presented as val results.
3. Latest completed training soft/hard gap: signed mean=724.247082B/frame, absolute mean=823.552399B/frame, near-threshold fraction=0.180642. Final per-frame data are retained separately.
4. Short-snippet native task gradient=0.001785; communication gradient=0.655835. Cold forced-only frames correctly have no optional gradient. Every formal epoch separately verifies a nonzero native task gradient and freezes native weights/BN.
5. INIT_ONLY-style degeneration: Not established until completed training and full val; inspect actual optional rate and cache staleness.. The coordinator records/stops on a completed-epoch optional update rate≤0.001 or≥0.999, finite-loss/gradient failures, and performs no second λ search.
6. Stable advantage versus paired P16 at nearby cost: All three inference seeds have a positive AMOTA difference at nearby observed cost. The claim requires all three matched pairs and their observed costs; no result is inferred from the design.
7. Evidence supports archive preservation, implemented method and explicitly passed gates/observed rows. Pending performance claims remain hypotheses; completed table entries describe official-val observations only. Independent test generalization and a full rate–utility curve are not established. A single best checkpoint is not a proven Pareto optimum.
8. Paper choice: V4 is supported as a main-method candidate by this completed official-val comparison. Retain V2 as a historical reference; its seed0 row is not a fresh paired V4/V2 experiment.

Official val was used for checkpoint selection/early stopping and performance reporting. These are not independent test-set results. Best checkpoint selection record:

```json
{
  "patience": 4,
  "last_validation_epoch": 24,
  "last_score": 0.3639398225385394,
  "best_score": 0.37533100220281657,
  "best_epoch": 12,
  "status": "PASS",
  "best_checkpoint": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/checkpoints/best.pth",
  "checkpoint_sha256": "0a15216b74b3fc2ebce0570e8aae436c156421b9c7119ec45755527d6cfa4a5f",
  "validation_checkpoint_sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07",
  "scheduler_state_sha256": "32fba573cc3743c7de65afd7bb5d790d45f7708d15a4ed8666154fe838033af2",
  "best_results": {
    "status": "PASS",
    "identity": {
      "protocol": {
        "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
        "model": "v4_budget_free_mlp_lambda005_seed0",
        "manifest_sha256": "33a534f091944ce8338254d487b1cef31c9040296d2953ce9ea5580c121c1319",
        "settings": {
          "model": "v4_budget_free_mlp_lambda005_seed0",
          "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
          "max_epochs": 36,
          "training_seed": 0,
          "inference_seed": 0,
          "paired_inference_seeds": [
            0,
            2025,
            2026
          ],
          "precision": "FP32",
          "batch_size": 4,
          "train_samples": 7362,
          "train_scenes": 46,
          "val_samples": 3316,
          "val_scenes": 21,
          "full_int8_mean_bytes": 12840.189523169913,
          "lambda_comm": 0.05,
          "threshold": 0.5,
          "scheduler_state_dimension": 2324,
          "scheduler_parameters": 71361,
          "latent_update_bytes": 780,
          "geometry_bytes_per_query": 52,
          "frequency_hz": 10,
          "validation_interval": 3,
          "patience": 4,
          "minimum_stop_epoch": 12,
          "diagnostic_steps": [
            2,
            10
          ],
          "optimizer": {
            "type": "AdamW",
            "peak_lr": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 5.0
          },
          "lr_schedule": {
            "warmup_epochs": 1,
            "warmup_start_lr": 1e-05,
            "min_lr": 1e-06,
            "cosine_total_horizon_epochs": 36
          },
          "objective": "native_task_mean + lambda_comm * mean_real_frames(780 * sum(p_optional) / R)",
          "communication_reduction": "sum optional probabilities within each real frame; mean over real active frames; no query-count division",
          "reference_source": "configs/main_experiment/budgets.json:full_int8_mean_bytes; train-only CTKMSG01 FEAT_INT8 serialized mean",
          "official_val_use": "checkpoint selection, early stopping, and performance reporting; not an independent test set",
          "serializer": "APTU0001",
          "forced_update": "no receiver cache only",
          "checkpoint_rule": "from-scratch scheduler; epoch-boundary latest/last and strict official-val AMOTA best",
          "scene_order_rule": "sorted scenes shuffled by Random(training_seed+epoch-1)",
          "gradient_diagnostic_rule": "steps 2 and 10 in every epoch; no posthoc selection"
        }
      },
      "method": "V4",
      "epoch": 12,
      "inference_seed": 0,
      "selection": true,
      "checkpoint_sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07",
      "used_checkpoint_sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07"
    },
    "pairing_identity": {
      "protocol": {
        "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
        "model": "v4_budget_free_mlp_lambda005_seed0",
        "manifest_sha256": "33a534f091944ce8338254d487b1cef31c9040296d2953ce9ea5580c121c1319",
        "settings": {
          "model": "v4_budget_free_mlp_lambda005_seed0",
          "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
          "max_epochs": 36,
          "training_seed": 0,
          "inference_seed": 0,
          "paired_inference_seeds": [
            0,
            2025,
            2026
          ],
          "precision": "FP32",
          "batch_size": 4,
          "train_samples": 7362,
          "train_scenes": 46,
          "val_samples": 3316,
          "val_scenes": 21,
          "full_int8_mean_bytes": 12840.189523169913,
          "lambda_comm": 0.05,
          "threshold": 0.5,
          "scheduler_state_dimension": 2324,
          "scheduler_parameters": 71361,
          "latent_update_bytes": 780,
          "geometry_bytes_per_query": 52,
          "frequency_hz": 10,
          "validation_interval": 3,
          "patience": 4,
          "minimum_stop_epoch": 12,
          "diagnostic_steps": [
            2,
            10
          ],
          "optimizer": {
            "type": "AdamW",
            "peak_lr": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 5.0
          },
          "lr_schedule": {
            "warmup_epochs": 1,
            "warmup_start_lr": 1e-05,
            "min_lr": 1e-06,
            "cosine_total_horizon_epochs": 36
          },
          "objective": "native_task_mean + lambda_comm * mean_real_frames(780 * sum(p_optional) / R)",
          "communication_reduction": "sum optional probabilities within each real frame; mean over real active frames; no query-count division",
          "reference_source": "configs/main_experiment/budgets.json:full_int8_mean_bytes; train-only CTKMSG01 FEAT_INT8 serialized mean",
          "official_val_use": "checkpoint selection, early stopping, and performance reporting; not an independent test set",
          "serializer": "APTU0001",
          "forced_update": "no receiver cache only",
          "checkpoint_rule": "from-scratch scheduler; epoch-boundary latest/last and strict official-val AMOTA best",
          "scene_order_rule": "sorted scenes shuffled by Random(training_seed+epoch-1)",
          "gradient_diagnostic_rule": "steps 2 and 10 in every epoch; no posthoc selection"
        }
      },
      "seed": 0,
      "native_checkpoint_sha256": "de543ce784175dd62f62d4e9d6b17c5ae7757f00548efbe49f8671e98ef0c4e2",
      "config_sha256": "d2b0fe0a28a8c05a2379fe6e8532b0b4f12006d4b594f5bc2d7fcaa343b07b03",
      "cache_sha256": "242b688e044ef5a879c16be19462a65638cc2874ad8d8f52f34100b8e5d8ede9",
      "cache_file_count": 3316,
      "val_order_sha256": "40c576b1aa9ea7389e05ff042f2d689064929bb1090127ad41697092eb7d2a4a",
      "native_source_sha256": "8a8029265dd63434873f574e94c4f871c963055f9cab818be4bbc0fa1309f920",
      "dataset_source_sha256": "822eefa75967897bbb709bf648045fd1f8be81c6dfab602fa3922a8274d7f3a4",
      "fields": [
        "query_feats",
        "query_embeds",
        "cache_motion_feats",
        "ref_pts",
        "pred_boxes"
      ],
      "filter_stage": "all cached sender rows before vehicle spatial filtering",
      "serializer": "APTU0001",
      "payload": "52*N+780*U",
      "frequency_hz": 10,
      "rng_reset": "after scheduler/model construction, checkpoint load and iter(loader); before first next(iterator); no scene reseeding",
      "environment": {
        "python": "3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) \n[GCC 13.3.0]",
        "torch": "1.9.1+cu111",
        "cuda": "11.1",
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
        "precision": "FP32",
        "batch": 1,
        "workers": 0,
        "tf32_matmul": true,
        "tf32_cudnn": true,
        "cudnn_deterministic": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false,
        "threads": 2,
        "PYTHONHASHSEED": "0"
      },
      "rng_after_reset": {
        "python": "58ae6fb433941c12f62693dbc475e5d07ed8e961a2ee47986acad525f9defd02",
        "numpy": "b7a5b53832e32aa9cbe9972fc70a83c749083c89e8a876d0d0bd88a06adb3a52",
        "torch_cpu": "1ccf17250133dec5e96478acd63e13d728d0dbbabb30389415a302c6f872f81f",
        "torch_cuda": [
          "4cedd2251d8625a086485e24cac8f68fb001f603144c3ebfbbb915c1acd632b8"
        ]
      }
    },
    "pairing": {
      "protocol": {
        "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
        "model": "v4_budget_free_mlp_lambda005_seed0",
        "manifest_sha256": "33a534f091944ce8338254d487b1cef31c9040296d2953ce9ea5580c121c1319",
        "settings": {
          "model": "v4_budget_free_mlp_lambda005_seed0",
          "protocol": "V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0",
          "max_epochs": 36,
          "training_seed": 0,
          "inference_seed": 0,
          "paired_inference_seeds": [
            0,
            2025,
            2026
          ],
          "precision": "FP32",
          "batch_size": 4,
          "train_samples": 7362,
          "train_scenes": 46,
          "val_samples": 3316,
          "val_scenes": 21,
          "full_int8_mean_bytes": 12840.189523169913,
          "lambda_comm": 0.05,
          "threshold": 0.5,
          "scheduler_state_dimension": 2324,
          "scheduler_parameters": 71361,
          "latent_update_bytes": 780,
          "geometry_bytes_per_query": 52,
          "frequency_hz": 10,
          "validation_interval": 3,
          "patience": 4,
          "minimum_stop_epoch": 12,
          "diagnostic_steps": [
            2,
            10
          ],
          "optimizer": {
            "type": "AdamW",
            "peak_lr": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 5.0
          },
          "lr_schedule": {
            "warmup_epochs": 1,
            "warmup_start_lr": 1e-05,
            "min_lr": 1e-06,
            "cosine_total_horizon_epochs": 36
          },
          "objective": "native_task_mean + lambda_comm * mean_real_frames(780 * sum(p_optional) / R)",
          "communication_reduction": "sum optional probabilities within each real frame; mean over real active frames; no query-count division",
          "reference_source": "configs/main_experiment/budgets.json:full_int8_mean_bytes; train-only CTKMSG01 FEAT_INT8 serialized mean",
          "official_val_use": "checkpoint selection, early stopping, and performance reporting; not an independent test set",
          "serializer": "APTU0001",
          "forced_update": "no receiver cache only",
          "checkpoint_rule": "from-scratch scheduler; epoch-boundary latest/last and strict official-val AMOTA best",
          "scene_order_rule": "sorted scenes shuffled by Random(training_seed+epoch-1)",
          "gradient_diagnostic_rule": "steps 2 and 10 in every epoch; no posthoc selection"
        }
      },
      "seed": 0,
      "native_checkpoint_sha256": "de543ce784175dd62f62d4e9d6b17c5ae7757f00548efbe49f8671e98ef0c4e2",
      "config_sha256": "d2b0fe0a28a8c05a2379fe6e8532b0b4f12006d4b594f5bc2d7fcaa343b07b03",
      "cache_sha256": "242b688e044ef5a879c16be19462a65638cc2874ad8d8f52f34100b8e5d8ede9",
      "cache_file_count": 3316,
      "val_order_sha256": "40c576b1aa9ea7389e05ff042f2d689064929bb1090127ad41697092eb7d2a4a",
      "native_source_sha256": "8a8029265dd63434873f574e94c4f871c963055f9cab818be4bbc0fa1309f920",
      "dataset_source_sha256": "822eefa75967897bbb709bf648045fd1f8be81c6dfab602fa3922a8274d7f3a4",
      "fields": [
        "query_feats",
        "query_embeds",
        "cache_motion_feats",
        "ref_pts",
        "pred_boxes"
      ],
      "filter_stage": "all cached sender rows before vehicle spatial filtering",
      "serializer": "APTU0001",
      "payload": "52*N+780*U",
      "frequency_hz": 10,
      "rng_reset": "after scheduler/model construction, checkpoint load and iter(loader); before first next(iterator); no scene reseeding",
      "environment": {
        "python": "3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) \n[GCC 13.3.0]",
        "torch": "1.9.1+cu111",
        "cuda": "11.1",
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
        "precision": "FP32",
        "batch": 1,
        "workers": 0,
        "tf32_matmul": true,
        "tf32_cudnn": true,
        "cudnn_deterministic": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false,
        "threads": 2,
        "PYTHONHASHSEED": "0"
      },
      "rng_after_reset": {
        "python": "58ae6fb433941c12f62693dbc475e5d07ed8e961a2ee47986acad525f9defd02",
        "numpy": "b7a5b53832e32aa9cbe9972fc70a83c749083c89e8a876d0d0bd88a06adb3a52",
        "torch_cpu": "1ccf17250133dec5e96478acd63e13d728d0dbbabb30389415a302c6f872f81f",
        "torch_cuda": [
          "4cedd2251d8625a086485e24cac8f68fb001f603144c3ebfbbb915c1acd632b8"
        ]
      }
    },
    "method": "V4",
    "model": "v4_budget_free_mlp_lambda005_seed0",
    "epoch": 12,
    "seed": 0,
    "sample_count": 3316,
    "scene_count": 21,
    "metrics": {
      "mAP": 0.36513092109991985,
      "AMOTA": 0.37533100220281657,
      "AMOTP": 1.3232086537008918,
      "MOTA": 0.301300270001421,
      "IDS": 249.0,
      "FP": 4471.0,
      "FN": 14947.0,
      "recall": 0.4689853630808583
    },
    "communication": {
      "queries": 44329,
      "updates": 4060,
      "forced": 1557,
      "optional_queries": 42772,
      "optional_updates": 2503,
      "mean_bytes": 2725.8582629674306,
      "p95_bytes": 5158.75,
      "bytes_per_second": 27258.582629674307,
      "serialized_message_mean_B_per_frame": 2725.8582629674306,
      "serialized_message_P95_B_per_frame": 5158.75,
      "serialized_message_BPS": 27258.582629674307,
      "serialized_BPS": 27258.582629674307,
      "feature_payload_mean_B_per_frame": 1650.1531966224368,
      "feature_payload_P95_B_per_frame": 4108.0,
      "feature_payload_BPS": 16501.531966224367,
      "feature_payload_latent_BPS": 9550.06031363088,
      "geometry_BPS": 6951.471652593486,
      "optional_update_ratio": 0.05851959225661648,
      "forced_init_ratio": 0.03512373389880214,
      "total_update_ratio": 0.09158789956913081,
      "estimated_optional_bytes": 4186056.559218136,
      "actual_optional_bytes": 1952340,
      "estimated_optional_mean_B_per_frame": 1262.3813507895466,
      "actual_optional_mean_B_per_frame": 588.7635705669481,
      "soft_hard_optional_gap_bytes": 2233716.559218136,
      "soft_hard_optional_gap_mean_B_per_frame": 673.6177802225983,
      "soft_hard_optional_gap_absolute_mean_B_per_frame": 756.4652578940629,
      "probability_near_threshold_fraction": 0.1735948751519686,
      "probability_histogram": [
        30160,
        1567,
        1143,
        1117,
        6282,
        2503,
        0,
        0,
        0,
        0
      ],
      "probability_histogram_bins": [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
      ],
      "probability_count": 42772,
      "probability_scope": "optional queries only",
      "cache_age_mean": 25.77480594781633,
      "cache_age_p95": 108.0,
      "cache_age_max": 248.0,
      "cache_age_scope": "optional queries before decision, infrastructure frames",
      "BPS_scope": "nominal 10Hz mean bytes/frame times 10; not measured network throughput",
      "optional_estimate_type": "learned optional probability sum"
    },
    "mAP": 0.36513092109991985,
    "AMOTA": 0.37533100220281657,
    "AMOTP": 1.3232086537008918,
    "MOTA": 0.301300270001421,
    "IDS": 249.0,
    "FP": 4471.0,
    "FN": 14947.0,
    "recall": 0.4689853630808583,
    "queries": 44329,
    "updates": 4060,
    "forced": 1557,
    "optional_queries": 42772,
    "optional_updates": 2503,
    "mean_bytes": 2725.8582629674306,
    "p95_bytes": 5158.75,
    "bytes_per_second": 27258.582629674307,
    "serialized_message_mean_B_per_frame": 2725.8582629674306,
    "serialized_message_P95_B_per_frame": 5158.75,
    "serialized_message_BPS": 27258.582629674307,
    "serialized_BPS": 27258.582629674307,
    "feature_payload_mean_B_per_frame": 1650.1531966224368,
    "feature_payload_P95_B_per_frame": 4108.0,
    "feature_payload_BPS": 16501.531966224367,
    "feature_payload_latent_BPS": 9550.06031363088,
    "geometry_BPS": 6951.471652593486,
    "optional_update_ratio": 0.05851959225661648,
    "forced_init_ratio": 0.03512373389880214,
    "total_update_ratio": 0.09158789956913081,
    "estimated_optional_bytes": 4186056.559218136,
    "actual_optional_bytes": 1952340,
    "estimated_optional_mean_B_per_frame": 1262.3813507895466,
    "actual_optional_mean_B_per_frame": 588.7635705669481,
    "soft_hard_optional_gap_bytes": 2233716.559218136,
    "soft_hard_optional_gap_mean_B_per_frame": 673.6177802225983,
    "soft_hard_optional_gap_absolute_mean_B_per_frame": 756.4652578940629,
    "probability_near_threshold_fraction": 0.1735948751519686,
    "probability_histogram": [
      30160,
      1567,
      1143,
      1117,
      6282,
      2503,
      0,
      0,
      0,
      0
    ],
    "probability_histogram_bins": [
      0.0,
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7,
      0.8,
      0.9,
      1.0
    ],
    "probability_count": 42772,
    "probability_scope": "optional queries only",
    "cache_age_mean": 25.77480594781633,
    "cache_age_p95": 108.0,
    "cache_age_max": 248.0,
    "cache_age_scope": "optional queries before decision, infrastructure frames",
    "BPS_scope": "nominal 10Hz mean bytes/frame times 10; not measured network throughput",
    "optional_estimate_type": "learned optional probability sum",
    "checkpoint_sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07",
    "used_checkpoint_sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07",
    "used_checkpoint": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/checkpoints/candidate.pth",
    "checkpoint_role": "V4 scheduler",
    "prediction_sha256": "b147acf6224d53173b97dd0d39788c396d4ccdcb0c78f55459ca70c808377004",
    "evaluator_inputs": {
      "checkpoint": {
        "path": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/checkpoints/candidate.pth",
        "sha256": "d764b1101c178d7eee5954764f3cd2844b9ce1a4706aa26c72e2eb357daf3b07"
      },
      "predictions": {
        "path": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/predictions.pkl",
        "sha256": "b147acf6224d53173b97dd0d39788c396d4ccdcb0c78f55459ca70c808377004",
        "archive": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/predictions.pkl.gz",
        "archive_sha256": "9eedf116dffe38faa48381ee0efec114814c892f9c21c666949a86678cb576d9"
      },
      "tracking": {
        "path": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/results_nusc.json",
        "sha256": "8c9ce345a6677669746d37234c34a1ae1cc470e1efabd95bcae1a97a0eedb66e",
        "archive": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/results_nusc.json.gz",
        "archive_sha256": "57672e25cfb6ec83d48fe6c67cec50d6be18325ad128e9b1137dcfb49711413c"
      },
      "detection": {
        "path": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/results_nusc_det.json",
        "sha256": "f49121b48bef5391fda62129ace70f416478034a03f0819c2a7d0d54e7461951",
        "archive": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/results_nusc_det.json.gz",
        "archive_sha256": "259d930d5a5d515e90a053b208fc08d2d180512cb1494a230f844c1434ea5aea"
      }
    },
    "result_file": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/validation/epoch012/result.json",
    "seconds": 1868.9791387759906,
    "completed_utc": "2026-10-12T07:30:50.160024+00:00"
  },
  "selection_metadata_commit_only": true,
  "stop_epoch": 24,
  "stop_reason": "early_stopping"
}
```

Run state:

```json
{
  "status": "COMPLETED",
  "phase": "complete",
  "utc": "2026-10-13T03:09:00.980317+00:00",
  "model": "v4_budget_free_mlp_lambda005_seed0",
  "epoch": 12,
  "seed": 2026,
  "resume_audit": "experiments/v4_budget_free/audit/git_merge_resume.json",
  "method": "P16",
  "best_epoch": 12,
  "checkpoint": "/mnt/d/CoopTrack_Audit/experiments/v4_budget_free/checkpoints/best.pth",
  "completed_epoch": 24
}
```

Minimal next action: follow `experiments/v4_budget_free/run_state.json` and `logs/v4_budget_free/coordinator.log`; resume an interrupted compatible run with `python -B scripts/v4_budget_free/run.py --launch --resume`.
