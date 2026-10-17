# 无预算刷新收益监督 VALUE_V1

状态：`COMPLETE`（独立 official-val 7000 阶段）；原 160000 阶段仍为 `TEACHER_LIMIT_REACHED`。

```text
FINAL_H8_CHECKPOINT: PASS
FINAL_H1_CHECKPOINT: PASS
H8_OFFICIAL_VAL: PASS
H1_OFFICIAL_VAL: PASS
EVAL_FORWARDS_USED: 6632
```

两个冻结模型各从 official val 第一帧重新开始，seed=0，3316 帧/21 scenes；实际独立 native frame-forward 计数为 6632/7000。没有生成新标签、重训、调参或重跑旧参考。

## 旧成果保护与目录入口

本次 fresh 基线保护为 `PASS_NEW_EVAL_PRESERVATION`：12925/12925 份 immutable 文件未变，原 VALUE 数据、标签、权重、160000 state/log/manifest、1816 帧 partial 全部保留。

历史整个工作区保护仍为 `RAW_PRESERVATION_FAIL`（原始检查 `FAIL`），不能被本次 PASS 覆盖。原检查 539 份选定 SHA 中有 3 份变化，另有 348 条旧路径缺失（116 文件、232 目录）；缺失构建/下载/bytecode 中间产物和空目录的解释保留原证据，不归因为本次新评估。

历史事实：[experiments/budget_free/value_v1/evidence/protection_after.json](../../experiments/budget_free/value_v1/evidence/protection_after.json)、[experiments/budget_free/value_v1/evidence/external_preservation_changes.json](../../experiments/budget_free/value_v1/evidence/external_preservation_changes.json)。fresh 保护：[experiments/budget_free/value_v1/final_official_val_7000/evidence/preservation_after.json](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/preservation_after.json)。原终端报告精确字节归档：[experiments/budget_free/value_v1/final_official_val_7000/evidence/original_terminal_report.md](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/original_terminal_report.md)。

旧入口：`artifacts/core/`、`experiments/main/`、`experiments/v2/`、`audits/v2_bug_audit/`、`experiments/v2/ablations/refresh_2x2/`。本轮训练入口 `scripts/budget_free/run.py`，配置 `configs/budget_free/value_v1.json`；新评估入口 `scripts/budget_free/final_official_val.py`，独立产物 `experiments/budget_free/value_v1/final_official_val_7000`。唯一最终报告为本文件；实验索引由根任务维护。

## 方法、标签与训练

学生输入为当前 INT8 roundtrip latent、接收端实际解码镜像及差值（各 768 维）+20 维 causal 状态，共 2324 维；无预算、GT、未来、车端私有状态或数值 ID embedding。三个 latent 共享 `768→64 ReLU`，拼接后 `212→128→64→1`，推理 sigmoid≥0.5 UPDATE，只有缺缓存时强制初始化。几何每帧 FP32，latent bundle UPDATE 增加 780 B；无 rate loss、TTL、Top-K 或硬带宽保证。

冻结 native eval 教师从相同完整前缀和 RNG 执行 UPDATE/REUSE；H8 为 t…t+7，候选后七帧 REUSE、其余 ID 固定参考 continuation；H1 取同分支第一帧。GT 只进入离线评分，`E=FN+FP+IDS+0.1×Σmin(distance/2,1)`，Δ=J(REUSE)−J(UPDATE)。固定 epsilon H8=0.010000、H1=0.010000，仅 Δ>epsilon 为正类，round0 锁定归一化、margin 尺度和权重均值。部署 continuation 与标签 continuation 有差异。

正确性：`PASS`；null、分支顺序、信息隔离等完整原证据见 [experiments/budget_free/value_v1/evidence/correctness.json](../../experiments/budget_free/value_v1/evidence/correctness.json)。

| 集合 | paired samples | train scenes | H8 positives / ratio | H1 positives / ratio |
| --- | --- | --- | --- | --- |
| Pilot | 1024 | 46 | 69 / 6.74% | 11 / 1.07% |
| D0 | 4096 | 46 | 300 / 7.32% | 43 / 1.05% |
| D1 | 2048 | 46 | 170 / 8.30% | 26 / 1.27% |

Pilot 是 D0 共享样本，不能相加为独立新增行。D0=4096、D1=2048，仅一轮自身 H8 策略采样；final H8/H1 同 6144 行、同 D0 归一化、同 seed0 初始化，从头拟合各 40 epochs。CPU FP32，batch128，AdamW wd1e−4、clip5，首 epoch 1e−4→1e−3，随后固定余弦至末步 1e−5；只训练学生，raw-logit BCE 使用固定 dataset mean weight，无类别平衡器。

| 模型 | rows / epochs | fit precision | fit recall | fit PR-AUC | final checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- |
| BF_VALUE_H8 | 6144/40 | 0.554391 | 0.900000 | 0.773517 | `111b14ae0b9a038b550e84d16cd665f9264dea9bcc670d40e868adafd5986d38` |
| BF_VALUE_H1 | 6144/40 | 0.820896 | 0.797101 | 0.812792 | `87044b5142e7cdd9407ad59d617bee46bcebd39606dbd649b22fab45116b39f1` |

以上 fit 指标是训练内诊断，不能称为独立泛化或 official-val。训练/模型核验：[experiments/budget_free/value_v1/evidence/final_training_identity_review.json](../../experiments/budget_free/value_v1/evidence/final_training_identity_review.json)、[experiments/budget_free/value_v1/evidence/new_evaluation_checkpoint_preflight.json](../../experiments/budget_free/value_v1/evidence/new_evaluation_checkpoint_preflight.json)。Pilot/D0/D1 完整分布与分位数见各 `dataset/*_summary.json`；原终端报告保留全部训练及来源修订细节。

## 完整 official-val 与指定旧参考

mean/P95 是实际 application payload B/frame。mAP/AMOTA/MOTA 越大越好；AMOTP/IDS 越小越好。所有有利和不利数值保留。

| 方法 | mean B/frame | P95 B/frame | mAP | AMOTA | AMOTP | MOTA | IDS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF_VALUE_H8 | 2465.701448 | 4640 | 0.362236 | 0.357951 | 1.322102 | 0.284248 | 217 |
| BF_VALUE_H1 | 2145.797949 | 3801 | 0.353286 | 0.333465 | 1.332828 | 0.267905 | 292 |
| mlp_B1 | 2630.828106 | 4825 | 0.362306 | 0.369316 | 1.322707 | 0.298956 | 274 |
| P16 | 2670.816043 | 5220.500000 | 0.360143 | 0.360109 | 1.324434 | 0.292348 | 280 |
| P32 | 2362.438480 | 4238 | 0.359524 | 0.357565 | 1.323500 | 0.285207 | 256 |

## UPDATE 与缓存年龄

optional ratio = optional UPDATE / 已有缓存 query；含 forced ratio = (optional UPDATE+forced init) / 全 query。年龄在动作之前统计，排除缺缓存 query。

| 方法 | optional UPDATE / queries | optional ratio | 含 forced UPDATE / queries | 含 forced ratio | age mean | median | P95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF_VALUE_H8 | 1397/42772 | 3.27% | 2954/44329 | 6.66% | 34.001964 | 20 | 117 | 247 |
| BF_VALUE_H1 | 37/42772 | 0.09% | 1594/44329 | 3.60% | 51.460137 | 35 | 157 | 248 |

重复刷新：BF_VALUE_H8 的 age0–1 optional UPDATE 占全部 optional UPDATE 33.50%；BF_VALUE_H1 的 age0–1 optional UPDATE 占全部 optional UPDATE 8.11%。额外通信与年龄关系见第三张图及逐帧 communication；无预算输入不保证省通信。

## 新减旧差值（描述性跨协议比较）

rate gap=(new mean−old mean)/old mean；delta 全部为 new−old。六对均 `strict_same_protocol=False`，没有插值或预算匹配重跑。

| new | old | ΔB/frame | rate gap | ΔP95 | ΔmAP | ΔAMOTA | ΔAMOTP | ΔMOTA | ΔIDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF_VALUE_H8 | mlp_B1 | -165.126659 | -6.28% | -185 | -0.000071 | -0.011364 | -0.000605 | -0.014708 | -57 |
| BF_VALUE_H8 | P16 | -205.114596 | -7.68% | -580.500000 | 0.002093 | -0.002157 | -0.002333 | -0.008100 | -63 |
| BF_VALUE_H8 | P32 | 103.262967 | 4.37% | 402 | 0.002712 | 0.000387 | -0.001398 | -0.000959 | -39 |
| BF_VALUE_H1 | mlp_B1 | -485.030157 | -18.44% | -1024 | -0.009020 | -0.035851 | 0.010121 | -0.031050 | 18 |
| BF_VALUE_H1 | P16 | -525.018094 | -19.66% | -1419.500000 | -0.006857 | -0.026644 | 0.008394 | -0.024442 | 12 |
| BF_VALUE_H1 | P32 | -216.640531 | -9.17% | -437 | -0.006237 | -0.024100 | 0.009329 | -0.017301 | 36 |

H8−H1：mean_bytes_per_frame=319.903498，p95_bytes_per_frame=839，mAP=0.008950，AMOTA=0.024486，AMOTP=-0.010727，MOTA=0.016342，IDS=-75。这给出同一训练流程下有限窗口监督的实际差异；保留每项方向，不能仅按模型名字判胜。

- BF_VALUE_H8 nearest：P24，rate gap=0.08%。
- BF_VALUE_H8 lower_bracket：P24，rate gap=0.08%。
- BF_VALUE_H8 upper_bracket：P16，rate gap=-7.68%。
- BF_VALUE_H1 nearest：P32，rate gap=-9.17%。
- BF_VALUE_H1 lower_bracket：N/A，rate gap=N/A。
- BF_VALUE_H1 upper_bracket：P32，rate gap=-9.17%。

全表数值 cost/mAP/AMOTA frontier：`INIT_ONLY, P24, P32, P6, mlp_B1, transformer_B2, BF_VALUE_H8, BF_VALUE_H1`。它仅为当前各协议测点的数值非支配集合，不证明严格同协议最优或统计显著。

指定三旧点的记录均为 APTU0001，已有审核确认 seed placement；本报告不虚构 serializer/RNG 差异。其训练目标、历史执行协议和冻结完整性仍各自保留，因此本比较严格限于描述性跨协议结果，不能当作同协议、同预算、因果优越性比较。旧 raw generation/input-time SHA 仍为 `UNAVAILABLE`；当前 archive SHA 只标识现存归档，不能回填历史生成链。

实际结果：BF_VALUE_H8 相对旧 MLP_B1 mean B/frame 减少165.126659、mAP 减少0.000071、AMOTA 减少0.011364、IDS 减少57；BF_VALUE_H1 相对旧 MLP_B1 mean B/frame 减少485.030157、mAP 减少0.009020、AMOTA 减少0.035851、IDS 增加18。各成本与任务方向并列披露，不把混合结果写成全面胜出。

## 预测、评估与结果身份链

完整 raw generation SHA=evaluator raw-input SHA；formatted tracking/detection 与官方 evaluator 输出分别绑定实际字节。


**BF_VALUE_H8**

- raw generation / evaluator raw input：`d44aab64155749e06fcc56c1651acc7dd2403184aca0cf933ba4e63bf3fe4c3d`。
- tracking_json：`5cc23d40707a8ae0cd3026249e1b064df86784047931de3004f136e3dd6182b0`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/results_nusc.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/results_nusc.json)。
- detection_json：`a12a05fa729dd73016656cfacd2b43275fc4c0059beaab07fd0c899234af1b55`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/results_nusc_det.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/results_nusc_det.json)。
- 官方 metric：`406343724b292a43f5cd91ee37c54126b54cc61d8457948a77a936b33b1815a2`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/det/metrics_summary.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/det/metrics_summary.json)。
- 官方 metric：`fc50a4672662f66d270d7aa07ca2cfbd9fc74157119719abdfb76ebfb1a5838b`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/track/metrics_summary.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H8_seed0/track/metrics_summary.json)。
- 汇总 metric：`c8ace07d3fa6da92808abff39bab8e86550990c1160ddf045d78d1ab4ccdbac0`，[experiments/budget_free/value_v1/final_official_val_7000/metrics/BF_VALUE_H8_seed0.json](../../experiments/budget_free/value_v1/final_official_val_7000/metrics/BF_VALUE_H8_seed0.json)。
- run manifest：`e8a0ab3ea992a8a0ae2705a459ea50394973fc38c8668fd884b0eb1085148422`，[experiments/budget_free/value_v1/final_official_val_7000/evidence/BF_VALUE_H8_seed0_manifest.json](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/BF_VALUE_H8_seed0_manifest.json)。

**BF_VALUE_H1**

- raw generation / evaluator raw input：`82116c923d94e88314173d9e68de0bfe0ac11e5542caddedccb96f08a6193745`。
- tracking_json：`4ac7bba52554970555a332b2063a9894418392e51ff71c892a005cde59430904`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/results_nusc.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/results_nusc.json)。
- detection_json：`00d8b9aab09196e1d9cdd526dc855faf39ebd10e62f2a3ca4a6c5b1715a0d5ec`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/results_nusc_det.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/results_nusc_det.json)。
- 官方 metric：`428fb54159e45436ef08a72c95a0ba1296452a696e752b3ed7f7f11af9202f13`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/det/metrics_summary.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/det/metrics_summary.json)。
- 官方 metric：`43e102b8e214613acfa08d087f8dd7f9b92baf71132bb8b1187fc737956780f5`，[experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/track/metrics_summary.json](../../experiments/budget_free/value_v1/final_official_val_7000/predictions/BF_VALUE_H1_seed0/track/metrics_summary.json)。
- 汇总 metric：`04a7e5e061a542618ba155dbcfcad3451062707bcc4fae52cb2ee23b5e65b94d`，[experiments/budget_free/value_v1/final_official_val_7000/metrics/BF_VALUE_H1_seed0.json](../../experiments/budget_free/value_v1/final_official_val_7000/metrics/BF_VALUE_H1_seed0.json)。
- run manifest：`443b8dc89d8dc00743891126ae310162b476b608e2a8f9bf45ad3383e2e13a39`，[experiments/budget_free/value_v1/final_official_val_7000/evidence/BF_VALUE_H1_seed0_manifest.json](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/BF_VALUE_H1_seed0_manifest.json)。

新 frozen evaluation manifest：`efb76f869e7d0371a4f05279002557c1308a71f8d2c0d0181fe3bc604ae1d6d5`；scoring audit `31439c48a47e41000e39acfc801f334b12a7de92d3d6fba126a020d0165bd6c3`（PASS）：[experiments/budget_free/value_v1/final_official_val_7000/evidence/final_official_val_audit.json](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/final_official_val_audit.json)。comparison：[experiments/budget_free/value_v1/final_official_val_7000/metrics/comparison.json](../../experiments/budget_free/value_v1/final_official_val_7000/metrics/comparison.json)。

## 原 160000 终端与 same-prefix 核验

原计数160000（含 offline native158184 + H8 partial1816），原 H8 为 `PARTIAL_NOT_EVALUATED` 1816/3316，原 H1 `NOT_RUN`；旧 state/log/manifest/partial 均未覆盖。独立新计数6632不回写旧 ledger，也不将旧 partial 作为新完整预测复用。

same-prefix `PASS`：原 H8 1816 帧与新 H8 完整结果前1816帧的全部 prediction 字段及 communication 语义一致，差异均0；仅排除 scheduler_ms/cooperative_ms 两项计时。纯CPU审查0native/GPU：[experiments/budget_free/value_v1/final_official_val_7000/evidence/h8_original_prefix_equivalence.json](../../experiments/budget_free/value_v1/final_official_val_7000/evidence/h8_original_prefix_equivalence.json)（`9791c1dca77ea266e922d75609c6b0328e2bb68cab1eb9239b3f1ce5d3a1b9a5`）。

## 图与限制

- [experiments/budget_free/value_v1/final_official_val_7000/figures/bytes_vs_map.png](../../experiments/budget_free/value_v1/final_official_val_7000/figures/bytes_vs_map.png)
- [experiments/budget_free/value_v1/final_official_val_7000/figures/bytes_vs_amota.png](../../experiments/budget_free/value_v1/final_official_val_7000/figures/bytes_vs_amota.png)
- [experiments/budget_free/value_v1/final_official_val_7000/figures/mlp_refresh_age_comparison.png](../../experiments/budget_free/value_v1/final_official_val_7000/figures/mlp_refresh_age_comparison.png)

仅 seed0，统计稳定性 `NOT_TESTED`；没有依据 val 选择 epoch、阈值、结构、窗口或重新拟合。教师在 train 上训练过，训练内诊断不能声称整个系统 unseen 泛化。有限窗口、sender 部分可观测、车端隐状态与未来运动、单query边际收益和联合决策非线性、GT离线代理与AMOTA不等价、无硬带宽保证、标签与部署 continuation 差异均限制解释。即使H8某项更好，也不能单独归因为去掉预算输入；本轮同时改变监督和有限窗口。

## 实际结论

相对旧 B1，H8 平均通信减少 6.28%，mAP 差值 -0.000071（数值接近持平），AMOTA 差值 -0.011364、MOTA 差值 -0.014708，IDS 减少57。较少通信与较少 ID switches 同时伴随 AMOTA/MOTA 下降，是混合结果，完整 val 没有证明相对 B1 的整体跟踪优势。

H1 的 optional UPDATE 仅 37/42772（0.09%），行为接近 INIT_ONLY；相对旧 B1 平均通信减少 18.44%，但 mAP 差值 -0.009020、AMOTA 差值 -0.035851、MOTA 差值 -0.031050，IDS 增加18。这里的低通信伴随任务性能下降，不能称为全面胜出。

在本次 H8/H1 同协议比较中，H8 多付 319.903498 B/frame（14.91%），mAP 增加0.008950、AMOTA 增加0.024486、MOTA 增加0.016342，IDS 减少75。因此本次测量支持 H8 以更多通信换取这些任务指标改善这一取舍；单 seed 和有限窗口限制仍在，不构成统计显著或去掉预算输入的因果证明。相对旧参考的判断仍严格属于跨协议描述性比较。

原160000停止、历史保护FAIL以及全部不利结果继续保留；没有据此改 epsilon、学习率、seed、阈值或训练方案。
