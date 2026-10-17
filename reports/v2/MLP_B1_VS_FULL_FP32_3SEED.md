# MLP B1 vs FULL FP32：三个推理 seed

**STATUS=PASS；VERDICT=CONSISTENT_POSITIVE_INFERENCE_SEEDS**

三个配对 ΔAMOTA 均为正，平均差值 **+0.008820096142**，样本标准差 **0.002910548487**，wins **3/3**。这支持固定路侧输入下这三个本车推理 seed 的数值一致性，不构成统计显著性结论。平均 ΔMOTA 为 **+0.006548718677**，平均 ΔmAP 为 **−0.000818341122**。

完整 case：6/6。每个 case 单独 Python 进程，21 个官方 val scene / 3316 帧，冻结官方 R50 coop 与 V2 MLP epoch-12 权重。

| seed | FULL_FP32 AMOTA | MLP_B1 AMOTA | ΔAMOTA | FULL IDS | MLP IDS | FULL B/frame | MLP B/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.362201595 | 0.369315594 | 0.007114000 | 321.000000000 | 274.000000000 | 42355.845597105 | 2630.828106152 |
| 2025 | 0.350805740 | 0.362986517 | 0.012180776 | 258.000000000 | 230.000000000 | 42355.845597105 | 2630.828106152 |
| 2026 | 0.356772032 | 0.363937544 | 0.007165512 | 252.000000000 | 222.000000000 | 42355.845597105 | 2630.828106152 |

标准差采用样本标准差（ddof=1）；未完成六个可比 case 时不汇总均值或胜率。

| 指标 | mean | sample SD | wins |
|---|---:|---:|---:|
| FULL_FP32_AMOTA | 0.356593122279 | 0.005700033399 | — |
| V2_MLP_B1_AMOTA | 0.365413218421 | 0.003412845706 | — |
| delta_AMOTA | 0.008820096142 | 0.002910548487 | 3 |
| FULL_FP32_mAP | 0.362166302273 | 0.000878325309 | — |
| V2_MLP_B1_mAP | 0.361347961150 | 0.001418747065 | — |
| delta_mAP | -0.000818341122 | 0.000618601647 | 0 |
| FULL_FP32_MOTA | 0.288534413339 | 0.000788282850 | — |
| V2_MLP_B1_MOTA | 0.295083132016 | 0.003382861875 | — |
| delta_MOTA | 0.006548718677 | 0.003174871659 | 3 |

| 方法 | seed | mAP | AMOTA | AMOTP | MOTA | IDS | FP | FN | recall | mean B/frame | P95 B/frame | 10Hz B/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_FP32 | 0 | 0.362957876 | 0.362201595 | 1.320378657 | 0.288759415 | 321.000000000 | 5405.000000000 | 14294.000000000 | 0.492184169 | 42355.845597105 | 75575.000000000 | 423558.455971049 |
| V2_MLP_B1 | 0 | 0.362306438 | 0.369315594 | 1.322706949 | 0.298955521 | 274.000000000 | 4485.000000000 | 14974.000000000 | 0.468026148 | 2630.828106152 | 4825.000000000 | 26308.281061520 |
| FULL_FP32 | 2025 | 0.361221413 | 0.350805740 | 1.320905270 | 0.287658093 | 258.000000000 | 4805.000000000 | 14988.000000000 | 0.467528776 | 42355.845597105 | 75575.000000000 | 423558.455971049 |
| V2_MLP_B1 | 2025 | 0.359718143 | 0.362986517 | 1.326637660 | 0.292702856 | 230.000000000 | 4044.000000000 | 15635.000000000 | 0.444543129 | 2630.828106152 | 4825.000000000 | 26308.281061520 |
| FULL_FP32 | 2026 | 0.362319617 | 0.356772032 | 1.322713952 | 0.289185733 | 252.000000000 | 4777.000000000 | 14979.000000000 | 0.467848515 | 42355.845597105 | 75575.000000000 | 423558.455971049 |
| V2_MLP_B1 | 2026 | 0.362019303 | 0.363937544 | 1.325743626 | 0.293591019 | 222.000000000 | 3993.000000000 | 15669.000000000 | 0.443335228 | 2630.828106152 | 4825.000000000 | 26308.281061520 |

RNG 重置：`after scheduler/model construction, checkpoint loading and iter(loader); before first next(iterator); no scene reseeding`。Python/NumPy/Torch CPU/CUDA 同步设 seed，PYTHONHASHSEED 在子进程启动前设置；逐 seed 配对 RNG digest：`PASS`。
数值设置：FP32、单卡、batch=1、workers=0；TF32 matmul/cuDNN=True，cuDNN deterministic/benchmark=False。继承 CUDA 算子可能非确定性。

固定同一份预生成路侧 query cache，只改变本车推理 seed；这不是三次训练，也不是全链路改变 seed。
历史 FULL 三 seed 使用构建前 seed、构建后不 reset、cuDNN deterministic=True，全部拒绝复用。本轮六个 case 全部重新推理。

通信按实际序列化 packet 长度统计。FULL 为 CTKFP321：当前三组 256 维 FP32 latent + 当前 ref_pts/pred_boxes，16 字节前缀/CRC 与 JSON 元数据；逐字段验证 decode 与原生输入完全相等。新明确封装不同于历史 CTKMSG01，因此不把历史 FULL 字节数回填。
MLP 为原 V2 APTU0001 INT8/cache/codec，0.5 门控，B1 名义 3474 B/frame，预算输入 3474/12840.189523169913；每帧当前几何。mean/P95 均来自真实 packet，10Hz B/s=mean×10。两者均为应用消息，排除 TCP/IP、无线开销及重传。

IDS、FP/FN、recall 与 MOTA 来自各自官方 evaluator 的 best-MOTA operating point；IDS 差异不直接等同于真实身份稳定性差异。
仅三个推理 seed，不作统计显著性、训练 seed 稳健性或新数据集泛化结论。FULL 与 MLP 同时改变量化、缓存和刷新，不能将正差值归因为“去除最新高维噪声”。

历史 seed0 审查（本轮新值减历史锚点）：
- FULL_FP32 mAP：new=0.362957876304，historical=0.362344799300，Δ=0.000613077004。
- FULL_FP32 AMOTA：new=0.362201594656，historical=0.356898250800，Δ=0.005303343856。
- V2_MLP_B1 mAP：new=0.362306438354，historical=0.362306438354，Δ=0.000000000000。
- V2_MLP_B1 AMOTA：new=0.369315594181，historical=0.369315594181，Δ=-0.000000000000。

本轮明确采用模型构建、权重加载和 DataLoader iterator 创建后 reset 的协议。历史诊断 FULL 三 seed 明确使用不同的 reset/cuDNN 设置；历史 Stage3 FULL seed0 锚点没有足以逐项证明相同 reset 位置的运行证据，故不复用、不回填，也不能证明全部差异只由 reset 引起。MLP seed0 的 mAP/AMOTA 复现历史值，给定锚点的末位差异仅为显示精度。

差异检查：本轮 3316 份 query cache 的 SHA/字节数全部匹配继承的 val cache manifest；六份 train/val infos 的 SHA 匹配继承数据准备记录；两个官方 checkpoint 与冻结 MLP 的 SHA 在执行前后相同。原生 `_forward_single_frame_inference` 的 AST 与历史导出的官方方法一致；该检查覆盖单帧推理方法，不代表已证明历史整个工作树相同。本轮整个原生工作树及数值环境在六个 case 间冻结一致。

保护核验：`PASS`。all files in named inherited directories, excluding this new robustness experiment and pycache; tracked files of both native repositories; datasets images/raw archives and other artifacts not rehashed

证据目录：`experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3`。`manifest.json` 冻结运行身份/命令/源码/配置/缓存 SHA；`metrics/*/run_manifest.json` 保留逐 case checkpoint、RNG digest 和评估前 raw SHA；原始预测保留为无损 `.pkl.gz`，官方 det/track summary/details 与逐帧通信单独保存。

冻结权重（所有 case 共享官方 coop；MLP 另加载以下 final；路侧 cache 保持预生成版本）：

| 文件 | SHA256 |
|---|---|
| checkpoints/cooptrack_r50_coop.pth | de543ce784175dd62f62d4e9d6b17c5ae7757f00548efbe49f8671e98ef0c4e2 |
| checkpoints/cooptrack_r50_inf.pth | a9ae9cecef72aa772666a1fb8aad4eb009d1f20ce1cb5b812de16e2945630f79 |
| experiments/v2/checkpoints/mlp/final.pth | d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c |

最终核验：**6 个 complete case / 19,896 帧**；六份压缩预测解压后的 SHA 与评估前 raw SHA 一致；tracking/detection JSON 各包含 3316 个 token，且与逐帧通信 token 集一致；报告指标等于官方 det/track summary；三组配对初始 RNG digest 一致。格式化 JSON 的额外 SHA 在最终归档核验时记录；评估前绑定使用原始 predictions.pkl SHA。

保护核验覆盖 4724 个继承文件（共 14,544,727,848 字节），包括旧 main/V2 模型、结果、配置、报告、完整 val query cache、infos 和两个原生仓库的已跟踪文件；运行前后文件列表、内容 SHA、仓库 commit/status 均一致。排除本次新实验目录、新报告和 pycache；原始图片/数据归档、其他 artifacts 未重新逐文件哈希。继承 artifacts/core、datasets、官方 checkpoints、data_infos 和 val query cache 的容器挂载保持只读。未训练、未创建新 checkpoint。

核验汇总与可复查证据：

- [配对 CSV](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/summary.csv)；[完整统计 JSON](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/summary.json)。
- [冻结 manifest](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/manifest.json)；[Docker 启动身份](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/docker_launch_provenance.json)。
- [原始预测/评估/通信归档核验](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/artifact_validation.json)；[可重跑的只读校验脚本](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/validate_artifacts.py)。
- [旧产物保护核验](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/protection_verification.json)；[继承 cache 身份核验](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/inherited_cache_provenance_check.json)；[单帧原生方法核验](../../experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/native_inference_method_audit.json)。
