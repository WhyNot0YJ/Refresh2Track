# 项目清理审计与逐项理由

> 后续执行：2026-10-13 已按用户指令清理 186 个已确认文件条目、42 个空目录和 11 个停止容器，详见 [执行记录](CLEANUP_EXECUTION_20261013.md)。本报告正文仍为 2026-10-12 的审计快照。空目录清单中 8 个 Git/正式数据结构目录的判断已纠正为保留。

2026-10-12，Asia/Shanghai。此次只读审计，没有删除、移动或压缩既有项目文件，没有停止训练。V4 正在训练 epoch 18，上一轮 V3 清理后的协调器交接已完成。

项目可直接清理的主要是旧 Example 视图/infos、Stage1 smoke 原始输出、下载 headers/log 和已结束的一次性交接脚本，合计 **186 个路径、100,830,055 B，约 96.16 MiB**。占空间更大的项目需要备份、无损压缩或修订恢复/保护绑定，不能把它们当作普通垃圾直接删除。

## 判断口径和覆盖范围

盘点到 **233,810 个文件条目和 933 个目录**；其中 25 个条目是通过 Docker Linux `readlink` 核验的链接。未跟随链接重复统计目标。文件逻辑字节合计 **62,714,957,943 B，约 58.41 GiB**；不是 NTFS 分配空间，也不是清理后可回收空间。

- **可清理**：已结束的开发/下载中间产物，不参与当前正式入口且不被当前保护清单锁定；理由同时说明清理后失去的旧功能。执行前仍需确认没有新增 writer。
- **条件清理**：有重复副本、可压缩或不进入当前训练，但仍涉及完整数据、正式研究证据、旧功能或恢复/保护绑定。没有满足列出的条件前，仍应保留。
- **保留**：当前输入/源码/权重/恢复状态，或支撑正式结果、数据来源与协议正确性的轻量证据。“保留”是按当前研究和复现需求的判断，不表示这些文件永远不能迁移。

逐项清单根据文件路径/大小、实际代码入口和导入、manifest/保护清单、阶段报告和抽查内容分类；不是把 23 万份图像、标签或缓存逐帧重新解码、语义阅读或重评。扫描期间 V4 继续写入，活动实验大小是盘点快照；重复内容核验仅覆盖已完整结束的验证目录。

| 清单 | 内容 |
|---|---|
| [files.csv.gz](../audits/cleanup_review_20261012/files.csv.gz) | 每一个文件/链接的路径、字节数、分类、理由、保护绑定及链接目标 |
| [directories.csv](../audits/cleanup_review_20261012/directories.csv) | 每一个目录的合计大小和子项判断；混合目录不能整删 |
| [safe_candidates.csv](../audits/cleanup_review_20261012/safe_candidates.csv) | 186 个可清理路径及逐项理由，不是已获执行授权的删除脚本 |
| [stage_files.csv](../audits/cleanup_review_20261012/stage_files.csv) | Stage1/2/3 的 53 个保留证据文件，逐文件解释 |
| [summary.json](../audits/cleanup_review_20261012/summary.json) | 分类统计和完整目录汇总 |

## 每个一级目录的结论

| 目录 | 文件条目 | 逻辑大小 | 判断 | 理由 |
|---|---:|---:|---|---|
| `archives/` | 0 | 0 B | 可清理空目录 | V3 整体归档已删除，此处为空。 |
| `artifacts/` | 922 | 6.43 GiB | 混合，不能整删 | core 身份、论文基线与阶段证据；部分大预测/诊断可压缩。 |
| `audits/` | 11 | 7.72 MiB | 保留 | Bug 审计、清理授权/执行记录和此次逐项清单。 |
| `checkpoints/` | 2 | 670.26 MiB | 保留 | 官方协同与路侧 R50 权重。 |
| `code/` | 999 | 57.35 MiB | 主要保留 | 实际原生 import、CUDA 算子与 Git 身份；仅旧 smoke 输出/Example 链接可清理。 |
| `configs/` | 15 | 2.22 MiB | 保留 | V4 与继承的预算参考/生成配置；旧名称仍有真实引用。 |
| `data/` | 182 | 2.99 GiB | 混合，不能整删 | Example 可清理；wheel/CUDA archive 是离线重建输入。 |
| `datasets/` | 220047 | 27.39 GiB | 主要保留 | raw 图像/标定支撑 prepared 链接；PCD 必须有可靠备份后单独决策。 |
| `data_infos/` | 12 | 2.05 GiB | 混合，不能整删 | Example 可清理；完整 cooperative/infra infos 当前必须保留。 |
| `docker/` | 39 | 416.57 KiB | 主要保留 | 精确环境重建和 gates；旧迁移配方须处理保护引用后再移出。 |
| `experiments/` | 707 | 12.48 GiB | 混合，不能整删 | V4 活动产物；V1/V2 正式结果，部分原始预测可压缩。 |
| `logs/` | 12 | 70.75 MiB | 混合，不能整删 | V4 正在写；main/v2 已结束日志可无损压缩。 |
| `patches/` | 1 | 801 B | 保留 | 4060 Ti 与旧 Torch 的 inverse 兼容补丁。 |
| `query_cache/` | 10762 | 6.26 GiB | 保留 | train/val 正式消息输入与 train-only 归一化参考。 |
| `reports/` | 53 | 9.90 MiB | 保留 | 阶段、正式结果、退役原因和完整事实记录。 |
| `scripts/` | 38 | 598.29 KiB | 主要保留 | V4 导入 main/v2 辅助模块；一次性维护脚本可在保护迁移后整理。 |

根目录的 `.gitignore`、README、研究提示词保留。`.gitignore` 只决定版本库忽略哪些文件，不能作为可删除判据。已废止的 V3/H 文档保留历史思路和原因；V4 提示词保留现行方法定义。`reports/EXPERIMENT_INDEX.md` 仍有 V2 主线的历史措辞，宜在另一次文档维护中显式更新现行状态，而不是删除报告；当前保护记录绑定了部分文档字节。

## 可以优先清理的具体内容

| 路径 | 快照大小 | 判断和理由 |
|---|---:|---|
| `data/V2X-Seq-SPD-Example/` | 12,852,499 B | 可清理。Stage1 Example 视图与 GT 表；V4 使用全量数据。13 个外部目标在当前 main 容器未挂载，不是现行输入。删除会放弃本地 Example 视图重跑能力；只删除链接本身，不跟随到项目外。 |
| `data_infos/V2X-Seq-SPD-Example/` | 59,277,539 B | 可清理。6 份 Example infos 不被正式 V4 路径或当前保护清单引用；保留完整 train/val infos 与 Stage1 摘要。 |
| `code/CoopTrack/datasets/V2X-Seq-SPD-Example` | 链接本身 | 可清理。仅为旧 Example 的源码树入口，当前正式配置显式指定完整数据路径。 |
| `code/CoopTrack/output/results.pkl` | 6,772,255 B | 可清理。旧 Example 推理输出，Stage1 真实指标已保存在 `official_example_eval.json`。 |
| `code/CoopTrack/test/stage1_coop/.../results_nusc.json` 和 `results_nusc_det.json` | 21,862,772 B | 可清理。旧 smoke 的原始评测输入，不是正式论文预测；保留该目录的轻量 det/track metrics 和 Stage1 报告。删除后不能从本地原件重评这次 smoke。 |
| `data/wheels/torch-cu111-parts/*.headers` 和 `*.log` | 61,766 B | 可清理。120 个下载中间记录；61 份分块 payload 早已清理，完整 wheel 与下载 manifest 保留。这里没有再藏着一份 2 GB wheel。 |
| `scripts/maintenance/v4_retirement_handoff.py` | 3,224 B | 可清理。交接状态已为 `COORDINATOR_RESTARTED`，当前没有该 watcher；保留 V4 审计 JSON，训练导入不需要一次性交接脚本。 |
| 空 `archives/` 和清单中的其他真正空目录 | 0 B | 可清理，主要减少杂乱，不释放文件空间。活动 V4 的预留/输出目录仍保留。 |

## Stage1 Stage2 Stage3 的逐文件判断

三个阶段的原工作目录和旧执行脚本已在此前整理中清理，当前留下的是 `artifacts/core/stage1/2/3` 的精选证据以及最终报告。合计 **78,469,025 B，约 74.83 MiB**，不是主要空间占用。阶段结束和证据失效是两回事。

Stage1 可缩减逐帧 cache/tensor/profile 细节，保留 Example 验收、资源来源、环境迁移和最终 gate。Stage2 应保留原始数据、官方修正、排除 ledger、完整 train/val 验收与转换命令。Stage3 应保留消息无损回归、wire 口径、梯度探针及聚合时序结论，逐对时序日志可压缩。

### Stage1 证据

15 个文件，9.07 MiB。

| 文件（相对该 stage 目录） | 大小 | 判断 | 理由 |
|---|---:|---|---|
| `cache_inf.json` | 2.73 MiB | 条件清理 | Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。 |
| `checkpoint_download_resources.json` | 3.76 KiB | 保留 | 官方 checkpoint 下载 URL/hash 与作者来源的小型清单，保留以证明权重来源。 |
| `converter_gate.json` | 203.18 KiB | 保留 | Example converter 与 GT 表验收；保留小型正式验收记录，避免 Stage1 报告失去转换证据。 |
| `coop_coop.json` | 72.36 KiB | 条件清理 | Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。 |
| `docker_cleanup_verification.json` | 885 B | 保留 | 早期环境迁移/清理结果，仅 885 B；保留用于解释宿主 prefix 已删除和 Docker 环境继承。 |
| `docker_dataset_mount_probe.json` | 9.74 KiB | 保留 | 旧 smoke 的数据挂载诊断，约 10 KiB；体积小，保留环境验收上下文。 |
| `host_installation_cleanup_review.json` | 4.36 KiB | 保留 | 宿主安装范围核验，证明未污染其他环境；小型历史记录保留。 |
| `inf_cache_all.json` | 4.70 MiB | 条件清理 | Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。 |
| `inf_first_two.json` | 68.80 KiB | 条件清理 | Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。 |
| `local_cleanup_manifest.json` | 292.57 KiB | 保留 | 宿主 prefix/cache 清理账本，仍被 V4 保护清单锁定，保留。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `official_example_eval.json` | 8.20 KiB | 保留 | 249 帧 Example 真实 evaluator 指标、命令与内存，替代大 smoke 预测的轻量验收摘要。 |
| `official_example_eval_memory.json` | 285 B | 保留 | Example 官方评测进程峰值内存，285 B；保留 4060 Ti 环境验收证据。 |
| `profile_coop.json` | 885.12 KiB | 条件清理 | Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。 |
| `resource_download_manifest.json` | 119.09 KiB | 保留 | 依赖、CUDA header 和官方资源来源及哈希清单，独立重建/溯源需要，保留。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `stage1_final_gate_checks.json` | 5.00 KiB | 保留 | Stage1 最终 PASS 条目和证据索引；保留阶段结论，不作为过时状态文件删除。 |

### Stage2 证据

24 个文件，42.90 MiB。

| 文件（相对该 stage 目录） | 大小 | 判断 | 理由 |
|---|---:|---|---|
| `stage2_gate_checks.json` | 18.62 KiB | 保留 | FULL_DATA_READY 最终验收与未运行阶段边界；保留数据准备结论。 |
| `data/archives_manifest.json` | 18.65 KiB | 保留 | 原始归档、分组、来源和哈希清单；后续备份和原始数据恢复需要。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/archive_hashes.json` | 4.44 KiB | 保留 | 本地原始归档哈希，小型溯源记录，删除无明显空间收益。 |
| `data/archive_integrity.json` | 13.51 KiB | 保留 | 分卷载荷 CRC、Headers Error 与独立 payload 校验结论；正式 raw 数据来源验收证据。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/cache_dependencies.json` | 2.07 MiB | 条件清理 | CPU shape/cache 计划或未要求点云 bin 的逐路径清单不驱动现行 V4；保留完整性和排除账本后可压缩/移出细节，历史重核会受限。 |
| `data/context_runtime_stdout_diagnostic.json` | 1.71 KiB | 保留 | 数据编排 JSON 解析失败的原始诊断；仅 1.7 KiB，保留以解释当时修复和边界。 |
| `data/conversion_completion.json` | 14.74 KiB | 保留 | 六次官方转换完成结果；保留正式 infos 的生成账本。 |
| `data/conversion_counts.json` | 17.07 KiB | 保留 | 三侧转换计数；保留 train/val 规模和排除范围的核对依据。 |
| `data/cpu_pipeline_probe.json` | 600.93 KiB | 条件清理 | CPU shape/cache 计划或未要求点云 bin 的逐路径清单不驱动现行 V4；保留完整性和排除账本后可压缩/移出细节，历史重核会受限。 |
| `data/excluded_records.jsonl` | 569.09 KiB | 保留 | 官方 test/invalid/coop 配对排除逐条原因；证明未任意删帧，必须保留，建议压缩而不删除内容。 |
| `data/full_trainval_integrity.json` | 19.26 KiB | 保留 | 46 train / 21 val、infos、GT 表与无交集核验；是完整数据验收依据。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/image_decode.json` | 860 B | 保留 | 23527 张 JPEG 的真实解码验收摘要，保留。 |
| `data/image_decode_manifest.jsonl` | 4.45 MiB | 保留 | 全图像解码逐帧账本及 SHA；仍受保护绑定，保留，压缩须另行迁移引用。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/label_processing_changes.jsonl` | 17.44 MiB | 保留 | 官方标签修正和 track_id 去重的逐条 ledger，prepared_integrity 记录其 SHA；保留原始修正证据，可规划无损压缩。 |
| `data/nonrequired_paths.jsonl` | 17.61 MiB | 条件清理 | CPU shape/cache 计划或未要求点云 bin 的逐路径清单不驱动现行 V4；保留完整性和排除账本后可压缩/移出细节，历史重核会受限。 |
| `data/official_selection.json` | 17.97 KiB | 保留 | 67 个非 test scene 的官方选集信息；保留以证明没有扩大/缩小正式 split。 |
| `data/prepared_completion.json` | 4.77 KiB | 条件清理 | 与 artifacts/core/stage2/data/prepared_integrity.json 的完整 SHA256 一致；保留后者，先迁移报告/验收引用后可删本副本。 |
| `data/prepared_integrity.json` | 4.77 KiB | 保留 | prepared 标签修正/元数据身份的权威验收版本，保留；与 prepared_completion 逐字节相同。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/raw_integrity.json` | 692 B | 保留 | 完整 raw 来源与布局验收，仅 692 B；保留。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `data/storage_plan.json` | 1.72 KiB | 保留 | 归档/解压空间的小型历史账本，仅 1.8 KiB；保留数据准备来源。 |
| `manifests/paths_inherited.json` | 2.21 KiB | 保留 | Stage1 到 Stage2 的历史路径继承记录；保留溯源，旧路径不表示目前仍可使用。 |
| `manifests/prepare_data_execution.json` | 20.15 KiB | 保留 | 官方转换的真实命令、退出状态和数据生成执行 ledger；保留。 |
| `manifests/stage2_context.json` | 19.26 KiB | 保留 | 继承环境、模型与数据身份的小型阶段上下文；保留。 |
| `manifests/train_smoke_manifest.json` | 10.29 KiB | 保留 | 从完整 train 派生的 scene0000/32 帧梯度探针映射；仍受保护绑定，保留，避免混入 val/截断监督。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |

### Stage3 证据

14 个文件，22.87 MiB。

| 文件（相对该 stage 目录） | 大小 | 判断 | 理由 |
|---|---:|---|---|
| `cache_train_mapping.json` | 45.87 KiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `execution.json` | 32.99 KiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `fp32_regression.json` | 876 B | 保留 | 完整 val 的 FP32 消息无损回归验收，证明 codec 未改变基线，保留。 |
| `plan.json` | 1.30 KiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `stage3_gate_checks.json` | 1013 B | 保留 | Stage3 最终诊断 PASS 权威小型汇总；保留，与 state.json 字节一致。 |
| `state.json` | 1013 B | 条件清理 | 与 artifacts/core/stage3/stage3_gate_checks.json 的完整 SHA256 一致；保留后者，先迁移报告/验收引用后可删本副本。 |
| `temporal.json` | 2.59 KiB | 保留 | 42772 对 Track Query 时序冗余的聚合统计；保留方法动机，明示不能由此证明跳过传输无损。 |
| `temporal_pairs.jsonl` | 20.78 MiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `train_receiver_audit.json` | 9.08 KiB | 保留 | 32 帧真实 train receiver 与 sender 映射核对，保留。 |
| `wire_audit.json` | 2.85 KiB | 保留 | 五个接收字段、字节口径、真实 FP32 roundtrip 与未发送 GT 的协议验收；保留。 |
| `wire_audit_manifest.jsonl` | 1.85 MiB | 保留 | 3316 val+32 train 的逐帧无损消息账本，wire_audit 锁定其 SHA，当前 V4 保护记录也锁定；保留。 当前还被 V4 protected_shared_files 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。 |
| `configs/coop_val.py` | 31.34 KiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `configs/inf_val.py` | 23.31 KiB | 条件清理 | 已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。 |
| `train_probe/result.json` | 99.39 KiB | 保留 | 真实 train32 原生 task loss 到接口梯度、冻结参数及显存探针结果，是后续训练可行性证据，保留。 |


[stage_duplicate_checks.json](../audits/cleanup_review_20261012/stage_duplicate_checks.json) 已实际读取完整文件并确认：`prepared_completion.json` 与 `prepared_integrity.json` 的 SHA256 均为 `e944b8634733e27cd61ceea44918693fd893295df461b14030defe1ff3eb0562`；`state.json` 与 `stage3_gate_checks.json` 的 SHA256 均为 `039e235ff3f12585a8d9fc1206ddc2b6a31f4a956f5c5c9adf5276de98950843`。保留权威后者，先迁移报告/验收引用后可合并前者；合计仅节约 5,901 B。

`reports/STAGE1_FINAL.md`、`STAGE2_FINAL.md`、`STAGE3_FINAL.md` 均建议保留，体积小且记录各阶段的范围、失败与真实验收结论。不能用 V4 的新协议回写或删除旧阶段不同口径的结果。

## 占空间较大但有条件的项目

| 路径或文件组 | 快照大小 | 为什么不能直接删，以及清理条件 |
|---|---:|---|
| raw 的 `vehicle-side/velodyne/*.pcd` 与 `infrastructure-side/velodyne/*.pcd` | 22,035,811,126 B，约 20.52 GiB | 当前 camera-only pipeline 不读取点云，但这是完整原始数据，prepared 的 velodyne 链接和 raw 验收引用它。先验证外部完整归档/备份及重建能力，再明确缩减本地 raw 的复现范围；本次未对项目外原始归档重新核验。不能删除整个 raw，也不能把 calibration JSON 混入点云删除范围。 |
| `experiments/main/predictions/` 中的 pkl 与 results JSON | 大文件约 6.24 GiB | V1 正式原始预测。建议无损压缩，核验解压后原 SHA，并维护重评/恢复引用；保留所有方法指标、通信和最终权重。删除整个目录会丢正式证据。 |
| `artifacts/core/communication_baselines/` 原始预测 | 整目录约 3.76 GiB | FULL FP32/FP16/INT8、CONF 与 Random 结果不是临时 smoke。可压缩预测，保留 `result.json` 和逐帧通信；FULL INT8 与 R50 car-only 对齐仍被 V4 基线引用。 |
| `artifacts/core/ids_diagnosis/` 大预测与 JSONL | 整目录约 2.58 GiB | 解释原生 RNG、seed 和 ID 生命周期差异。可 gzip 保留原字节及摘要、patch 对照和输入身份；不能仅因已得出结论就删除所有重核依据。 |
| V4 epoch003/006/009/012/015 的原文件副本 | 1,859,408,861 B，约 1.73 GiB | 15 个 raw 文件与各自 `.gz` 完整解压 SHA256 一致，详见 [verified_duplicates.json](../audits/cleanup_review_20261012/verified_duplicates.json)。但是 `evaluate.py` 已完成验证的恢复分支仍同时强制检查 raw 和 archive：当前直接删 raw 会使恢复失败。待 V4 全部完成且保留可恢复方案，或先显式迁移恢复读取/源码维护绑定后，再删重复 raw；不动活动验证目录。 |
| `data/wheels/` 的 3 份完整 wheel | 2,342,475,151 B，约 2.18 GiB | 当前模型不再安装，但独立重建脚本明确需要 Torch/Open3D/ninja 精确包。可迁到可靠外部备份并调整配方，或接受未来联网重新下载且核验锁定 hash；未满足前保留。 |
| `data/cuda_packages/` 的 3 份 archive | 859,962,077 B，约 0.80 GiB | 重建只提取已锁定 CUDA headers，但仍依赖原归档。迁移备份或准备经过验证的等价重建输入后才能删，不能按旧 Stage1 下载缓存直接删。 |
| `data_infos/.../vehicle-side/` | 509,063,949 B，约 485.48 MiB | 当前 V4 主要读取 cooperative 与 infrastructure infos；vehicle-only infos 可在放弃/迁移该侧独立基线与完整三侧转换重核后清理。cooperative/infrastructure infos 必须保留。 |
| V1 `epoch_01` 至 `epoch_11` 权重 | 53.38 MiB | V1 已完成，早期权重可在保留 epoch12/latest、训练历史后删除，但它们被当前 V4 的共享保护记录锁定，必须先迁移保护绑定。 |
| `logs/main/` 和 `logs/v2/` | 67.61 MiB | 已结束实验日志可无损压缩。当前 `logs/v4_budget_free/` 仍在写入，应保留。 |
| 旧维护脚本与 `docker/build-image.sh` 等早期配方 | 几十 KiB | 多数旧任务已完成；旧 build 脚本依赖已移除的宿主 prefix，现行独立重建使用 `Dockerfile.rebuild`。可迁到轻量历史归档，但部分仍被保护清单或报告绑定，不能直接整删 `scripts`/`docker`。 |

上表的大目录数字包含应保留的小型指标，不能把目录总量当作可回收量。条件清理项之间的具体候选已在逐文件清单中排他分类；不会因为某个文件出现在多个讨论范围就重复计入分类统计。压缩只能回收原文件与新压缩文件的大小差，不是整个原文件大小。

## 当前需要保留的关键依赖

| 依赖 | 具体理由 |
|---|---|
| `datasets/V2X-Seq-SPD-raw` 的图像/标定/元数据与 `datasets/prepared/...` | 已核验 9 个 prepared 链接实际指向 `/data/spd_raw`，且当前容器目标存在。raw/prepared 不是两份完整图片副本；删除 raw 图像/标定会让 prepared 输入失效。 |
| `query_cache/train` 和 `query_cache/val` | 合计约 6.26 GiB；train 每帧 pkl 是正式 scheduler 输入，val 是 V4/Periodic 的一致对照输入。它们是已锁定的实验资产，不是可随意再生成的临时缓存。 |
| `checkpoints/cooptrack_r50_coop.pth` 和 `cooptrack_r50_inf.pth` | V4 仍加载原生冻结模型；两个官方权重是协同/路侧模型与缓存身份的基准。 |
| `code/CoopTrack`、`code/mmdetection3d` 及其 `.git` | 运行时实际 `mmdet3d.__file__` 位于工作区源码，三个抽查 `.so` 也从源码树内加载；不是 Docker 已经安装后就可删除。HEAD 是 merge/resume 验收的一部分，`.git` 必须保留。旧 build `.o/.so` 副本已在上次清理中删除。 |
| `scripts/main_*`、`scripts/v2_*`、`scripts/v4_budget_free` | V4 直接复用旧 runtime、training、serializer/cache 与 STE；文件名带 V1/V2 不代表仅旧版本使用。 |
| V4 `manifest.json`、`run.lock`、checkpoints、audit | 当前训练恢复、选模、源码散列映射和合并身份依赖它们。`v3_retirement_transition.json` 属于 V4 当前身份依赖，不能因名字含 V3 再删掉。 |
| V2 最终权重、metrics、communication、robustness、ablations | 历史对照和三 seed/消融完整证据；V4 `references.py` 直接读取多个 V2 点。正式预测已 gzip，整删会失去复评能力。 |
| `artifacts/core/{r50_alignment,full_data_ready,full_protocol,checkpoint_manifest,val_cache_manifest,...}` | 真实数据规模、作者权重来源、cache 身份和 car-only 指标口径的权威来源；不能用新的 V4 表格替代这些输入来源记录。 |
| `docker/locks`、重建脚本、兼容 patch、probes、package `.egg-info` | 精确环境和 editable 安装身份需要。`.so`、`.egg-info` 虽被 Git 忽略，也可能是当前运行依赖。 |
| 阶段最终报告、Bug 审计、H/V3 退役文档 | 占用小、解释研究结论及已主动放弃的复现能力；保留历史事实和授权清理的来源。 |

## Docker 中的旧阶段容器

以下 11 个容器均已停止，可在保存 inspect/所需日志后删除容器本身：

| 已停止容器 | 独立写入层字节 | 理由 |
|---|---:|---|
| `cooptrack-ids-cpu` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-ids-gpu` | 65649 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-r50-alignment-cpu3` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-r50-alignment-cpu2` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-r50-alignment-cpu` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage3-gpu` | 65649 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage2-ready` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage2-data` | 42385 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage2-prepare` | 0 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage2-extract` | 65649 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |
| `cooptrack-stage1-smoke` | 65649 | 原阶段已结束，正式产物保存在 bind mount/工作区；共享镜像仍由 main 使用。 |

[docker_inventory.json](../audits/cleanup_review_20261012/docker_inventory.json) 保存本次实际 inspect 和挂载。11 个容器的独立写入层合计 **516,906 B，约 0.49 MiB**，未计独立容器日志；它们共用约 7.94 GB 的镜像层，不能把该镜像大小乘以容器数量计为可回收空间。bind mount 的数据仍属于工作区，不是容器独有数据。

**`cooptrack-main` 和 `cooptrack-stage1:29f1c52-official` 镜像必须保留**，该镜像 ID 仍与运行容器/入口要求一致，名字保留了早期 Stage1 命名。当前 `docker system df` 的其他可回收镜像/容器涉及本机其他项目，不在此任务的可删除范围内；此时 build cache 显示 0 B，不能沿用之前已清理的 14.71 GB 数字。

## 核验结果和建议顺序

逐文件分类总数与原始枚举一致，每条都有理由；所有被当前保护清单或冻结源码绑定的文件均未列为“可直接清理”，活动 V4 实验也没有直接删除候选。实际检查了 25 个链接、15 对 raw/gzip 副本及 2 对阶段汇总，gzip 均完整读取到 EOF 并核对归档 SHA、原字节 SHA 和长度。没有全量重新哈希图片、PCD、GT 或 query cache，也没有重新跑训练/官方评测。

建议先处理约 96 MiB 的旧 Example/smoke/下载记录和空目录，主要让结构清楚；随后无损压缩 V1 与老基线/IDS 大预测。V4 验证副本等当前训练结束或恢复校验迁移后处理。若需要回收十几 GiB，再单独决定原始点云的外部备份与本地复现范围。Stage1/2/3 的小型验收证据优先保留，而不是为几十 MiB 丢失数据和协议来源。

本报告的“可清理”是具体审计结论；本次删除数为 **0**。
