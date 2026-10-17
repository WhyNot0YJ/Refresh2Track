# 无预算刷新收益监督 VALUE_V1

状态：`TEACHER_LIMIT_REACHED`；阶段：`official_val`。报告时间：2026-10-10 00:30:44 +08:00。

All new teacher calls reached the fixed 160000 frame-forward cap

这是根据旧 V2 结果提出的后续探索。Pilot、D0=4096、D1=2048和三次40轮训练已完成；原160000上限run的完整official-val未完成。H8只保存1816/3316帧partial，H1为 `NOT_RUN`，正式指标均为N/A。未用新val选择epoch、阈值、结构或窗口。本任务不删除旧文件、不修改旧冻结成果；核心正式结果保留，整个工作区快照仍有下文列明的保护FAIL。

## 旧成果保护与入口

初始清单：233542 个文件、1050 个目录、27 个链接，合计 234619 条；选定内容 SHA 539 个。扫描不跟随 junction/symlink，不全量重哈希原始数据。

最终全工作区保护核验：`RAW_PRESERVATION_FAIL`（原始检查状态 `FAIL`）。检查539份选定旧文件内容，536份未变、3份变化；缺失348条旧路径＝116个文件＋232个目录。范围是选定SHA和既有非链接路径存在性，不是所有原始数据内容或链接目标重新认证。`protection_after.old_files_deleted=348` 在原实现中包含目录，不能当成348个文件。

本任务 `old_files_deleted=0`、`old_frozen_files_modified=0` 与全工作区偏离分别报告。缺失集合与既有清理清单完全一致：116个文件是47份构建重复/中间物、61份wheel下载分片、1份下载探针、7份旧pyc；232个目录是191个空evaluator输出目录、40个构建目录、1个bytecode目录。没有旧正式模型、指标、预测或label文件缺失。3份变化为 `scripts/refresh_ablation_protection.py`、`scripts/refresh_ablation_report.py`、`reports/PROJECT_STRUCTURE_CLEANUP.md`。另行清理报告及路径匹配提供变动背景，具体writer身份未独立证明；未恢复、覆盖或重写原基线。

| 类别 | 实际入口/产物 | 有范围的终端核验 |
| --- | --- | --- |
| Stage1/2/3、R50、IDS | `artifacts/core/`、原阶段报告 | 对应选定报告未变；目录快照仍有空图目录偏离 |
| V1 | `experiments/main/`、`reports/MAIN_EXPERIMENT_FINAL.md` | 选定核心SHA225/225未变，结果文件无缺失；36个空plots目录及5个旧pyc缺失 |
| V2 | `experiments/v2/`、`reports/v2/V2_FINAL.md` | 选定核心SHA142/142未变，结果文件无缺失；32个空plots目录及2个旧pyc缺失 |
| Bug审计 | `audits/v2_bug_audit/`、`reports/v2/V2_BUG_AUDIT.md` | 9/9选定SHA未变；清理报告的变化单独列明 |
| 已有消融 | `experiments/v2/ablations/refresh_2x2/` | 旧结果文件无缺失；推理入口/配置未变，两个辅助源码变化；运行记录属原并发可变范围 |
| 本轮 | `scripts/budget_free/run.py`、`configs/budget_free/value_v1.json`、`experiments/budget_free/value_v1/` | 原160000上限run为TEACHER_LIMIT_REACHED，全部新标签/模型/partial与失败证据保留 |

核心未变限定为选定SHA与结果路径核验，不声称每个历史raw payload重新哈希。终端原始保护[experiments/budget_free/value_v1/evidence/protection_after.json](../../experiments/budget_free/value_v1/evidence/protection_after.json)、终端执行[experiments/budget_free/value_v1/evidence/terminal_execution_verification.json](../../experiments/budget_free/value_v1/evidence/terminal_execution_verification.json)、变动明细[experiments/budget_free/value_v1/evidence/external_preservation_changes.json](../../experiments/budget_free/value_v1/evidence/external_preservation_changes.json)可复核上述范围。

索引：[reports/EXPERIMENT_INDEX.md](../../reports/EXPERIMENT_INDEX.md)。状态：[experiments/budget_free/value_v1/state.json](../../experiments/budget_free/value_v1/state.json)。保护证据：[experiments/budget_free/value_v1/evidence/protection_before.json](../../experiments/budget_free/value_v1/evidence/protection_before.json)、[experiments/budget_free/value_v1/evidence/protection_after.json](../../experiments/budget_free/value_v1/evidence/protection_after.json)。

外部并发范围：experiments/v2/ablations/refresh_2x2: pre-existing C evaluation is running; do not modify or stop。实际启动时进程见 state.json；不能把该子树新增日志/结果归因于本轮。

## 方法、训练协议与信息边界

学生只看当前三个 INT8 roundtrip latent、接收端实际解码镜像、差值，以及 elapsed、当前/变化 confidence、ref_pts/变化、pred_boxes、track_age，共 `3×768+20=2324` 维。ID/scene/session 仅用于身份和缓存索引。不输入 B1–B4、GT、未来、车端私有状态、数值 ID embedding。

固定结构：三个 768 维块共用 `Linear(768,64)+ReLU`，拼接 20 维状态后 `212→128→64→1`，两层 ReLU，无 Dropout/BatchNorm。推理单次 MLP，sigmoid≥0.5 刷新；只缺缓存强制初始化。几何每帧发送当前 FP32，latent bundle UPDATE 增加 780 B。无 rate loss、TTL、Top-K、冷却期或硬带宽保证。

教师为冻结 CoopTrack 的真实 `eval()/return_loss=False`、learned affinity→Hungarian→fusion。U/R 从相同完整前缀和 Python/NumPy/CPU/CUDA RNG 恢复，仅候选 t 帧动作不同；候选后七帧 REUSE，其他 ID 使用固定参考日程，新 ID 正常初始化，消失不伪造，不跨 session。H8 为 t…t+7 八帧；H1 直接使用同一分支首帧。在线部署不锁定后七帧，存在标签—部署 continuation 差异。

GT 只在 official train 离线输出评分器使用。共同匹配历史、car/global 坐标、正式输出阈值 0.30，2 m 地面中心距离门限（实际继承严格 `<2 m` 边界）。`E=FN+FP+IDS+0.1×Σmin(distance/2,1)`；`ΔH=JH(REUSE)−JH(UPDATE)`。`epsilon=max(0.01,10×P99(|null Δ|))`，`Δ>epsilon` 才标 UPDATE；round0 锁定尺度、截断 margin 权重和固定权重均值下的 raw-logit BCE。空 GT 仍计 FP。此代理不是 AMOTA；sigmoid 不是物理收益或自然频率校准概率。

只用全部 official train：46 scenes/7362 cooperative samples。参考 roll-in P8/P32/INIT_ONLY；一轮自身 H8 状态采样；最终 H8/H1 同一数据行、归一化和初始化规则，从头训练。配置 40 epochs、batch128、CPU FP32、seed0、AdamW peak1e−3/min1e−5、weight decay1e−4、clip5，不运行原生 task backward。配置次数与已执行次数在下文分别列出。

协议及本地身份：[experiments/budget_free/value_v1/protocol.json](../../experiments/budget_free/value_v1/protocol.json)、[experiments/budget_free/value_v1/evidence/native_identity.json](../../experiments/budget_free/value_v1/evidence/native_identity.json)、[experiments/budget_free/value_v1/evidence/label_source_identity.json](../../experiments/budget_free/value_v1/evidence/label_source_identity.json)。

## 正确性与固定收益容忍度

| 检查 | 实际状态 |
| --- | --- |
| 整体 | PASS |
| null 重复 | PASS |
| U/R 顺序交换 | PASS |
| 特权信息隔离 | PASS |
| 可变缓存独立 | PASS |
| 未发送 latent 强扰动 | PASS |
| 实际 INT8 解码 | PASS |
| 当前几何 | PASS |
| 冻结参数/BN | PASS |
| 候选至少通过一帧native XY输入mask | True |

epsilon H8=0.010000；H1=0.010000。null ΔH8=[0.0, 0.0]，ΔH1=[0.0, 0.0]；配对窗口 payload 差=780 B。

附加固定 seed1：primary Δ=[0.00030294877729475367, 0.0]；secondary Δ=[-1.3091912632878788e-05, 0.0]。仅敏感性诊断，不择优替换标签。

完整原始证据：[experiments/budget_free/value_v1/evidence/correctness.json](../../experiments/budget_free/value_v1/evidence/correctness.json)、[experiments/budget_free/value_v1/evidence/correctness_branch_records.json](../../experiments/budget_free/value_v1/evidence/correctness_branch_records.json)、[experiments/budget_free/value_v1/evidence/constructed_checks.json](../../experiments/budget_free/value_v1/evidence/constructed_checks.json)。受控片段通过不能扩大为所有时序状态均已证明。

## Pilot、D0、D1 实际标签与成本

| 集合 | 状态 | 配对数 | scene数 | H8 UPDATE数/比例 | H1 UPDATE数/比例 | XY mask至少1帧/0帧样本 |
| --- | --- | --- | --- | --- | --- | --- |
| Pilot | PASS | 1024 | 46 | 69 / 6.74% | 11 / 1.07% | 634 / 390 |
| D0 | 完成 | 4096 | 46 | 300 / 7.32% | 43 / 1.05% | 2703 / 1393 |
| D1 | 完成 | 2048 | 46 | 170 / 8.30% | 26 / 1.27% | 1340 / 708 |

Pilot使用D0共享样本，不把Pilot+D0相加为独立新增样本。D0、D1均已consolidate。`effective_infra_query_count` 是八帧窗口内候选通过native XY几何输入mask的帧数（0…8）；表中分别统计至少1帧和0帧样本。它不等于Hungarian实际匹配/融合次数，也不保证任务收益；0帧样本保留，未按此诊断筛掉。

| 集合/H | min | P25 | P50 | P75 | P95 | P99 | max | 负Δ比例 | \|Δ\|≤epsilon比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pilot/H8 | -0.867123 | -0.000003 | 0.000000 | 0.000026 | 0.124969 | 0.598572 | 2.500326 | 26.66% | 89.65% |
| Pilot/H1 | -1.001186 | 0.000000 | 0.000000 | 0.000000 | 0.000742 | 0.010488 | 1.000000 | 17.19% | 98.54% |
| D0/H8 | -2.624718 | -0.000006 | 0.000000 | 0.000044 | 0.125047 | 0.625008 | 2.500326 | 28.20% | 88.35% |
| D0/H1 | -1.001186 | 0.000000 | 0.000000 | 0.000000 | 0.001269 | 0.011257 | 1.051160 | 16.92% | 98.66% |
| D1/H8 | -2.624986 | -0.000012 | 0.000000 | 0.000083 | 0.128913 | 0.893996 | 1.753530 | 28.42% | 85.35% |
| D1/H1 | -1.903759 | 0.000000 | 0.000000 | 0.000000 | 0.001880 | 0.013044 | 1.971625 | 17.14% | 97.41% |

Pilot scene 分布：0000:23, 0001:23, 0002:23, 0004:23, 0005:23, 0008:23, 0010:23, 0016:23, 0018:23, 0022:23, 0023:23, 0025:23, 0029:22, 0030:22, 0032:22, 0033:22, 0034:22, 0035:22, 0036:22, 0037:22, 0040:22, 0041:22, 0047:22, 0048:22, 0049:22, 0050:22, 0054:22, 0055:22, 0056:22, 0057:22, 0060:22, 0062:22, 0066:22, 0068:22, 0070:22, 0072:22, 0077:22, 0078:22, 0079:22, 0080:22, 0081:22, 0082:22, 0084:22, 0087:22, 0093:22, 0094:22。

Pilot 六年龄桶计数：0–1=236, 2–3=276, 4–7=193, 8–15=140, 16–31=91, 32+=88。

D0 scene 分布：0000:91, 0001:91, 0002:91, 0004:91, 0005:91, 0008:91, 0010:91, 0016:91, 0018:91, 0022:91, 0023:91, 0025:91, 0029:90, 0030:90, 0032:90, 0033:90, 0034:90, 0035:90, 0036:90, 0037:90, 0040:90, 0041:90, 0047:90, 0048:90, 0049:90, 0050:90, 0054:90, 0055:90, 0056:90, 0057:90, 0060:90, 0062:90, 0066:90, 0068:90, 0070:90, 0072:90, 0077:90, 0078:90, 0079:88, 0080:90, 0081:90, 0082:90, 0084:90, 0087:36, 0093:90, 0094:90。

D0 六年龄桶计数：0–1=571, 2–3=710, 4–7=856, 8–15=898, 16–31=625, 32+=436。

D1 scene 分布：0000:47, 0001:47, 0002:47, 0004:46, 0005:46, 0008:46, 0010:46, 0016:46, 0018:46, 0022:46, 0023:46, 0025:46, 0029:46, 0030:46, 0032:46, 0033:46, 0034:46, 0035:46, 0036:41, 0037:46, 0040:46, 0041:46, 0047:46, 0048:46, 0049:46, 0050:46, 0054:46, 0055:46, 0056:46, 0057:46, 0060:46, 0062:46, 0066:46, 0068:46, 0070:46, 0072:46, 0077:46, 0078:31, 0079:29, 0080:46, 0081:46, 0082:46, 0084:46, 0087:12, 0093:46, 0094:46。

D1 六年龄桶计数：0–1=215, 2–3=306, 4–7=356, 8–15=396, 16–31=394, 32+=381。

实际新native前向计数：160000/160000；包含正确性、roll-in、paired rollout、pilot、聚合及重放，也包括official-val核心推理。val前累计158184＋H8已完成1816＝160000；第160001次在执行前被拒绝，没有提前因剩余额度不足而停止。已存 U/R 分支记录 98304 次 forward，累加 native cooperative 计时 35064.128 s；这只是已存配对分支计时，不等于总耗时或网络吞吐。

八处理帧实际时间窗 min/median/max=0.567050/0.695420/1.969750 s，不宣称精确 0.8 s。Pilot 门限为至少 16 个 H8 UPDATE 正例、至少 8 scenes；未满足时保持停止状态。

D0全4096行与D1全2048行的CPU归档全扫描均PASS；合计6144个唯一sample key，D0/D1交集0，H8/H1使用同一行集，H1取同一分支首帧。证据：[experiments/budget_free/value_v1/evidence/d0_label_fullscan.json](../../experiments/budget_free/value_v1/evidence/d0_label_fullscan.json)、[experiments/budget_free/value_v1/evidence/d1_label_fullscan.json](../../experiments/budget_free/value_v1/evidence/d1_label_fullscan.json)、[experiments/budget_free/value_v1/evidence/aggregate_label_review.json](../../experiments/budget_free/value_v1/evidence/aggregate_label_review.json)。归档复核不新增native推理。

标签原始分项、身份、特征与抽样计划：[experiments/budget_free/value_v1/dataset/d0](../../experiments/budget_free/value_v1/dataset/d0)、[experiments/budget_free/value_v1/dataset/d1](../../experiments/budget_free/value_v1/dataset/d1)、[experiments/budget_free/value_v1/dataset/pilot_summary.json](../../experiments/budget_free/value_v1/dataset/pilot_summary.json)、[experiments/budget_free/value_v1/dataset/sampling_plan.json](../../experiments/budget_free/value_v1/dataset/sampling_plan.json)。sender NPZ 只有 features/sample_keys；GT/未来分项仅在独立标签 metadata。

## 训练实际执行与模型身份

| 模型 | 状态 | 配置epochs | 记录epochs | precision | recall | PR-AUC | 预测UPDATE比 | 经验regret | always-REUSE regret | 实际checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round0_h8 | COMPLETE | 40 | 40 | 61.17% | 90.33% | 0.822775 | 10.82% | 0.001225 | 0.023154 | f578364db47286c1adaaa29a0988e73cec0d0b68754bf62f43e438b4c374b86d |
| final_h8 | COMPLETE | 40 | 40 | 55.44% | 90.00% | 0.773517 | 12.42% | 0.002231 | 0.025397 | 111b14ae0b9a038b550e84d16cd665f9264dea9bcc670d40e868adafd5986d38 |
| final_h1 | COMPLETE | 40 | 40 | 82.09% | 79.71% | 0.812792 | 1.09% | 0.000029 | 0.005266 | 87044b5142e7cdd9407ad59d617bee46bcebd39606dbd649b22fab45116b39f1 |

这些是拟合数据上的可学习性诊断，不是独立泛化验证。上述三个模型均实际记录40轮；单类PR-AUC规则为N/A，本轮有两类标签。完整逐轮 loss、梯度和加权错误见 [experiments/budget_free/value_v1/metrics/round0_h8_training.json](../../experiments/budget_free/value_v1/metrics/round0_h8_training.json)、[experiments/budget_free/value_v1/metrics/final_h8_training.json](../../experiments/budget_free/value_v1/metrics/final_h8_training.json)、[experiments/budget_free/value_v1/metrics/final_h1_training.json](../../experiments/budget_free/value_v1/metrics/final_h1_training.json)。

| 模型 | 末轮loss | 加权分类错误 | 梯度范数mean/max | 所有动作相同 | 优于always-REUSE | 优于最佳常量 |
| --- | --- | --- | --- | --- | --- | --- |
| round0_h8 | 0.124491 | 0.038724 | 1.143887/2.300551 | False | True | True |
| final_h8 | 0.159786 | 0.052339 | 1.210527/1.757213 | False | True | True |
| final_h1 | 0.017439 | 0.002709 | 0.283819/0.694456 | False | True | True |

两个final模型均使用6144行、同一归一化/锁定尺度与独立从头seed0初始化，CPU身份复核PASS：[experiments/budget_free/value_v1/evidence/final_training_identity_review.json](../../experiments/budget_free/value_v1/evidence/final_training_identity_review.json)。checkpoint SHA与state、训练记录及首次val冻结manifest一致；训练完成不等于完整性能评估完成。

## 新旧完整通信—任务表

| 方法 | 来源 | 实际B/frame | P95 B/frame | 名义10Hz B/s | mAP | AMOTA | AMOTP | MOTA | IDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INIT_ONLY | old_V2 | 2137.094692 | 3801.000000 | 21370.946924 | 0.351785 | 0.331364 | 1.333205 | 0.266591 | 296.000000 |
| mlp_B1 | old_V2 | 2630.828106 | 4825.000000 | 26308.281062 | 0.362306 | 0.369316 | 1.322707 | 0.298956 | 274.000000 |
| mlp_B2 | old_V2 | 3210.417973 | 6200.000000 | 32104.179735 | 0.361867 | 0.358024 | 1.321121 | 0.286663 | 263.000000 |
| mlp_B3 | old_V2 | 4065.218938 | 7776.750000 | 40652.189385 | 0.362019 | 0.363621 | 1.319018 | 0.287267 | 283.000000 |
| mlp_B4 | old_V2 | 5029.163450 | 9576.000000 | 50291.634499 | 0.362941 | 0.360350 | 1.320765 | 0.291246 | 320.000000 |
| P10 | old_V2 | 3042.468637 | 6261.500000 | 30424.686369 | 0.361066 | 0.357876 | 1.320195 | 0.288511 | 279.000000 |
| P12 | old_V2 | 2876.165862 | 5676.250000 | 28761.658625 | 0.362203 | 0.361646 | 1.323265 | 0.287942 | 277.000000 |
| P16 | old_V2 | 2670.816043 | 5220.500000 | 26708.160434 | 0.360143 | 0.360109 | 1.324434 | 0.292348 | 280.000000 |
| P24 | old_V2 | 2463.819662 | 4595.750000 | 24638.196622 | 0.359570 | 0.357263 | 1.323016 | 0.291495 | 297.000000 |
| P32 | old_V2 | 2362.438480 | 4238.000000 | 23624.384801 | 0.359524 | 0.357565 | 1.323500 | 0.285207 | 256.000000 |
| P6 | old_V2 | 3710.737636 | 8420.000000 | 37107.376357 | 0.363033 | 0.361545 | 1.320432 | 0.284994 | 274.000000 |
| P8 | old_V2 | 3295.333534 | 7335.000000 | 32953.335344 | 0.360608 | 0.357244 | 1.322270 | 0.286450 | 277.000000 |
| transformer_B1 | old_V2 | 2957.317853 | 4374.000000 | 29573.178528 | 0.361828 | 0.366951 | 1.323104 | 0.291140 | 229.000000 |
| transformer_B2 | old_V2 | 4174.127262 | 6058.000000 | 41741.272618 | 0.363251 | 0.364243 | 1.321886 | 0.285953 | 284.000000 |
| transformer_B3 | old_V2 | 5870.556695 | 8675.250000 | 58705.566948 | 0.361564 | 0.353208 | 1.321840 | 0.284958 | 317.000000 |
| transformer_B4 | old_V2 | 7560.870326 | 11310.500000 | 75608.703257 | 0.360639 | 0.357989 | 1.322362 | 0.290358 | 320.000000 |
| P2 | inherited_reference | 7100.773824 | 14782.000000 | 71007.738239 | 0.362819 | 0.352596 | 1.323629 | 0.287303 | 366.000000 |
| P4 | inherited_reference | 4559.422799 | 10330.000000 | 45594.227986 | 0.360797 | 0.354025 | 1.325711 | 0.283004 | 372.000000 |
| P1 | inherited_reference | 12161.645356 | 21043.000000 | 121616.453559 | 0.361107 | 0.353343 | 1.323493 | 0.287303 | 316.000000 |
| FULL_FP32 | inherited_reference | 42636.558806 | 75885.000000 | 426365.588058 | 0.362345 | 0.356898 | 1.320563 | 0.290038 | 354.000000 |
| BF_VALUE_H8 | 本轮新方法 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| BF_VALUE_H1 | 本轮新方法 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

旧20点来自实际文件的只读快照：[experiments/budget_free/value_v1/metrics/inherited_references.json](../../experiments/budget_free/value_v1/metrics/inherited_references.json)、[experiments/budget_free/value_v1/metrics/inherited_references.csv](../../experiments/budget_free/value_v1/metrics/inherited_references.csv)，保留每个源结果 SHA。H8部分run与H1未运行不构成完整新测点，正式表保持N/A；partial通信不进入完整新点。

P1/P2/P4 保留历史 serializer/RNG 标签；P1 保留 CTKMSG01 原费用，未统一重计。FULL_FP32 为继承 multiclass evaluator 的 car 分类别值，格式/RNG 有差异。16个新正式 V2 run 缺生成时预测内容 SHA；当前源文件 SHA 只能标识现存内容，不能补成历史绑定，也不能据此声称已复现预测 Bug。

## 新点比较、行为与结果绑定

新模型尚无完整 official-val 比较；最近 Periodic、上下邻点、rate gap、前沿新增判断均为 N/A。

H8 vs H1：N/A；没有两个完成run，不能判断短窗口监督是否优于即时监督。

BF_VALUE_H8：`PARTIAL_NOT_EVALUATED`，实际完成1816/3316帧。已CPU反序列化raw确认1816条预测，token逐个与communication一致、无重复（首000870，末010812）；23608个tensor均在CPU，审查未初始化CUDA或构造/执行native模型。恢复next_frame_index=1816。部分内容SHA绑定已核验，完整预测→evaluator→正式结果链未完成。

PARTIAL_ONLY raw生成SHA：`4f628e28877ae78e226bbc11d449653833be3444b646f0cf99059efc3e076991`；communication SHA：`ce366d798934a14cbdb3396e17cace40122067ce1a8650f88eb5cd6036479538`；partial manifest SHA：`a9d3dee363dbbc7e26cffb67a60c7b2df7cdf0f0b33c98c2adb09088b02fe192`。evaluator raw-input SHA=`NOT_RUN`，格式化det/tracking SHA=`NOT_RUN`，官方指标=`N/A`。partial明确标记official_evaluator_input=false、evaluator_consumed=false。在线反事实教师调用0、GT决策调用0；正常native推理1816次已计入原160000总额。

以下仅为已完成前1816帧的子集诊断，不是完整21-scene结果，不能与旧3316帧测点作通信匹配、指标差或前沿判断：

| 前1816帧子集诊断 | 实际值 |
| --- | --- |
| mean / P95 B/frame | 2097.894824 / 4335.000000 |
| 总payload bytes | 3809777 |
| geometry+control / forced-init latent / optional latent总B | 2880017 / 424320 / 505440 |
| forced-init / optional query / optional UPDATE数 | 544 / 18280 / 648 |
| optional UPDATE比例 | 3.54% |
| cache age mean / median / P95 / max | 34.997101 / 20.000000 / 121.000000 / 247.000000 |
| age0–1重复刷新占optional UPDATE比例 | 34.57% |
| scheduler CPU mean / P95 ms | 0.664915 / 0.972284 |

scheduler用CPU perf_counter计学生forward/阈值；名义10Hz B/s仅是子集均值乘10，不是网络吞吐。十档分数直方图、六年龄桶及逐帧记录见[experiments/budget_free/value_v1/communication/BF_VALUE_H8_seed0_partial.json](../../experiments/budget_free/value_v1/communication/BF_VALUE_H8_seed0_partial.json)；[experiments/budget_free/value_v1/predictions/BF_VALUE_H8_seed0/predictions.partial.pkl](../../experiments/budget_free/value_v1/predictions/BF_VALUE_H8_seed0/predictions.partial.pkl)、[experiments/budget_free/value_v1/evidence/BF_VALUE_H8_seed0_partial_manifest.json](../../experiments/budget_free/value_v1/evidence/BF_VALUE_H8_seed0_partial_manifest.json)、[experiments/budget_free/value_v1/metrics/BF_VALUE_H8_seed0_failure.json](../../experiments/budget_free/value_v1/metrics/BF_VALUE_H8_seed0_failure.json)保留。

BF_VALUE_H1：`NOT_RUN`，未执行native前向、未产生预测/格式化输入或官方指标；生成SHA、evaluator raw-input SHA、det/tracking SHA均为 `NOT_RUN`。H8/H1完整mean/P95、mAP/AMOTA/AMOTP/MOTA/IDS均为N/A；没有新完整comparison/frontier或图表，figures目录为空。

首次新val冻结身份[experiments/budget_free/value_v1/evaluation_manifest.json](../../experiments/budget_free/value_v1/evaluation_manifest.json)；SHA=`4e10461335ad36c4882ba7c829e22a060e37e70c1c8322c7eaa740cbd986790b`。仅partial内容身份可核验，不声明完整新provenance PASS，也不补造旧生成时hash。终端执行证据[experiments/budget_free/value_v1/evidence/terminal_execution_verification.json](../../experiments/budget_free/value_v1/evidence/terminal_execution_verification.json)；本次报告准确性审查[experiments/budget_free/value_v1/evidence/report_completion_review.json](../../experiments/budget_free/value_v1/evidence/report_completion_review.json)。

## 实现修正、可复现来源与限制

[experiments/budget_free/value_v1/evidence/implementation_revisions.json](../../experiments/budget_free/value_v1/evidence/implementation_revisions.json)：阶段=before_first_teacher_forward_and_before_any_labels_or_new_val，修正时 teacher forwards=0，协议超参数改变=False。

修正内容：unconditional cloning of frozen native Instances fields；full per-frame native prediction and decoded-input comparisons in null/order/poison tests；explicit FEATURES import and actual training report filenames；sample hash includes seed and explicit round/scene/session/policy；checkpoint SHA and identity JSON SHA recorded separately；explicit new label-generation source/feature schema freeze and per-row source identity；real native frozen-Instances clone regression self-check；streamed per-shard count and resume-position evidence。

[experiments/budget_free/value_v1/evidence/model_counter_guard_revision.json](../../experiments/budget_free/value_v1/evidence/model_counter_guard_revision.json)：实际边界 `post_round0_pre_d1`，计数118054→118054，修复新增native/GPU前向0。原post_d0 watcher因state.json原子替换窗口中的FileNotFoundError失败，未发信号；随后暂停自身3933，在round0已完成、D1未开始的边界归档并修复counter/source guard。旧round0 checkpoint、D0行及协议/epsilon/抽样/结构/训练超参数保留；新进程4117恢复，未重训round0。证据：[experiments/budget_free/value_v1/evidence/guard_boundary_pause.json](../../experiments/budget_free/value_v1/evidence/guard_boundary_pause.json)、[experiments/budget_free/value_v1/evidence/guard_boundary_resumed.json](../../experiments/budget_free/value_v1/evidence/guard_boundary_resumed.json)、[experiments/budget_free/value_v1/evidence/model_counter_guard_review.json](../../experiments/budget_free/value_v1/evidence/model_counter_guard_review.json)。

[experiments/budget_free/value_v1/evidence/readonly_encoder_revision.json](../../experiments/budget_free/value_v1/evidence/readonly_encoder_revision.json)：阶段=pre_new_official_val_readonly_encoder_equivalence_revision，修正时 teacher forwards=2893，协议超参数改变=False。

修正内容：validate input-content keyed frozen predecision image/BEV reuse with full native output/state/RNG parity before enabling；replace large raw evidence report with compact complete evidence renderer。

可选图像缓存等价性证据 [experiments/budget_free/value_v1/evidence/readonly_cache_equivalence.json](../../experiments/budget_free/value_v1/evidence/readonly_cache_equivalence.json)：{'status': 'PASS', 'prefix_sha256': '8916d89fb3a30c72dd6817f7dcc874904f195e4455fe6643db8eb9678fb5972f', 'preserved_original_teacher_window': 'PASS', 'selected_cache': 'none', 'selection_rule': 'lowest primary cold-pair native time ratio below 1; exact parity and real warm hits required; timing is a single implementation diagnostic', 'entry_bound': 16, 'storage': 'CPU detached independent copies; fresh native returns', 'frozen_params_buffers': 'PASS', 'teacher_forward_accounting': 'Every uncached/cached native forward counted; cap unchanged', 'temporal_sharing': 'NONE: only input-content-identical frozen predecision encoder quantities; independent cloned outputs', 'teacher_frame_forwards': 246}。它只覆盖已测输入/输出；未提供证据的范围不扩大声称等价。

| 标签来源identity SHA256 | 实际行数 | 当前/归档证据 |
| --- | --- | --- |
| 02a7cba82e6a95dbcdbd1bccb2d379afe9c5a643b1bd0f8e3855dc029cc24cb1 | 4062 | [experiments/budget_free/value_v1/evidence/source_revisions/revision1/label_source_identity.json](../../experiments/budget_free/value_v1/evidence/source_revisions/revision1/label_source_identity.json) |
| 7f715ca416157d34a607c3e94cf14c2d97f9764e5ccd42b399e59ec788f1b293 | 2048 | [experiments/budget_free/value_v1/evidence/label_source_identity.json](../../experiments/budget_free/value_v1/evidence/label_source_identity.json) |
| aae5a65c061d4a0c1736848eeb5660313f25ab96c4015c18ce562c019281beb6 | 34 | [experiments/budget_free/value_v1/evidence/source_revisions/revision0/label_source_identity.json](../../experiments/budget_free/value_v1/evidence/source_revisions/revision0/label_source_identity.json) |

实际D0来源为34行`aae5…`与4062行LF身份`02a7…`；D1全部2048行为当前LF `7f715…`。早期`184aff…`对应同值JSON的CRLF序列化，其raw SHA与实际LF `02a7…`不同，不能替代真实行来源。只解析身份并保留归档，未重写旧行/identity；证据[experiments/budget_free/value_v1/evidence/source_identity_serialization.json](../../experiments/budget_free/value_v1/evidence/source_identity_serialization.json)。

`native_identity.files`的170条从原保护基线复制，属于历史字典，不能称当前170份源码现场认证；两个refresh helper不被本轮模型消费。[experiments/budget_free/value_v1/evidence/preval_dependency_recheck.json](../../experiments/budget_free/value_v1/evidence/preval_dependency_recheck.json)对148个明确必要依赖逐路径核验无漂移，包括104份旧eval源码、11份新源、权重/infos/split和11份活跃native库。历史203只是无路径名单的旧聚合，不伪称复算同一集合。首次val冻结manifest和本次11份源码现场核验仍与实际SHA一致。

存在多个 source identity 时按实际行计数分别披露；当前源码身份不能覆盖旧行来源。归档保留意味着可核对来源，不意味着改动前后所有输出已独立证明等价。

单seed，统计稳定性 `NOT_TESTED`。教师在 train 场景上训练过，训练内指标不能称为整个系统 unseen 泛化。有限窗口、sender 部分可观测、GT离线未来标签和联合决策非线性限制收益预测；单query边际收益不保证可线性相加。代理不保证与整体mAP/AMOTA单调一致，无预算输入不保证省通信或满足链路容量。H8即使更好也不能单独归因为去掉B；本轮同时改变预算条件、监督和有限窗口标签。标签中的候选后续锁定与部署动作历史不同。

`work/`仅保留被修订证据引用的`evaluation_counter_guard.patch`、`model_identity_guards.patch`，无临时缓存待删。冻结formatter未改；本报告在主进程exit0后直接修正事实表述，未重跑训练、native教师或evaluator。

所有有利、不利、常量输出和失败结果均保留。未根据结果自动调epsilon/学习率、切换Task+通信惩罚/RL/BPTT、扩张种子或启动未完成消融。最终停止原因与恢复位置见 [experiments/budget_free/value_v1/state.json](../../experiments/budget_free/value_v1/state.json)；原始运行日志见 [logs/budget_free/value_v1.log](../../logs/budget_free/value_v1.log)。
