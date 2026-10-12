| Setting | mAP | AMOTA |
|---|---:|---:|
| Paper 10Hz R50 | 0.356 | 0.346 |
| Current original evaluation | 0.1787047145 | 0.1309931778 |
| Current car-only re-evaluation | 0.3623447993 | 0.3568982508 |
| Difference after alignment (current − paper) | +0.0063447993 | +0.0108982508 |

完成日期：2026-10-04（Asia/Shanghai）。**ROOT_CAUSE = EVALUATION_CATEGORY_MISMATCH**。

本地 `0.1787047 / 0.1309932` 是 car、bicycle、pedestrian 三类平均；论文 Table 3 的 `0.356 / 0.346` 是 vehicle-only。复用相同 FULL_FP32 预测，仅改变 evaluator 的计分类别，实际 CPU 重评估得到 `0.3623447993 / 0.3568982508`。主要差距来自比较口径，未发现权重错误证据。

这是 mAP/AMOTA 的近似对齐，不能宣称完全数值复现：仍高出 0.63448 / 1.08983 个百分点，IDS 也仍为 354，对照论文 201。公开权重对应 Table 3 精确运行的映射未获确认；这些剩余差异未被强行归因。按提示词 Gate A 停止后续深入排查，不执行 No-Fusion、full-val GPU 重跑或训练。

论文目标来自 [ICCV 2025 主论文 Table 3](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhong_CoopTrack_Exploring_End-to-End_Learning_for_Efficient_Cooperative_Sequential_Perception_ICCV_2025_paper.pdf)。[Supplementary B.3](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhong_CoopTrack_Exploring_End-to-End_ICCV_2025_supplemental.pdf) 明确只报告 vehicle 类别；Abstract/README 的 `0.390 / 0.328` 另行记录，未用于本次 R50/10Hz 对比。详细出处及本地来源快照见 [paper_protocol.md](../artifacts/core/r50_alignment/reports/paper_protocol.md)。

| 核对项 | 状态 | 证据范围 |
|---|---|---|
| Official checkpoint bytes match | YES | 继承原生加载记录与受保护下载哈希；coop/inf 官方 MD5 一致，加载无缺失、额外 key 或形状差异 |
| Paper-to-public-checkpoint mapping confirmed | UNCONFIRMED | 官方文件身份不能证明其属于 Table 3 的精确运行 |
| Split matched | YES / MATCHED | 锁定 split SHA256 一致；作者 Issue #8 回复确认该 split；train 46 / val 21 scenes，val 3316 samples |
| 10Hz matched | YES | 原准备命令 freq=10，继承实际时间戳检查 |
| Evaluator category matched | 原始 NO；本次 YES | 三类计分改为只计 car；模型与数据集类别映射保持不变 |
| Config matched | YES | 已检查的公开 R50 推理模型设置一致；路径、batch1、初始化与缓存保存设置按既有流程调整；非新增穷尽配置审计 |
| Query cache matched | YES | 继承 3316 条原生五字段 FP32、scene 顺序/reset、入模 torch.equal 验证；非论文未公开缓存的逐字节等价声明 |

状态基于已有 Stage1/2/3 证据及少量只读元数据检查，未重新校验全部 raw 数据或 cache payload。完整身份、配置值、阈值、split/hash、频率及缓存证实范围见 [inherited_evidence.md](../artifacts/core/r50_alignment/reports/inherited_evidence.md)。官方 split 说明见 [作者 Issue #8 回复](https://github.com/zhongjiaru/CoopTrack/issues/8#issuecomment-3701582050)。

| 类别 | AP（0.5 / 1 / 2 / 4 m 均值） | AMOTA |
|---|---:|---:|
| car | 0.3623447993 | 0.3568982508 |
| bicycle | 0.1700385788 | 0.0360812825 |
| pedestrian | 0.0037307655 | 0.0000000000 |

三类 AP 算术均值精确为 `0.1787047145084463`，AMOTA 算术均值精确为 `0.13099317775947647`，没有使用 GT 数量加权。AP 各距离阈值与跟踪各类别完整指标见 [per_class_metrics.md](../artifacts/core/r50_alignment/reports/per_class_metrics.md)。

| 指标 | Paper 10Hz R50 | 本次 car-only | 本次 − 论文 |
|---|---:|---:|---:|
| mAP | 0.356 | 0.362344799 | +0.006344799 |
| AMOTA | 0.346 | 0.356898251 | +0.010898251 |
| ATE | 0.777 | 0.784273527 | +0.007273527 |
| ASE | 0.154 | 0.153668106 | -0.000331894 |
| AOE | 0.117 | 0.106025755 | -0.010974245 |
| AVE | 2.594 | 2.534476383 | -0.059523617 |
| AMOTP | 1.376 | 1.320563485 | -0.055436515 |
| IDS | 201 | 354 | +153 |

ATE 单位 m，AOE 单位 rad，AVE 单位 m/s；IDS 为 evaluator 选定最佳 MOTA operating point 的身份切换计数。误差指标及 IDS 的剩余差异保留，不将其称为已完全复现。

实际评估直接读取原 `results_nusc.json` 和 `results_nusc_det.json`；没有从 pkl 重新生成 JSON，也没有模型 forward。`dataset.CLASSES`、GT 类别映射与全部 `class_range` 保留（car=50 m；bicycle/pedestrian=40 m），只将 detection `class_names`、tracking `class_names`/`tracking_names` 改为 `[car]`。保留其他类别的合法输入及官方过滤，避免通过删除预测改变输入。距离阈值、tracking config 其他字段、val split、样本/scene 顺序均验证不变。

FULL_FP32 的 [result.json](../artifacts/core/r50_alignment/car_only_eval/result.json) 与 [evaluation_config.json](../artifacts/core/r50_alignment/car_only_eval/evaluation_config.json) 记录输入 SHA256 前后、唯一配置差异、CPU 环境及实际新指标；其他 case 的同名文件位于各自 communication_car_only 子目录。重评估结果与旧结果中的 car 指标逐项精确一致。NATIVE 与 FULL_FP32 原检测、跟踪 JSON 的实际 SHA256 分别完全相同（检测 `6011f2f74540ae92f0b856cf2b5c00a9dd9f4f4b7335c16ab944b4323908885c`，跟踪 `0524a47f4881186bb319fa83f44ae33d32a746f526f3e772ca06e082da15a88c`）；继承 tensor 验证另覆盖 3316 帧、43108 次比较，全部 torch.equal。本次 FULL_FP32 评估由此直接覆盖原生基准的同一输入，未运行 NATIVE forward。

| 通信条件 | seed | car-only mAP | car-only AMOTA |
|---|---:|---:|---:|
| FULL_FP32 | 0 | 0.3623447993 | 0.3568982508 |
| FEAT_FP16 | 0 | 0.3622880930 | 0.3582699576 |
| FEAT_INT8 | 0 | 0.3611074772 | 0.3533432787 |
| CONF_50 | 0 | 0.2963493096 | 0.2957629568 |
| CONF_25 | 0 | 0.2256043770 | 0.2217379436 |
| RANDOM_50 | 0 | 0.2826758206 | 0.2307294457 |
| RANDOM_50 | 1 | 0.2819993677 | 0.2339455947 |
| RANDOM_50 | 2 | 0.2782699052 | 0.2331845495 |
| RANDOM_25 | 0 | 0.2192912919 | 0.1288622799 |
| RANDOM_25 | 1 | 0.2245930147 | 0.1370831766 |
| RANDOM_25 | 2 | 0.2252504243 | 0.1333707329 |

FP16/INT8 在既有 Stage3 中命名为 FEAT_FP16/FEAT_INT8。七条件及 RANDOM_50/RANDOM_25 各三种 seed 共 11 个 case 均完成真实 CPU 重评估，保留各 case 原有预测与通信统计。结果见 [car_only_eval](../artifacts/core/r50_alignment/car_only_eval/) 和 [communication_car_only](../artifacts/core/r50_alignment/communication_car_only/)。

类别口径也影响通信对照排序：原三类 mAP 中 CONF_50 为 0.124384，低于三个 RANDOM_50 seed；car-only 中 CONF_50 为 0.296349，高于 RANDOM_50 的 0.278270–0.282676。与论文比较时应使用本次类别一致的表。

独立验证：37 个既有脚本/config/evaluator/split/metrics 文件 SHA256 均未变化；11 个 case 原 prediction JSON、pkl、manifest 哈希前后均一致；每个 case 的重评估都精确复原原 car 类别指标。见 [diagnostic_validation.json](../artifacts/core/r50_alignment/diagnostic_validation.json)。

```text
ROOT_CAUSE: EVALUATION_CATEGORY_MISMATCH
DOES_MODEL_NEED_RERUN: NO
DO_COMMUNICATION_EXPERIMENTS_NEED_RERUN: NO
NEXT_MINIMAL_ACTION: 使用本次 vehicle-only 表比较论文；保留三类结果并明确类别口径。若进一步要求 Table 3/IDS 精确复现，先核实官方运行及权重映射证据。
```

机器结果：[final.json](../artifacts/core/r50_alignment/final.json)。复核脚本：reevaluate_car_only.py (historical script removed during cleanup)、finalize_report.py (historical script removed during cleanup)。
