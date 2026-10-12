# 几何与高维 latent 刷新频率 2×2 消融

**REFRESH_2X2_ABLATION_COMPLETE**。四组任务结果独立审计 PASS；必需保护产物检查 PASS，全量起始清单的原始比较 FAIL（差异披露见下文）。Official val：21 scenes / 3316 samples，R50、FP32、10Hz nominal、car-only、seed 0。模型与缓存均继承冻结版本；本轮没有训练、GT 在线调度、val 规则调整或 checkpoint 选择。

本次固定策略下，持续几何刷新的平均 AMOTA 增益为 0.344009，高维 latent 刷新为 0.012646。仅刷新几何的 B 相比 Full-Refresh D 节省 83.56% 实际 packet，AMOTA 降低 0.025291；停止后续几何刷新后的 A/C 均为 AMOTA=0。该结果支持当前策略保留几何刷新，latent 的收益依赖几何状态；结论限于本次单 seed 描述性比较。

## 目录与保护

沿用现有正式目录，新增入口 `scripts/refresh_ablation.py` 与独立 ablation 结果；被正式协议引用的两个 `coop_val.py` 保留。本轮自身清理 0 文件。必需资产范围 **PASS**：2037 个仍存在的原始强哈希文件一致，9 类正式结果/模型、数据/cache/环境目录摘要一致；11 个运行时 `.so` 与完整 wheel SHA256 通过，当前 Git 内容/路径/大小及原 HEAD/status/diff、fsck 通过。3316 个 val cache 另做完整 SHA256/大小核验；大型数据目录仍主要依赖 metadata 清单，未宣称全量重新哈希。

原始全文件比较与严格 Git 元数据对账均保留 **FAIL**。另一任务完成白名单清理 116 文件、232 空目录（下载副本、build 派生文件与 bytecode），实际缺失逐项与 Apply 证据对账；其中 51 项属于原始强哈希清单，没有发现仍存在的原文件内容变化。另保留 16 项并行任务新增文件，不把两类外部变化计成本轮清理。详见 [清理报告](../PROJECT_STRUCTURE_CLEANUP.md) 与 [固定机器证据](../../audits/project_structure_cleanup/cleanup_evidence.json)。

两个库的 `FETCH_HEAD` 与 `.git/index` mtime 推进，内容与大小一致；原始 native 扫描还比随后稳定读取少 103 B，原因未证实，临时 fetch 截断仅为推断。原 Git baseline 只有 aggregate 摘要，无法补证整个 `.git` 起始与结束的逐文件内容完全相等。这些差异没有被改写为 PASS；必需范围的内容检查与全清单/时间戳一致性分开披露。完整清单、时间、哈希与限制见 [目录审计](../../experiments/v2/ablations/refresh_2x2/metrics/structure_audit.json) 和 [Git 差异核验](../../experiments/v2/ablations/refresh_2x2/metrics/git_metadata_drift_verification.json)。

报告生成时本轮 refresh 推理 writer 已退出，观察到 budget_free writer（PID 3933）。本轮未删除、覆盖或停止另一任务；观察时点及命令保存在 [机器汇总](../../experiments/v2/ablations/refresh_2x2/metrics/report_summary.json)，不宣称整个工作区无其他 writer。

## 实现与复用

几何与 latent 按 `(scene, native reset session, predicted track ID)` 独立缓存，新 ID 强制初始化两类信息；ID 消失后保留本 session 已接收状态，scene/session 变化隔离缓存，无 TTL、年龄上限或填充更新。几何字段为 `ref_pts` 与 `pred_boxes`，latent 为三组 256 维字段。

`pred_boxes=[cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy]`；中心与速度在路端 LiDAR 物理坐标中，速度单位 m/s。`ref_pts` 为 infrastructure point-cloud range `[0,-51.2,-5,102.4,51.2,3]` 的仿射归一化坐标。A/C 从最后实收状态按实际 timestamp（μs→s）执行固定恒速平移，保持初始化 z、尺寸、物理 yaw 与速度；发送的必要标定/时序 metadata 将状态从实收路端坐标系输运到当前路端坐标系。参考点应用同一物理位移，并核验反归一化参考点与框中心的一致性；车端融合继续使用官方 `veh2inf_rt`。没有读入当前未发送框、未发送 reference point、GT 或未来帧补偿。

预测目标是当前路端 sender timestamp，不额外向车端时刻外推。同 session 的真实相邻发送间隔 min/median/p95/max=0.041133/0.0990025/0.11449215/0.499930 s，未强行改成 0.1 s。原始路车配对时差（路端−车端 pointcloud timestamp）范围 -0.062181 至 0.070789 s，正时差 1491 帧；四组保留该配对与时间偏差。通信 B/s 仍按原协议 nominal 10Hz 报告。

INT8 延用 per-query/per-field absmax/127、`np.rint` 与 FP32 scale，每次 latent UPDATE 为 780 B/query。CTKR2X01 序列化包含 ID、两个独立 flags、必要 metadata、长度和 CRC；只有对应 UPDATE 才写入当前 payload。几何 UPDATE 为 52 B/query，控制开销另计。逐帧验证 `len(packet)=control+geometry+latent`，本地预测与缓存复制不计通信；无线/TCP/IP/重传开销未计入。

B 通过 3316 帧完整 CPU replay，五组 receiver 张量与 frozen INIT_ONLY 逐元素 bitwise 相等，沿用其预测和 evaluator 指标，仅按统一新协议重新计费。A/C 为新增推理。D 为独立命名的 V2-native Full-Refresh：历史 P1 的 CTKMSG01 与模型构建前 seed 放置不同，而 native inference 存在 CPU `torch.randperm`；无法证明历史预测与 V2 seed 放置相等，因此本轮不复用历史 P1 指标。P1/P2/P4 全部保持原文件不变。

所有新增运行在构建模型、加载 checkpoint、创建 loader iterator 后重置 seed 0，继承 V2 CUDA 数值设置。Correctness PASS：A: 49 帧/49 次对齐前检查/49 次实际 aggregation 检查/45 次未发送字段扰动；B: 49 帧/49 次对齐前检查/49 次实际 aggregation 检查/45 次未发送字段扰动；C: 49 帧/49 次对齐前检查/49 次实际 aggregation 检查/45 次未发送字段扰动；D: 49 帧/49 次对齐前检查/49 次实际 aggregation 检查/0 次未发送字段扰动。未发送字段修改保持 flags 与实收状态固定，并检查真正进入 crossview fusion 的五组张量；正式运行也检查融合输入及有限输出。额外混合 flags、多 ID、跨 scene/session、>32 帧缓存与 INT8/字节检查见原始 correctness 证据。

## 四组任务指标

| 组 | Geo update | Latent update | B/frame | mAP | AMOTA | AMOTP | MOTA | IDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 初始化+预测 | 仅初始化 | 1308.931 | 0.123578 | 0.000000 | 1.681646 | 0.000000 | 571 |
| B | 每帧 | 仅初始化 | 1979.662 | 0.351785 | 0.331364 | 1.333205 | 0.266591 | 296 |
| C | 初始化+预测 | 每帧 | 11369.896 | 0.122976 | 0.000000 | 1.656752 | 0.000000 | 1079 |
| D | 每帧 | 每帧 | 12040.627 | 0.361648 | 0.356655 | 1.322297 | 0.286024 | 312 |

## 真实通信分解

分解列均为 B/frame；UPDATE ratio 按 query 加权，初始化也计为 UPDATE。Bytes/s 使用 nominal 10Hz。

| 组 | P95 B/frame | B/s | Control | Geometry | Latent | Geo ratio | Latent ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2661.250 | 13089.309 | 918.272 | 24.416 | 366.242 | 0.035124 | 0.035124 |
| B | 3649.000 | 19796.619 | 918.272 | 695.147 | 366.242 | 1.000000 | 0.035124 |
| C | 19799.000 | 113698.960 | 918.272 | 24.416 | 10427.207 | 0.035124 | 1.000000 |
| D | 20994.000 | 120406.270 | 918.272 | 695.147 | 10427.207 | 1.000000 | 1.000000 |

## 几何预测质量

以下为 receiver 重建状态相对当前路端预测的离线差异，参考框仅用于分析，不能把它解释为相对 GT 的绝对定位误差。Age 为 query 加权的使用时年龄（UPDATE 后为 0）；帧年龄同时保留真实 dt。Yaw 为 wrap 后绝对角误差（rad）；速度差为 XY 速度 L2（m/s）。

| 组 | Age mean | Age p95 | Age max | dt mean s | dt p95 s | Position mean m | Position p95 m | Yaw mean rad | Velocity error mean m/s | Received speed mean m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 50.233 | 156.000 | 248.000 | 5.050 | 15.723 | 9.551 | 39.637 | 0.172 | 1.528 | 3.119 |
| B | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.199 |
| C | 50.233 | 156.000 | 248.000 | 5.050 | 15.723 | 9.551 | 39.637 | 0.172 | 1.528 | 3.119 |
| D | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.199 |

| 组 | Geo age frames | Queries | Position mean m | Position p95 m | Yaw mean rad | Velocity error mean m/s |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0-1 | 2616 | 0.144 | 0.698 | 0.031 | 0.202 |
| A | 2-3 | 1834 | 0.760 | 2.026 | 0.096 | 0.764 |
| A | 4-7 | 3128 | 1.541 | 4.209 | 0.125 | 1.073 |
| A | 8-15 | 5305 | 2.997 | 8.632 | 0.169 | 1.387 |
| A | 16-31 | 8257 | 5.811 | 17.018 | 0.216 | 1.805 |
| A | 32+ | 23189 | 15.219 | 53.381 | 0.185 | 1.732 |
| C | 0-1 | 2616 | 0.144 | 0.698 | 0.031 | 0.202 |
| C | 2-3 | 1834 | 0.760 | 2.026 | 0.096 | 0.764 |
| C | 4-7 | 3128 | 1.541 | 4.209 | 0.125 | 1.073 |
| C | 8-15 | 5305 | 2.997 | 8.632 | 0.169 | 1.387 |
| C | 16-31 | 8257 | 5.811 | 17.018 | 0.216 | 1.805 |
| C | 32+ | 23189 | 15.219 | 53.381 | 0.185 | 1.732 |

A 单 query 年龄桶平均位置误差（m）：0-1 桶 0.1443 → 32+ 桶 15.2192；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 C 单 query 年龄桶平均位置误差（m）：0-1 桶 0.1443 → 32+ 桶 15.2192；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 这是年龄桶横截面均值，不能解释为同一个目标的连续误差轨迹。

## 年龄与性能的描述性检查

在 A/C/D 正式结果可见前固定了以下离线规则：保存的 detection JSON，car score≥0.5、XY 中心距离<2m、按分数降序贪心一对一匹配；继承 50m ego range 与非零点数 GT filter。它没有重新推理、修改策略或计算另一份正式指标。用每帧 A 所有路端 query 的平均 geometry age 分桶，四组使用同一帧集合；GT=28054，无路端 query 帧=0（不进入年龄分桶）。Recall/F1 是该固定诊断点的 micro aggregate，不能称为 mAP/AMOTA。

以下位置误差与性能差使用完全相同的帧集合；位置误差与 dt 按该集合中的路端 query 加权，recall/F1 按该集合的 detection/GT 数量 micro 聚合。它们与上节按单个 query 年龄分桶的几何误差具有不同分组单位。表内差值为 A−B/C−D；下述趋势中的退化差为 B−A/D−C，正数代表预测组在该诊断点更差。

| Mean frame age | Frames | GT | Queries | dt mean s | A position mean m | C position mean m | A recall | B recall | C recall | D recall | A−B recall | A−B F1 | C−D recall | C−D F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1 | 60 | 223 | 657 | 0.085 | 0.252 | 0.252 | 0.2152 | 0.2197 | 0.2108 | 0.2197 | -0.0045 | -0.0056 | -0.0090 | -0.0124 |
| 2-3 | 48 | 180 | 549 | 0.296 | 0.752 | 0.752 | 0.2889 | 0.3278 | 0.2889 | 0.3278 | -0.0389 | -0.0485 | -0.0389 | -0.0459 |
| 4-7 | 98 | 390 | 1150 | 0.584 | 1.463 | 1.463 | 0.2821 | 0.3410 | 0.3026 | 0.3282 | -0.0590 | -0.0679 | -0.0256 | -0.0318 |
| 8-15 | 235 | 1032 | 2927 | 1.225 | 3.097 | 3.097 | 0.3450 | 0.4128 | 0.3605 | 0.4215 | -0.0678 | -0.0802 | -0.0610 | -0.0782 |
| 16-31 | 600 | 3893 | 7795 | 2.482 | 6.239 | 6.239 | 0.3000 | 0.4988 | 0.3144 | 0.5127 | -0.1988 | -0.2346 | -0.1983 | -0.2422 |
| 32+ | 2275 | 22336 | 31251 | 6.402 | 11.630 | 11.630 | 0.3172 | 0.5698 | 0.3291 | 0.5756 | -0.2526 | -0.2834 | -0.2465 | -0.2920 |

A 平均位置误差（m）：0-1 桶 0.2522 → 32+ 桶 11.6296；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 C 平均位置误差（m）：0-1 桶 0.2522 → 32+ 桶 11.6296；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 B−A recall 退化差：0-1 桶 0.0045 → 32+ 桶 0.2526；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 D−C recall 退化差：0-1 桶 0.0090 → 32+ 桶 0.2465；首末总体增加，6 个有效桶随年龄非单调（并非严格增加）。 B−A F1 退化差：0-1 桶 0.0056 → 32+ 桶 0.2834；首末总体增加，6 个有效桶随年龄非递减（严格增加）。 D−C F1 退化差：0-1 桶 0.0124 → 32+ 桶 0.2920；首末总体增加，6 个有效桶随年龄非单调（并非严格增加）。

上述趋势仅描述已填充年龄桶的聚合值，不是对象的时间轨迹或显著性检验。帧平均年龄不是某个最终车端 track 的年龄；场景、目标数量、可见性、初始化时机与 score operating point 均有混杂。即使长龄桶出现更大退化，也不能据此证明 object-age 条件下 AMOTA 单调下降。完整 TP/FP/FN、每帧结果及 F1 见离线诊断 JSON。

## 两因素效应

差值为表中左组减右组；mAP/AMOTA/MOTA 越高越好，AMOTP/IDS 越低越好。全部差值均披露。

| 对比 | ΔB/frame | ΔmAP | ΔAMOTA | ΔAMOTP | ΔMOTA | ΔIDS |
| --- | --- | --- | --- | --- | --- | --- |
| B-A | 670.731 | 0.228206 | 0.331364 | -0.348441 | 0.266591 | -275 |
| C-A | 10060.965 | -0.000603 | 0.000000 | -0.024894 | 0.000000 | 508 |
| D-B | 10060.965 | 0.009863 | 0.025291 | -0.010907 | 0.019433 | 16 |
| D-C | 670.731 | 0.238672 | 0.356655 | -0.334455 | 0.286024 | -767 |

固定 init-only latent 时持续刷新几何（B-A）：两项指标均提高，ΔmAP=0.228206、ΔAMOTA=0.331364。 固定逐帧 latent 时持续刷新几何（D-C）：两项指标均提高，ΔmAP=0.238672、ΔAMOTA=0.356655。 固定预测几何时持续刷新 latent（C-A）：检测与跟踪结果混合，ΔmAP=-0.000603、ΔAMOTA=0.000000。 固定逐帧几何时持续刷新 latent（D-B）：两项指标均提高，ΔmAP=0.009863、ΔAMOTA=0.025291。

几何平均效应 `((B−A)+(D−C))/2`：ΔmAP=0.233439、ΔAMOTA=0.344009；latent 平均效应 `((C−A)+(D−B))/2`：ΔmAP=0.004630、ΔAMOTA=0.012646。按本次 AMOTA 平均差值，贡献更大的是：显式几何。该排序只描述当前固定策略与单 seed。

交互差值 `D−C−B+A`：mAP=0.010466，AMOTA=0.025291。这是完整 2×2 的数值 difference-in-differences；不能把无重复实验的非零值当成显著交互或可推广的因果效应。

## 对原有低通信结果的解释

| 冻结 V2 参考 | 原协议 B/frame | mAP | AMOTA | IDS |
| --- | --- | --- | --- | --- |
| INIT_ONLY_seed0 | 2137.095 | 0.351785 | 0.331364 | 296 |
| P32_seed0 | 2362.438 | 0.359524 | 0.357565 | 256 |
| mlp_B1_seed0 | 2630.828 | 0.362306 | 0.369316 | 274 |

INIT_ONLY 对应本轮 B：它每帧仍接收显式几何，只停止后续 latent 刷新。P32 和 MLP B1 也持续刷新显式几何，同时以不同频率/策略更新 latent；其较低通信成本不能证明可停止几何发送。MLP B1 的单 seed AMOTA 改善是在给定预算与状态选择策略下观察到的结果，不代表本轮恒速几何预测或 Transformer 架构优势。上表保留 frozen V2 的原协议字节，跨协议时仅用于背景解释，不混入本轮四组通信表。

## 可以支持的结论与限制

本实验量化了两类显式字段持续刷新的相对贡献、相应真实 payload/control 成本，以及初始化后恒速预测的累计差异。是否有必要持续刷新应依据 B−A 与 D−C、C−A 与 D−B 的具体差值及应用容忍度判断，不能从一次策略比较宣称普遍必要或不必要。初始化+预测包含实收几何；初始化一次 latent 包含高维信息；latent 本身仍可能含空间线索。

持续显式几何刷新：两种另一因素设置下 mAP 与 AMOTA 均提高，本结果支持在所测试策略下保留该刷新以获得相应指标。 持续高维 latent 刷新：收益依赖另一因素及指标，不能统一判断为必要或可省略。 这些是单 seed 的描述性收益，不能据此把所有应用的刷新需求定为相同。

停掉初始化后的几何刷新（B→A）减少 geometry payload 670.731 B/frame（96.49%），总 packet 减少 33.88%；停掉初始化后的 latent 刷新（D→B）减少 latent payload 10060.965 B/frame（96.49%），总 packet 减少 83.56%。这些节省需要与对应表中性能差同时评估；控制开销、初始化和必要 pose metadata 仍被真实计入。

四组使用相同 split、路端预测、车端权重、量化、evaluator、seed 放置与消息实现；B 的严格张量等价复用节省了完整 GPU 推理。单 seed、未复制实验且 native CUDA kernels 未保证全局确定，无法支持显著性、跨 seed 鲁棒性或跨数据泛化。不同策略改变 retained queries 数量及原生随机 query filling 的后续调用，受控 seed 并不保证逐帧随机模板完全一致。几何误差采用当前路端预测参考，年龄诊断仅有 frame proxy；后续如需年龄条件的跟踪结论，应预注册对象对应关系与共同 operating point，再做独立重复，不在当前 val 调规则。

模型与消息检查未发现未发送信息旁路、NaN/Inf、身份覆盖异常或计费异常。目录保护发现 Git 元数据摘要漂移、另一并行任务新增文件以及独立白名单清理，原始全文件 FAIL 与必需产物 PASS 分开保留。不能把范围调整后的状态解释为整个 `.git` 前后字节相等、起始清单全部文件仍在或整个工作区没有其他 writer。任何必需资产未证实保持的部分均按 FAIL/未确认披露，而不以完成推理掩盖。

## 原始证据与图

- [配置](../../configs/main_experiment_v2/refresh_ablation.json)，[继承的车端配置](../../configs/main_experiment_v2/coop_val.py)，[manifest](../../experiments/v2/ablations/refresh_2x2/metrics/ablation_manifest.json)，[correctness](../../experiments/v2/ablations/refresh_2x2/metrics/correctness.json)，[独立审计](../../experiments/v2/ablations/refresh_2x2/metrics/independent_audit.json)，[目录与保护审计](../../experiments/v2/ablations/refresh_2x2/metrics/structure_audit.json)，[Git 元数据差异核验](../../experiments/v2/ablations/refresh_2x2/metrics/git_metadata_drift_verification.json)，[3316 个 val cache SHA256/大小核验](../../experiments/v2/ablations/refresh_2x2/metrics/val_cache_verification.json)，[机器汇总](../../experiments/v2/ablations/refresh_2x2/metrics/report_summary.json)，[离线年龄诊断](../../experiments/v2/ablations/refresh_2x2/metrics/detection_age_diagnostic.json)。
- A/B/C/D：`experiments/v2/ablations/refresh_2x2/metrics/<group>/result.json`，逐帧通信与 query 误差：`communication/<group>.json`；A/C/D 完整 evaluator 与 lossless gzip 预测：`predictions/<group>/`；B 指向 [原 INIT_ONLY 预测](../../experiments/v2/predictions/official_val/INIT_ONLY_seed0/)。
- [冻结 V2 正式报告](V2_FINAL.md)，[运行日志](../../experiments/v2/ablations/refresh_2x2/metrics/run.log)。

![真实通信与 AMOTA](../../experiments/v2/ablations/refresh_2x2/figures/bytes_vs_amota.png)

![四组任务指标](../../experiments/v2/ablations/refresh_2x2/figures/four_group_metrics.png)
