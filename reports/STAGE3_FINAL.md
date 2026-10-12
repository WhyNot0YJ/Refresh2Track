# Stage3 诊断报告

**STAGE3_DIAGNOSTICS_PASS**；完成时间 `2026-10-04T08:59:31.518688+08:00`。

沿用 FULL_DATA_READY，不重新解压、转换或校验完整数据，official split 保持不变。模型/转换运行环境均复用固定 Docker 镜像；本阶段只新增独立 stage3.py 及诊断产物。

| Gate | 结果 |
|---|---|
| inherited_full_data_ready | PASS |
| official_split_unchanged | PASS |
| no_data_preparation_reexecuted | PASS |
| train32_native_cache | PASS |
| train32_field_and_frame_mapping | PASS |
| complete_val_scene_pilot | PASS |
| val3316_cache_21_scenes | PASS |
| official_native_val3316_metrics | PASS |
| wire_format_unit_guards | PASS |
| all_actual_tensors_fp32_roundtrip | PASS |
| complete_val_fp32_exact_regression | PASS |
| all_seven_conditions_and_random_seeds | PASS |
| predicted_track_id_temporal_diagnostic | PASS |
| adapter_gradient_milestones | PASS |
| official_parameters_and_buffers_frozen | PASS |
| no_val_or_formal_innovation_updates | PASS |

训练片段来自 scene `0000` 的32帧；完整 val 为21 scenes /3316条。首个 val scene pilot 为 `0003`，完整207帧。全量路侧生成从原生scene起点重放，保留原生时间顺序与重置规则；单scene pilot是进入全量阶段前的历史执行证据，全量canonical缓存以最终3316份manifest为准。

原生官方 cooperative R50 本机完整参考：mAP `0.1787047145084463`；AMOTA `0.13099317775947647`。这是本机原生参考，不宣称与外部论文数字数值复现。

通信统计使用 `len(payload)`，包含16字节prefix、JSON header、字段名/shape/dtype、计数、scene/token/timestamp、选择索引和INT8 scale；不使用pkl大小代替消息大小。三个feature字段为 query_feats/query_embeds/cache_motion_feats；ref_pts与pred_boxes保持FP32。

口径：**feature_message_only**，不包含相机图像、在线位姿/标定流、IP/链路封装或重传；接收端使用既有本地标定及官方帧映射。原方法已有活跃轨迹阈值和接收端空间筛选，本阶段未将其误称为新增选择机制。

| 条件 | seed | B/frame | P95 B | mAP | AMOTA | pack ms | unpack ms | cooperative ms | pack+unpack+cooperative ms | pipeline total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_FP32 | 0 | 42636.55880579011 | 75885.0 | 0.1787047145084463 | 0.13099317775947647 | 0.43942642548933086 | 0.2009265009369888 | 325.2306944903903 | 325.8710474168166 | 388.1999568208085 |
| FEAT_FP16 | 0 | 22100.71019300362 | 39021.0 | 0.17862304150288566 | 0.13151599940359307 | 0.4955100793167939 | 0.19030861518381686 | 323.06740559831013 | 323.7532242928107 | 387.19273010980385 |
| FEAT_INT8 | 0 | 12161.645355850422 | 21043.0 | 0.17826884799399076 | 0.1289974987259437 | 0.6135048959437845 | 0.2258655557412826 | 321.9980497725824 | 322.8374202242674 | 383.6691866013619 |
| CONF_50 | 0 | 22551.85404101327 | 38368.0 | 0.12438415421610315 | 0.10264331430614508 | 0.4432976474392591 | 0.16759006692909603 | 325.1660110729767 | 325.77689878734503 | 387.4234007059854 |
| CONF_25 | 0 | 12521.402291917973 | 19596.0 | 0.09472202825649571 | 0.07627763857126126 | 0.42661708688402716 | 0.15177300751276807 | 322.8964854653009 | 323.4748755596977 | 384.4104581366366 |
| RANDOM_50 | 0 | 22552.91164053076 | 38367.0 | 0.13493297843760674 | 0.08371437513429147 | 0.58738975658497 | 0.17519717308696778 | 322.59327309047137 | 323.3558600201434 | 385.77471336457205 |
| RANDOM_50 | 1 | 22552.915259348614 | 38367.0 | 0.13424854959137705 | 0.08409972411003269 | 0.5906799715694269 | 0.17440659533308797 | 324.735726261952 | 325.5008128288545 | 388.08856019515895 |
| RANDOM_50 | 2 | 22552.921893848008 | 38367.0 | 0.13395545081930121 | 0.08583248566861239 | 0.5890503851308915 | 0.17233137963946338 | 319.0496886870329 | 319.81107045180335 | 381.70768845906537 |
| RANDOM_25 | 0 | 12522.713811821472 | 19597.0 | 0.10530984990167995 | 0.042954093288874895 | 0.5739074294299017 | 0.1620046903057996 | 324.2751337822002 | 325.0110459019359 | 386.5883778314037 |
| RANDOM_25 | 1 | 12522.688480096502 | 19597.0 | 0.10625357133278944 | 0.04765930449263711 | 0.5735865474050298 | 0.1688213358579752 | 324.658364869723 | 325.400772752986 | 387.71525282208603 |
| RANDOM_25 | 2 | 12522.737937273823 | 19597.0 | 0.10662084707845552 | 0.04599458190040979 | 0.5759443437761138 | 0.16776671737629573 | 325.8957862675396 | 326.639497328692 | 389.160433099733 |

延迟均为实际测量均值；每帧及P95延迟在各case JSON中。pack/unpack为CPU编码/解码；cooperative为GPU同步包围的scatter/forward/输出；pipeline total还包括图像/cache加载和接收身份检查，不含评测。路侧模型生成缓存的时间另计，未将缓存回放冒称双端在线网络总延迟。

随机50%/25%分别使用预先固定seed 0/1/2；神经网络seed、FP32/TF32配置保持一致；每个case重新构建模型及tracker，从scene起点运行。按预测confidence取ceil(N×比例)，非空至少1，所有五字段使用同一索引且保留原始相对顺序。

FP32回归要求全部五字段torch.equal且完整3316条预测tensor/ID/输出结构精确一致；没有放宽容差。检测和跟踪使用同一官方SPDDataset.evaluate协议。

时序诊断按 `scene + native reset session + predicted obj_idxes` 匹配，禁止按行号或GT ID配对。统计为相邻实际观察帧中的同一预测ID；生命周期、cosine和相对L2分布见 [temporal.json](../artifacts/core/stage3/temporal.json)。几何时序比较尚未做坐标统一，不给少发/历史复用收益结论。

实际匹配 `42772` 对，原生 reset session `22` 个。

| 字段 | cosine P50 | relative L2 P50 | relative L2 P95 |
|---|---:|---:|---:|
| query_feats | 0.9995374083518982 | 0.030414793640375137 | 0.1879075944423667 |
| query_embeds | 0.9999871253967285 | 0.005784191191196442 | 0.03682948015630245 |
| cache_motion_feats | 0.9998332858085632 | 0.018261088989675045 | 0.12207392081618301 |

梯度探针：仅scene `0000`，batch1，1/3/20为同一运行中的里程碑，总计最多20次adapter更新；官方参数和BN保持冻结，原生历史detach规则不变，观测nan_to_num之前的任务loss。不使用0058或任何val参数更新，也不开始最终动态通信模型训练。

| 里程碑 | 状态 | peak allocated B | peak reserved B |
|---|---|---:|---:|
| 1 | PASS | 382652416 | 473956352 |
| 3 | PASS | 389658624 | 476053504 |
| 20 | PASS | 389683200 | 476053504 |

该探针仅证明所执行的冻结训练路径及资源占用，不代表正式创新训练、完整epoch或全模型训练可行。

原生接收范围审计：32帧中 `9` 帧有有效路侧query。本次为末尾token `000033–000041`，每帧1个；其余23帧发送缓存非空，但全部被原生坐标变换后的范围过滤去除。详细逐帧映射见 [train_receiver_audit.json](../artifacts/core/stage3/train_receiver_audit.json)。

首帧零梯度来自实际接收query为0，未进行参数更新。完成探针时按同一32帧片段原生顺序重放3次；空接收帧只做冻结、无梯度前向预热（共 `69` 帧），有效帧才更新adapter，实际更新严格为20次。时间回退重置和历史detach沿用原生路径，没有扩大接收范围、添加辅助loss或改用val。

官方参数和buffer前后SHA256均为 `429c7b296dccb196030c4a45b54a2a117ff13587518e067b3580843c1646de67`；官方可训练参数数目与参数gradient数目均为0。adapter初始identity已实测；20步原始task loss（nan_to_num之前）、接口梯度、参数增量和GPU峰值记录见 [train_probe/result.json](../artifacts/core/stage3/train_probe/result.json)。

证据：[最终门禁](../artifacts/core/stage3/stage3_gate_checks.json)、[执行记录](../artifacts/core/stage3/execution.json)、[计划与split身份](../artifacts/core/stage3/plan.json)。

历史 Stage3 诊断入口已在一次性精简中移除，最终诊断结果直接复用。当前主实验恢复入口为项目根目录下的 `python scripts/run_main.py --resume`。
