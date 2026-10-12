# V3 Fixed/Shared 思路与废止说明

**STATUS=RETIRED / 已废止；日期：2026-10-12。**

此文仅用于保留历史研究思路，不是可执行的训练计划。用户明确要求删除 V3 相关环境、数据和 checkpoint，仅保留思路文档。

## 原思路

冻结原生 CoopTrack，仅训练 MLP Scheduler，比较 Fixed-4K、Fixed-8K、Fixed-12K 与 Shared-4/8/12K。Fixed 始终使用对应的 4000/8000/12000 B/frame 名义预算；Shared 按 scene 与 epoch 轮换预算，研究固定预算专用模型与多预算共享模型的通信—跟踪性能权衡。使用真实序列化消息字节数、UPDATE/REUSE 缓存及 hard gate + STE。

## 废止原因

1. **研究目标调整。** 后续 V4 研究不指定每帧预算时如何通过任务损失与通信正则学习刷新决策，主要比较相近实际通信量的 Periodic 基线；V3 的固定预算专用/共享对照不再作为开发主线。
2. **原目标对预算内通信缺少持续约束。** V3 使用单边超限惩罚；预计通信未超过可实现预算时，该项不再推动进一步节省通信。预算是软约束，不能保证 4K/8K/12K 形成不同的真实工作点，也不是硬性上限。V4 移除预算输入与对应分支，对可选 UPDATE 概率总和持续施加通信正则；R 仅作为训练集归一化常数。
3. **对照未完成。** Fixed-4K 完成 epoch 21，epoch 22 中断；Fixed-8K、Fixed-12K 与 Shared 未开始正式训练。缺少完整 Fixed/Shared 对照和多训练 seed 证据，不能宣称专用模型或共享模型更优。
4. **评价证据有限。** 同一 official val 同时用于选模、早停与结果报告，存在 selection bias。废止是路线取舍，不等同于已证明 V3 无效或 V4 一定更优。

## 废止前的历史观察

| 模型 | 最后状态 | 历史结果 |
|---|---|---|
| Fixed-4K | 暂停；完成 epoch 21，中断 epoch 22 | best epoch 15：AMOTA 0.36890589976964766，mAP 0.3630987247583841；真实序列化均值 5638.15621230398 B/frame，P95 9552 B/frame |
| Fixed-8K / Fixed-12K / Shared | 未开始正式训练 | 无可比较的训练结果 |

Fixed-4K 的 best 工作点超过名义 4000 B/frame，说明名义预算不能替代实际通信量。上述数值在删除前从原报告和归档说明读取，仅保留为历史文字记录；其预测、机器指标和 checkpoint 已按用户要求删除，现已不能从本地原件复核或恢复。单个未完成实验不能支持统计显著性或预算单调性结论。

## 清理范围

| 删除路径 | 文件数 | 逻辑字节数 |
|---|---:|---:|
| `experiments/v3_budget_specialization` | 126 | 1067201952 |
| `configs/v3_budget_specialization` | 7 | 66974 |
| `logs/v3_budget_specialization` | 3 | 4557355 |
| `scripts/v3_budget` | 9 | 151011 |
| `archives/v3_fixed_budget_20261012` | 1166 | 1135313494 |
| `scripts/maintenance/archive_v3.py` | 1 | 21297 |

**清理已完成**：删除 1312 个文件、2207312083 字节（约 2.06 GiB 逻辑文件空间）。以上所有路径均已核验不存在。

V3 独有源码、配置、日志、训练/验证数据、曲线、checkpoint，以及包含环境/源码快照的归档副本均删除。没有发现 V3 专用 Docker 镜像、容器或卷；当前运行环境由 V1/V2/V4 共用，因此保留。共享原始数据集、infos、query cache、官方/V2/V4 checkpoint 保留。

V4 正在使用的 native_hooks 按字节迁入 `scripts/v4_budget_free/native_hooks.py`，保留原辅助函数和内部历史命名以避免改变训练计算。V4 的原 manifest 和 checkpoint 身份不修改；`experiments/v4_budget_free/audit/v3_retirement_transition.json` 显式记录 import/路径与审计、报告维护改动的前后散列，并独立保护剩余共享资产。该记录属于 V4 的恢复依赖，不包含 V3 数据或权重。

保留的历史方案：[V3 原始思路](../../CoopTrack_V3_Fixed_Shared_Training_Prompt.md)。V4 迁移提示词中的 V3 归档章节已经标记失效，不应再恢复 V3。

## 清理后核验

`python -B scripts/maintenance/check_v4_retirement.py` 通过：V4 原 manifest 身份保持一致；辅助模块算术与插入位置自检通过；1,417 个保留文件散列和大小核验通过；恢复时的 freeze 检查通过；训练计算文件及此次维护文件的再次改动均会被拒绝。

在原 Docker Python 环境中以 CPU 加载 V4 `latest.pth`（epoch 15）与 `best.pth`（epoch 12），两者记录的完整 identity 均与迁移后的 runtime.identity 一致；未改写这些 checkpoint。

V4 报告已重新生成，移除了已删除 V3 产物的结果引用。当前 epoch 16 的 GPU worker 未中断；旧协调器暂停等待本轮结束，后台程序将在完整 checkpoint 写入后重新启动支持独立资产核验的新协调器。交接状态记录在 `experiments/v4_budget_free/audit/v3_retirement_handoff.json`，启动时为 `WAITING_FOR_COMPLETE_EPOCH`，并非已完成下一轮训练。
