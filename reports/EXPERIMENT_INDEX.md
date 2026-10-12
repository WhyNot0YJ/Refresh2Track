# 实验索引

2026-10-10 研究索引修订：`ACTIVE = V2_BUDGET_CONDITIONED_REFRESH`；`RETIRED = BF_VALUE_H8 / BF_VALUE_H1 / VALUE_V1_DIAGNOSTICS`。

用户已定向授权删除 H 专用代码、标签、权重、预测、通信和大日志，保留轻量事实记录。实际执行状态以 [H_RETIREMENT_MANIFEST.json](budget_free/H_RETIREMENT_MANIFEST.json) 和 [退役摘要](budget_free/RETIRED_VALUE_V1_SUMMARY.md) 为准。原索引字节保存在 `budget_free/EXPERIMENT_INDEX_BEFORE_H_RETIREMENT.md.gz`；以下历史报告原文不改写。

| 实验 | 入口与证据 | 状态 |
|---|---|---|
| Stage1/2/3、R50 对齐 | `artifacts/core/`；`reports/STAGE*_FINAL.md`、`reports/R50_PAPER_ALIGNMENT_FINAL.md` | 历史成果保留 |
| V1 | `scripts/run_main.py`；`experiments/main/`；`reports/MAIN_EXPERIMENT_FINAL.md` | 历史正式结果保留 |
| V2 多预算刷新 | `scripts/v2_run.py`；`experiments/v2/`；`reports/v2/V2_FINAL.md` | 当前研究主线；MLP 与 Transformer 的 B1/B2/B3/B4 全部保留 |
| V2 Bug 审计 | `audits/v2_bug_audit/`；`reports/v2/V2_BUG_AUDIT.md` | 审计和历史溯源缺口说明保留 |
| 2×2 刷新消融 | `scripts/refresh_ablation.py`；`experiments/v2/ablations/refresh_2x2/` | A/B/C/D 已完成；结果和冻结身份保留 |
| 无预算收益监督 H8/H1 | `reports/budget_free/VALUE_V1_FINAL.md`；训练标量、完整精度 CSV、协议快照 | RETIRED；原160000阶段的限制与1816帧 partial 历史单独保留；独立6632/7000完整评估不回写旧计数 |
| VALUE_V1 诊断 | `reports/budget_free/VALUE_V1_DIAGNOSTICS.md`；`value_v1_diagnostic_scalars.json` | RETIRED；保留 D0 五折20轮诊断，不能当作正式40轮模型的独立测试 |

原始数据、转换数据、官方权重、共享 train/val query cache、Docker 环境和原生源码保留原位置。此次不运行训练或 official val。历史保护快照保存在 H 轻量归档中；本次独立的定向 V1/V2 保护检查见 `budget_free/H_PROTECTED_BASELINE.json`。

历史工作区原始保护结果仍为 FAIL：539 份选定 SHA 中3份变化，348条旧路径缺失（116文件、232目录）。这与本次 `AUTHORIZED_H_RETIREMENT` 分开记录，不能把旧 FAIL 改成全盘 PASS。

后续只使用一个共享参数的原 V2 MLP，接收 B1/B2/B3/B4 条件（3474/6382/9290/12198 B/frame，实际字节以 serializer 为准）。保留当前特征、最后实收缓存、差值与状态输入，Task+单边超预算 Rate、INT8 和几何逐帧更新。B1 是现有 seed0 的 AMOTA—平均通信候选；保留 B2/B3/B4 完整结果，不保证 AMOTA 随预算单调增长，当前选择参考过 official val。Transformer 保留结构对照。

待另行实验指令再核验 MLP B1 与相近通信 Periodic 的重复性、完整预算曲线和真实通信行为。本次取消 H8 额外 seed、H1/H8 对照与反事实正例复核，不启动 Delta+32，不恢复 Top-K、TTL/max_staleness 或四帧强刷。计划修订见 [H_RETIREMENT_PLAN_REVISION.md](budget_free/H_RETIREMENT_PLAN_REVISION.md)。
