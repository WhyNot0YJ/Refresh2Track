# CoopTrack：退役 H1/H8 路线、释放磁盘空间、回归 V2 多预算主线

项目根目录：`D:\CoopTrack_Audit`。

## 0. 本次授权与目标

用户已决定：停止 Budget-Free VALUE_V1 / H1 / H8 / D0 / D1 的研究路线，只保留少量结果文档和轻量指标，删除其专用代码、训练标签、模型权重、预测、通信轨迹和大日志，释放本机磁盘空间。研究回归原 V2 的 B1/B2/B3/B4 多预算刷新方案。

**这是对先前“所有旧结果禁止删除”的定向更新：仅解除 H 路线的产物与专用代码保护，不解除 V1、原 V2、多预算模型、原始数据、官方权重、Periodic、几何×latent 2×2 消融或原 V2 Bug 审计的保护。**

本次要实际完成定向清理，不是仅输出清理建议；不能把生成脚本或 Dry-Run 成功当作删除成功。不要因为旧保护清单包括 H 文件，就把本次用户明确授权的清理永久搁置；同时不得篡改原有保护清单来假装文件从未存在。

本次不训练新模型、不新增数据集、不运行完整 official val、不优化阈值/损失、不增加新刷新策略。

## 1. 先定位和检查，不重新做全盘审计

1. 读取项目 README、实验索引以及 H 路线最终报告、诊断报告、运行状态和已有清理报告。
2. 核对真实目录布局、Windows/WSL/Docker 路径映射、当前进程及未提交改动。
3. 只完整统计 H 专用目录及相关依赖，不重新为全部原始图像、数据集、query cache 做全量哈希或复制备份。
4. 以下均为已知的候选路径，不是可以不经核对就删除的整棵目录授权。路径不存在时记录，不猜测同名替代路径，也不重建空目录。

```text
scripts/budget_free/
configs/budget_free/value_v1.json
experiments/budget_free/value_v1/
experiments/budget_free/value_diagnostics_v1/
logs/budget_free/value_v1.log
reports/budget_free/
```

其中常见的 H 大型产物包括：

```text
value_v1/dataset/d0/
value_v1/dataset/d1/
value_v1/ 下的 round0/final H8/H1 checkpoints、optimizer 状态
value_v1/ 下的预测、逐帧通信、分支回放、临时文件和重复源码归档
value_v1/final_official_val_7000/ 下的完整预测、通信记录及冗余证据
value_v1/ 下原 160000 阶段的 1816 帧 partial 预测
value_diagnostics_v1/ 下的逐样本 NPZ、OOF 数组、十次拟合过程和大型中间诊断文件
```

所有类别均需用文件路径、所属实验、配置和引用证据确认。不得对全项目使用 `*H1*`、`*H8*`、`*value*` 之类通配删除。

## 2. 与活动进程、旧任务的关系

- 检查是否仍有 H1/H8 标签生成、拟合、推理或诊断进程。
- 对能通过 PID、命令行、工作目录、输出目录明确认定属于本次退役 H 路线的任务，优先正常退出；不再为了产出 H 结果继续计算。
- 禁止结束无关 Python、Docker 服务、V1/V2 任务、几何 2×2 任务或其他用户进程。
- 若共享进程同时执行原 V2 和 H 评估，不强杀。只在安全边界移除尚未开始的 H 任务；无法安全隔离的目标暂缓删除，不影响其他已经确认安全的 H 文件清理。
- 不移动、删除仍有写入者的文件，不执行“先删文件、让进程继续跑”。
- 先前“论文证据收敛”计划包含 H8 额外 seed 评估。本次取消这些尚未执行的 H 项目。保存旧计划作为历史，建立小型计划修订说明；不得偷偷改写已经冻结的旧运行身份。
- 若该计划已经有 H-only 输出，按本次相同规则先保留摘要再清理；混合 V2/H 的表格与目录不能整体删掉。
- 本次也不启动先前提出的 Delta+32 兜底方案，不恢复任何 TTL/max_staleness/Top-K。

## 3. 删除前先保存轻量研究记录

优先使用现有 `reports/budget_free/`，没有该目录再创建，不新建复杂归档框架。

只保留以下必要内容，不复制整套 H 实验目录：

```text
reports/budget_free/
  VALUE_V1_FINAL.md                      # 原最终报告，原文保留
  VALUE_V1_DATA_DIAGNOSTICS.md            # 原诊断报告；实际文件名可沿用
  RETIRED_VALUE_V1_SUMMARY.md             # 一份退役摘要与限制说明
  value_v1_results.csv                   # H8/H1完整标量指标及明确来源
  final_h8_training.json                 # 40轮标量训练记录（存在才保留）
  final_h1_training.json                 # 同上
  value_v1_protocol_snapshot.json        # 小型协议/特征schema/权重身份信息
  H_RETIREMENT_MANIFEST.json             # 实际清理清单与校验结果
```

已有等价文件直接复用，不为匹配名字制造副本。原报告仍按原字节保存，不抹除不利结果。归档目标是几 MB 级；报告必须使用实际字节统计，不能为达到固定大小而丢失唯一结果。

### 3.1 需要保留的结论与数值

从真实 JSON/CSV 读取完整精度；以下小数只帮助定位，不能据此伪造或反写原结果：

- D0=4096，D1=2048，总计6144；H8/H1 正例470/69；Pilot包含于D0。
- H8使用八帧**平均**收益，H1使用首帧收益；两者epsilon=0.01。
- 输入2324维，无预算；教师冻结，学生MLP监督拟合40轮。
- H8：2465.701448 B/frame，mAP=0.362236，AMOTA=0.357951，IDS=217。
- H1：2145.797949 B/frame，mAP=0.353286，AMOTA=0.333465，IDS=292。
- 同时保留AMOTP、MOTA、P95、更新比例等已有标量，不只保留有利项。
- D0五折20轮诊断：H8 OOF PR-AP=0.158256；H1=0.023220。它们不是正式40轮模型的独立官方测试结果。
- 保留“H8平均收益算术核验通过、后续帧代理收益存在、跨scene预测较弱、H1阈值决策失败”的事实；不写成“标签全错”或“网络完全没学会”。
- 保留单seed、固定continuation、教师/学生可观测信息差异，以及未完成的随机性/安慰剂控制边界。
- 原160000限制触发和后续独立6632次完整评估均保留摘要，不改历史计数。

保留两份final H权重的**哈希字符串和训练身份**即可，不需要保留权重文件本体。保留核心配置与特征字段定义，不保留整套可执行H源码。

### 3.2 归档验证与可复现性边界

1. 复制需要保留的小文件后，校验副本与源文件SHA相同，解析JSON/CSV并核对指标。
2. 在退役摘要中逐项指出：哪些预测、训练标签、权重、代码已经主动删除；原报告指向这些文件的历史链接将不再可访问。
3. **不得因仍保存哈希和结果表，就声称H路线还能完整重跑、独立复评或复现。** 本次用户选择保留结论、放弃本地完整复现能力。
4. 不重新计算上万份即将删除的大文件的哈希；优先沿用已有manifest，按需核验小型保留文件。没有历史哈希就如实标N/A。
5. 保留项未成功归档前，不得删除其唯一来源。

## 4. 精确保护原 V1/V2 与共享依赖

下列内容绝对不属于本次删除范围：

```text
experiments/main/**
experiments/v2/**
scripts/v2_*.py
scripts/main_*.py
scripts/run_main.py
configs/main_experiment/**
configs/main_experiment_v2/**
reports/MAIN_EXPERIMENT_FINAL.md
reports/v2/**
audits/v2_bug_audit/**
code/CoopTrack/**
code/mmdetection3d/**
checkpoints/**
datasets/**
data/**
data_infos/**
query_cache/**
docker/**
patches/**
```

路径以实际索引为准。特别保护：

- 原V2 MLP final、Transformer final，及各自B1/B2/B3/B4全部结果。
- Periodic P1/P2/P4/P6/P8/P10/P12/P16/P24/P32、INIT_ONLY。
- FULL_FP32、INT8参考。
- 已完成几何×latent 2×2实验的A/B/C/D、脚本、配置及证据。
- 原生CoopTrack与共享codec、receiver cache、train/val数据。
- 已经执行的原V2重复seed/论文补充实验。

H生成过程复用的train/val query cache是**共享输入**，不是H训练标签，不得删除。

已知原V2 final权重身份：

```text
MLP:
d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c

Transformer:
58bda699962f479a7556e75c6ad3b0389d6a909ca7b798e5eece2ea7f2fc65a3
```

## 5. H专用代码与配置的删除要求

删除前检查Python imports、动态导入、CLI命令、配置、README、PowerShell/Docker调用及paper_validation入口。

区分：

- 仅H路线使用的代码：在轻量摘要保存方法说明后删除，包括教师、标注、评分、学生、H评估、H诊断及其专用测试。
- 仅记录历史路径的报告/manifest引用：在退役摘要标记失效，不因此永远阻止删除。
- 仍被原V2运行链路使用的共享代码：保留，不得因为位于H目录就删除。
- 非冻结的论文验证入口导入H helper：优先删除已经取消的H分支和导入。若V2分支也需要通用功能，最小化抽取到已有公共模块，验证行为，记录新修订；禁止改动原冻结V2源码来迁就删除。
- 依赖关系无法确认的单个文件：标为`KEEP_SHARED`或`BLOCKED_ACTIVE`，仅暂缓该文件，不把整个H目录所有清理一概搁置。

不要为了删除几十KB代码创建一套庞大兼容层。磁盘回收优先处理已经归档后的H独占标签、预测、通信和大日志。

## 6. 生成并实际执行定向清理脚本

复用现有维护目录，建议创建：

```text
scripts/maintenance/retire_value_v1.ps1
```

脚本：

- 默认`Dry-Run`；`-Apply`才真正删除。
- 清单包含解析后的精确路径、长度、用途、保留/删除分类、依赖检查结果。
- 删除范围必须在项目根内，拒绝根目录、保护目录、链接/junction/reparse及其越界目标。
- 不把整个`reports/budget_free/`列为删除目标，它承载保留结果。
- 不把包含混合方法的`experiments/`或`paper_validation/`整棵列为删除目标。
- H专用目录全部子项清单确认后可按白名单删除文件，再自底向上移除空目录；清单生成后出现的新文件或变更必须重新核查，不用宽泛递归吞掉。
- 每次删除记录实际成功/失败；文件已经不存在记`ALREADY_ABSENT`，不能算本次释放空间。
- 不使用`git clean -fdx`、全盘通配删除、强制取消权限保护或删除整个项目。
- 不创建整个项目或H大型目录的备份/回收副本，因为目的是释放空间。

执行顺序：

```text
检查真实目录与活动进程
→保存轻量结果并验证
→建立H精确删除白名单
→Dry-Run
→通过后按当前合法权限运行-Apply
→验证原V2不变
→记录实际空间释放
```

用户已明确授权该范围，无需反复询问每个文件是否可以删除。若平台提供正常审批入口，使用正常审批；若执行被`blocked by policy`拒绝，停止破坏性尝试，保留脚本与清单并报告本机手动命令。**不得通过改Shell、Python、Docker或间接包装绕过拒绝。**

## 7. 验证：区分定向删除与意外损坏

不执行新训练或完整val。仅做必要轻量验证：

1. H保留文档、CSV、40轮标量记录可打开；归档时的SHA一致。
2. H删除清单中的大文件确实不存在；剩余H共享或活动文件单列原因。
3. 原V2两个final权重SHA不变；V1/V2核心配置、源码、manifest与已冻结身份一致。
4. B1～B4完整结果、Periodic、2×2、原V2 Bug审计仍可读。
5. 当前仍要使用的V2/paper_validation入口不再引用被删除的H可执行代码。
6. 必要的关键模块import/CLI/codec小测试通过；使用禁写bytecode方式避免无意义重新生成缓存。
7. 基于成功删除文件的实际长度报告回收字节，并报告磁盘可用空间前后值。逻辑字节与实际卷空间可能不同；并发写入另记，不虚构释放量。
8. 旧全项目保护快照若因本次授权删除而FAIL，保留原状态；新增`AUTHORIZED_H_RETIREMENT`差异记录，不能篡改旧快照、把旧FAIL改成全盘PASS，也不能把预期删除误报为原V2被破坏。

## 8. 回归原V2多预算路线：只整理计划，不自动跑新实验

建立或更新一份简短的研究索引：

```text
ACTIVE = V2_BUDGET_CONDITIONED_REFRESH
RETIRED = BF_VALUE_H8 / BF_VALUE_H1 / VALUE_V1_DIAGNOSTICS
```

后续主线：

- 一个原V2 MLP共享参数，接收B1/B2/B3/B4条件，不是四个独立模型。
- 预算为3474/6382/9290/12198 B/frame；实际字节以serializer为准。
- 保留原来的当前特征、最后实收缓存、差值和状态输入。
- 保留Task+单边超预算Rate、INT8、几何逐帧更新；不恢复Top-K填满或四帧强刷。
- B1是现有seed0下较好的AMOTA—平均通信候选，不是证明所有预算/种子/指标均最优。
- B2/B3/B4全部保留并完整报告。预算输入调节通信量，并不保证AMOTA随预算单调增长；不得为掩盖不利数据而删结果。
- Transformer继续保留结构对照，不删除，也不立即扩大网络或重训。
- 下一批实验优先核验原MLP B1与相近通信Periodic的重复性、完整预算曲线及真实通信行为。
- 先前安排的H8额外种子、H1/H8对照、反事实正例复核全部取消；不把已删除H模型加入待执行任务。
- 原V2是否补训练种子、增加其他基线或修改loss，留待单独的新实验指令；本次清理完成后停止。

不能把冻结输入B1的多预算模型改名为无预算训练模型。不得宣称当前选择完全未参考official val。

## 9. 最终回复

```text
H_METHOD_RETIREMENT
STATUS=COMPLETE/PARTIAL/BLOCKED

ARCHIVED
result_reports=
results_csv=
training_scalar_logs=
protocol_snapshot=
archive_total_bytes=

REMOVED
h_only_source_files=
h_only_config_files=
d0_d1_dataset_bytes=
h_checkpoint_bytes=
h_prediction_and_communication_bytes=
h_diagnostic_and_log_bytes=
files_removed=
directories_removed=
logical_bytes_reclaimed=
volume_free_before=
volume_free_after=

REMAINING
shared_files=
active_or_uncertain_files=
policy_blocked_items=

PROTECTED
V1=
V2_MLP_FINAL_SHA=
V2_TRANSFORMER_FINAL_SHA=
B1_B2_B3_B4_RESULTS=
PERIODIC=
REFRESH_2X2=
NATIVE_SOURCE_AND_DATA=
RUNTIME_IMPORTS=

RESEARCH
active=V2_BUDGET_CONDITIONED_REFRESH
retired=VALUE_V1_H8_H1
pending_H_runs_cancelled=
new_training_started=NO
new_official_val_started=NO

REPRODUCIBILITY
H_full_local_rerun_available=NO（清理完成后）
H_result_documents_retained=YES
V2_reproduction_assets_preserved=

REPORT
reports/budget_free/RETIRED_VALUE_V1_SUMMARY.md
```

本次核心要求：**真正释放H路线占用的空间，保留轻量事实记录，原V2多预算成果完整不动。清理完成即停止，不再启动一条新研究路线。**
