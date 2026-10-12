> **V3 归档要求已废止（2026-10-12）**：用户随后明确要求删除 V3 环境、数据和 checkpoint，仅保留思路文档。因此本文第 1 节及其他“保留 V3 原件/归档”的历史要求已被本次清理取代，不应重建 V3。V4 方法与训练协议仍有效；现行 V4 使用独立共享资产保护记录，不再依赖 V3 归档。见 [V3 废止说明](reports/v3/V3_FIXED_SHARED_FINAL.md)。

# CoopTrack V3 归档 + V4 Budget-Free MLP：Codex 执行提示词

> 将本文件完整交给在 **CoopTrack_Audit 项目根目录**工作的 Codex。按顺序执行，不要把本文中的示意路径或历史数值冒充现场核验结果。

## 0. 任务与最终研究目标

你是该项目的研发执行者。请先**安全归档既有 V3 Fixed-4K / Fixed-8K / Fixed-12K / Shared-4_8_12K**，保留所有可追溯证据，再在**独立的 V4 目录/入口**实现和运行一个 **Budget-Free MLP**。

研究问题已明确调整为：

> 在不指定每帧通信预算的情况下，学习对每个路端 Track Query 执行 UPDATE 或 REUSE，通过联合优化协同跟踪任务损失和高维特征通信损失，在较低实际通信量下保留跟踪性能。核心检验是**与相近实际通信量的 Periodic 基线相比是否更有效**。

本次不研究“给定4K/8K/12K预算是否严格满足”；不以 AMOTA 随预算递增为成功条件。**严禁**为了得到好看的论文曲线捏造预算满足、修改已经完成的旧指标或事后挑选有利基线。

预计项目位置：Windows `D:\CoopTrack_Audit`，Docker 可能映射到 `/mnt/d/CoopTrack_Audit`。**首先根据真实工作区确认根路径、实际源码、Git 状态、配置及运行进程；不要只凭本文推断路径存在。**

---

## 1. 第一优先级：冻结并归档 V3，禁止破坏既有研究结果

### 1.1 只读盘点

检查并记录：

- 当前仓库 Git HEAD、工作区改动、子模块/固定 CoopTrack 上游 commit（历史基准 `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`），以及 Python/Docker/依赖版本。
- 所有 V3 固定预算与 Shared 模型的**实际**代码入口、配置、实验状态、每轮/最佳 Checkpoint、训练日志、验证指标、曲线、运行命令、训练种子、优化器状态及 README。
- 四条实验是否已经结束、停止于哪个 epoch、最佳 epoch、最后一次验证、实际 B/frame、AMOTA；如缺文件则明确写“未发现”，不要补造。
- 所有正在运行的 V3 训练/评估 PID、容器内外进程、启动参数与写入目录；也检查计划任务/脚本是否可能继续写入旧路径。
- V2 MLP、V2 Transformer、Periodic、FULL FP32/INT8、已验证协议及原始 query cache 的依赖关系，区分“V3 独有”与“所有版本共享”。

**运行进程安全门禁：** 若 V3 仍在写日志/Checkpoint，先输出 `WAIT_FOR_V3_PROCESS_CONFIRMATION`，列出 PID、实验名和安全处理建议，**请求用户确认停止或等待完成**。不得擅自 kill、不得在写入中的 checkpoint 上声称完成一致性归档、不得为了新版本抢 GPU 而终止其他任务。

### 1.2 归档方式

只有通过运行状态门禁后，创建一个独立的、带日期的归档目录，例如：

```text
archives/v3_fixed_budget_20261012/
├── README_ARCHIVE.md
├── manifest.json
├── sha256_manifest.txt
├── git_status.txt
├── git_diff.patch
├── configs/
├── source_snapshot/
├── checkpoints/
├── logs/
├── metrics/
└── figures/
```

要求：

1. **按真实文件列表**复制/快照 V3 独有产物，保留原相对路径映射与原文件名；不假设已有 `experiments/v3` 等目录。
2. 不能重新跑实验来“补齐”归档；未完成结果标注 `PARTIAL/INCOMPLETE`，已完成标注 `COMPLETE`。
3. 记录每个归档文件的来源路径、大小、SHA-256、归档后的 SHA-256、更新时间；抽样/必要时全量核对 hash 一致。
4. 如代码与 V2/V3 共用且存在未提交修改，归档实际相关源码快照及 `git diff --binary`，避免未来无法复现。
5. 大型共享数据集、路端 Query cache、官方 backbone weights、Docker 镜像可在 manifest 中**引用锁定路径+校验信息**，无需重复复制多个 GB；不得移动或重写这些共享资产。
6. V3 原目录及 checkpoint 原件**不删除、不覆盖、不重命名为 V4**；归档独立保存，并在 `README_ARCHIVE.md` 写明 V3 已停止作为新方法开发的主线。
7. 归档结尾输出逐实验清单：`Fixed-4K/8K/12K/Shared` 的代码身份、进度、已存 checkpoint、最佳验证结果、是否成功校验。

**不要运行 `git clean -fdx`、`rm -rf`、批量删除日志、重置 Git 工作区或覆盖现有 checkpoint。**

归档成功门禁：`V3_ARCHIVE_VERIFIED`；未成功则停止后续会写磁盘的阶段，明确报告原因。

---

## 2. V4 的方法定义：只改训练目标与预算条件，其他尽量复用

新版本统一标记为 **`v4_budget_free`**，必须采用独立目录、独立配置、独立 checkpoint、独立运行日志，**不得修改 V2/V3 原模型权重或实验数据**。

### 2.1 先核对旧代码再改

只读检查当前 V2/V3 真实实现，定位：

- Track Query 输入特征和构造处；既有每 Query **2324 维状态特征**（三组 current/last/delta 各768维，共2304维，加20维状态）；
- V3 MLP 中 Budget 数值究竟在何处拼接或嵌入；
- Scheduler 输出 logits、`sigmoid`、固定0.5阈值、Hard Forward + Straight-Through Estimator（STE）；
- 缺缓存时强制 UPDATE、接收端真实缓存、发端 mirror、scene/session reset；
- 路端 INT8 编解码、每次三组256维特征更新 **780 B**（768字节 INT8 值 + 三个 FP32 scale 共12字节）；
- CoopTrack 冻结骨干与原生 tracking/association task loss 反向路径；
- 每帧当前 `ref_pts`、`pred_boxes` 的传输与实际报文计数；
- V3 训练优化器、数据顺序、Checkpoint 与评估协议。

如现场与历史描述不同，**以实际源码为准，列出差异后采用最小兼容改造；不要编造类名、接口或参数。**

### 2.2 Scheduler 输入与输出

- **删除 Budget 作为神经网络输入**；不再拼接 `4000/R`、`8000/R`、`12000/R` 等固定预算，也不使用 budget embedding、budget token、FiLM budget conditioning 或任何间接预算条件。
- 保留原2324维 Query 状态，保留同一 V3 MLP 主体（除预算输入维度的必要修改，不新增 Transformer、GRU、全局聚合或其他网络）。
- 继续输出每个可选 Query 的 logit `z_i`、概率 `p_i=sigmoid(z_i)`。
- 必须保持确定性 Hard Gate，阈值 **0.5**；没有缓存的 Query 无条件 `FORCED UPDATE`，不能从旧缓存中伪造数据。
- **禁止**重新加入预算 Top-K fill、Gumbel/logistic 采样、max-staleness/TTL 强制刷新、预测 ID 数值/行号作为学习输入、GT 或未来信息。
- Fixed 模型中的常数 Budget 输入属于可以删除的冗余条件；Shared Budget 机制整体留在 V3 归档中，不带入 V4。

**前向决定的仍是 UPDATE/REUSE 二元通信动作，不是把真实当前高维特征与历史特征按概率做混合并冒称“节省了相同比例的通信”。**

### 2.3 训练前向与反向：保留原有 STE

对于已有接收端缓存的可选 Query：

```python
p = torch.sigmoid(logits)
hard = (p >= 0.5).to(dtype=p.dtype)
gate_ste = hard.detach() - p.detach() + p
```

- 前向输出与硬 UPDATE/REUSE 决策一致；
- 反向梯度由概率路径近似传回 MLP；
- `no_cache` 行直接强制 UPDATE，并从可选惩罚集合中排除；
- 训练/验证/推理的**硬决策、实际解码与缓存演化**语义一致；
- 不得通过向车端直接输送 sender 的当前 raw latent 绕过 REUSE；只允许 decoded packet 五字段进入协同模型。

不需要在本版额外研究 Hard-rate STE 或修改门限。这是后续诊断项，**不是本次前置条件**。

---

## 3. 核心变化：直接通信正则项，不设置目标 Budget

定义：

- `O_t`：第 t 帧已经存在合法接收缓存、可以自行决定 UPDATE/REUSE 的 Query 集合；
- `F_t`：第 t 帧因缺缓存而必须初始化的 Query 数；
- `N_t`：第 t 帧实际进入消息的 Query 数；
- `p_{t,i}`：可选 Query 的 UPDATE 概率；
- `c = 780 B`：一次三组 INT8 latent 更新的实际高维载荷成本（含 scale）；
- `R`：固定的**训练集通信成本参考常数**，优先复用已有 train-only `full_int8_mean_bytes = 12840.189523169913 B/frame`，由真实配置/统计核验，绝不从 val 调整。**R 只是无量纲归一化，不是预算上限。**

建议的每帧通信正则项：

\[
L_{\mathrm{comm},t}=\frac{c}{R}\sum_{i\in O_t}p_{t,i}
\]

多帧/多条 batch lane 的聚合：先按实际处理的帧计算，再对**真实有效帧数**取均值；不按所有可选 Query 个数再除一次，避免把“减少总字节数”误改成“减少平均刷新概率”。不计 padding，也不计不存在的 Query。

总目标：

\[
\boxed{L_{\mathrm{total}}=L_{\mathrm{native\ task}}+\lambda_{\mathrm{comm}}\,\operatorname{mean}_t(L_{\mathrm{comm},t})}
\]

- `L_native_task`：沿用 CoopTrack 的既有监督与损失定义，不擅自加 teacher/辅助损失。
- `λ_comm`：不可避免的任务/通信权衡权重，**不是4K/8K/12K预算**。本次只跑一个预注册候选：**`lambda_comm = 0.05`**，这是首轮工程试验值，**不宣称最优**；锁定于实验配置和 manifest，禁止用 official-val 指标反复搜索 λ。
- 固定开销 `52N_t`（3D参考点12B + 10维检测框40B）及强制初始化成本 `780F_t` 不可由本次 optional 决策消除，**日志里如实统计，但不作为需要对 scheduler 求梯度的正则项**。
- 如果现场发现当前 V3 的损失平均方式/梯度缩放与上述不一致，列出修改前后单样本数值和梯度差异，确保 Batch Size 不会隐式乘大 λ。

保留两套成本日志，**不得混用**：

1. `estimated_optional_bytes = 780 * sum(p_optional)`：训练 Soft 预计成本，只用于优化与诊断；
2. `actual_optional_bytes = 780 * number_of_optional_hard_updates`：实际 Hard 决策成本。

还要记录两者差值和概率在 `[0.45,0.55]` 的占比，检查 Soft/Hard gap，但**不要为了让指标好看修改 STE 或阈值**。

---

## 4. 统计口径：论文主结果与辅助结果分开

每帧真实硬决策：

\[
U_t=F_t+\sum_{i\in O_t}\mathbf1[p_{t,i}\ge0.5].
\]

**论文主通信指标：模型字段载荷 BPS，排除自定义 Header、ID、flag、网络传输层开销**：

\[
C_{\mathrm{payload},t}=52N_t+780U_t,
\qquad
\mathrm{BPS}_{10\mathrm{Hz}}=10\frac{\sum_t C_{\mathrm{payload},t}}{T}.
\]

必须同时保留：

- `feature_payload_BPS`、`feature_payload_mean_B_per_frame`；
- `feature_payload_latent_BPS`（单独三组 INT8 latent+scale）、`geometry_BPS`；
- `serialized_message_mean_B_per_frame`、`serialized_message_P95_B_per_frame` 和 `serialized_message_BPS`（按现有 `len(packet)`，含 JSON/ID/flag/前缀）；
- forced/optional UPDATE 数、optional UPDATE rate、实际 cache staleness；
- mAP、AMOTA、AMOTP、MOTA、IDS，准确记录 seed 和 checkpoint。

**注意：** Nominal 10Hz BPS = 平均 B/frame × 10；不能冒充网络实测吞吐量。当前 Full FP32 对照需用实际三组 FP32 与几何统计 `3124N_t`，Full INT8 用 `832N_t`，所有新旧方法保证同一 Query/字段、同一数据类型和相同过滤阶段。若历史基线只有完整报文日志，优先从逐帧计数/实际协议独立复算，无法复算则明确标注不可直接比较。

不得为凑齐 CoopTrack 原论文的 `285000 B/s` 而删除 `query_embeds`、事后把车端空间 mask 当成路端已过滤数据，或把量化带来的节省归功于更新调度。

---

## 5. 最少但必须通过的正确性门禁

在长训练和完整验证之前做 CPU/短片段自检，所有测试尽可能复用已有 V2/V3 审计代码；不能通过则停止。

1. **V3 只读保护**：归档前后关键源码、Checkpoint 和既有 V2/V3 报告 hash 不变；V4 运行不写入旧目录。
2. **无预算输入**：读取实际模型定义、调用参数、训练配置，证明 V4 Scheduler 不再接收 B、归一化 B 或 budget token，输入仍为2324维状态，输出与 Query 数对齐。
3. **Forward Hard**：构造 `p=0.49/0.50/0.51` 边界测试，确保阈值与实际发送 flags 一致；REUSE 的高维字段来自 receiver 最后实际解码值。
4. **通信增量**：其余字段与元数据相同，仅使一个 optional Query 从 REUSE 改成 UPDATE，序列化长度精确增加 **780B**；纯 latent payload 也增加780B。
5. **强制初始化**：无 cache 必须 UPDATE，跨 scene/session 正确清空；ID 行顺序变化不会误绑缓存；空 Query 帧安全。
6. **梯度**：用真实2～8帧小片段验证 `L_native_task` 与 `L_comm` 可分别沿合法 optional Gate 路径向 MLP 产生有限、非零的梯度（当存在有效监督/optional Query 时）；若某样本确无梯度，报告原因而非伪造。
7. **冻结原模型**：官方 CoopTrack backbone/association 等权重不更新，只有 V4 MLP 优化；不能访问未来 GT、未来图像或未发送 latent 作为接收端输入。
8. **参考对照**：FULL/Periodic 的旧评估与数据映射不可因 V4 改造而发生变化；保留版本哈希与受控对比证据。

输出 `V4_PREFLIGHT_PASS/FAIL`，包含每项实际命令、断言与证据路径。Fail 时不开始训练。

---

## 6. 首轮 V4 训练：仅一个 MLP，不展开调参矩阵

在通过门禁后，**只训练一个** `v4_budget_free_mlp_lambda005_seed0`：

- 数据：官方 train **46 scenes / 7362 cooperative samples**，沿用现有 R50、car-only、10Hz、路端固定 Query cache 与既定 train/val 配对；不重建数据集。
- 训练种子：`0`；保持 scene 内时间顺序和 batch lane 的各自时序/cache隔离。
- Batch Size：优先 `4`；仅遇到真实显存/内存问题才允许回退到 `2`，并在训练前记录原因，不能中途改变 batch 以继续同一实验又不标注。
- 优化器：复用 V3 当前已经验证的 AdamW、峰值 LR `1e-4`、weight decay `1e-4`、clip `5`、warmup+cosine 配置；**不要自行更换骨干、调参或增加结构**。
- 最大训练 `36 epochs`，每 `3 epochs` 做一次固定 official val 诊断，从 epoch>=12 开始在连续4次评估无提升后 early stop；**如果实际 V3 实现的调度细节不同，报告并固定 V4 的确切设置**。
- Checkpoint：从头训练，保留 `latest`、按预先锁定的 `val AMOTA` 选出的 `best` 和必要的恢复状态；**固定 λ，不因 official val 结果增训第二个 λ**。选模确实使用 official val，最终论文必须披露其不是独立、未触碰的测试集；不能声称 test 泛化显著性。
- 对同一 best epoch，同步记录真实 `feature_payload_BPS` 和 `serialized_BPS`；最高 AMOTA 不意味着最低通信，不能把单一 best 误写成 Pareto 最优。
- 可断点续训，但 resume 必须核对 experiment ID、seed、optimizer/LR状态、原模型权重哈希，不能复用 V3 的预算条件 MLP checkpoint 假装从头训练。
- 实际 train loss、Task Loss、Communication Loss、λ加权项、梯度范数、hard/soft UPDATE、强制刷新比例、概率分布、BPS 和学习率必须可追溯。
- 若出现 `nan/inf`、任务梯度消失、几乎全 REUSE 或近乎全 UPDATE，请记录诊断并停止自动扩展实验；**不要自行触发 λ 网格搜索、更换模型或新数据集实验**。

时间紧迫，训练过程无需临时增加新的 architecture ablation、2Hz 数据集或第二骨干。

---

## 7. V4 验证与核心论文对比

选好 Checkpoint 后，必须在完整官方 val（**21 scenes / 3316 samples**）按原生时序从每个 scene/session 正确 reset，做如下对比：

- 新 V4 Budget-Free MLP：至少先 seed0 完整评估，通过后做 seed **0、2025、2026** 的推理稳定性评估；这不等于三个训练 seed。
- 关键对照：**Periodic-16**，必须在尽可能相同的车端推理种子、同一锁定 infra cache、同一模型构建/重置协议和统计口径下配对比较；复用历史结果前先检查是否真的可配对。
- 如果已有 FULL FP32、FULL INT8、P32、INIT_ONLY、旧 V2 MLP B1、V3 的可靠旧结果，应在旁列作参考；不得因为 V4 需要“更好看”而重跑或筛选旧方法。
- 主比较是**实际相近 BPS 下的 AMOTA / mAP**。若 V4 与 P16 的真实 BPS 差距较大，明确说“非同等成本”，不得直接宣称 V4 优于相同成本的周期方法，也不得在 official val 上扫参数拼凑同成本。
- 可以附描述性 Pareto 图，但当前**单个 λ 的 V4 只能形成一个学习式数据点**，不能假装已经得到完整 Rate–Utility 曲线；无须启动多 λ 训练。
- 报告三 seed 的每组逐项 AMOTA、mAP、BPS，以及配对差均值、样本标准差（若实际跑齐）；**不要声称统计显著性或跨训练种子稳定性**。
- 保存逐帧通信计数、实际预测、原始 evaluator 输出、Checkpoint SHA、源代码 HEAD/dirty diff、配置、日期与运行命令。

论文故事应当精确表述为：**Task-aware temporal selective refresh of instance-level V2X features**。不宣称“通信必然越低性能越高”或“任何带宽条件下都满足硬约束”。

---

## 8. 交付目录与最终报告

文件路径必须以实际项目布局为准，建议新增：

```text
archives/v3_fixed_budget_20261012/         # 经过校验的 V3 只读快照
configs/v4_budget_free/                  # 单个 λ、完整训练配置
scripts/v4_budget_free_*.py              # 与旧版隔离的新入口/适配逻辑
experiments/v4_budget_free/checkpoints/
experiments/v4_budget_free/metrics/
experiments/v4_budget_free/communication/
logs/v4_budget_free/
reports/v4_budget_free/V4_METHOD_AND_AUDIT.md
reports/v4_budget_free/V4_FINAL.md
```

不要为了符合这个示例而挪动共享的 `query_cache`、官方权重、数据集、旧评估文件。

`V4_METHOD_AND_AUDIT.md` 必须包含：

- V3 归档目录、SHA 校验、进程状态和保留的结果；
- V4 原代码差异（特别是输入维度、删除 Budget、直接通信正则项）；
- 完整每帧目标公式、Batch reduction、λ和R的来源；
- 训练与评估的相同 Hard Forward / STE / receiver cache 协议；
- 所有正确性门禁及通过/失败的真实日志；
- 作者原论文“张量计数 BPS”与本项目“纯载荷 BPS”和“完整序列化 BPS”的区别。

`V4_FINAL.md` 必须包含完整结果表，并明确回答：

1. 是否真正去掉 Budget 输入和 Budget 超额惩罚？
2. 学到了多少 optional UPDATE？每帧真实 payload BPS 是多少？
3. 训练 Soft 成本与实际 Hard 成本相差多大？
4. Task Loss 与 Communication Loss 均有有效梯度吗？
5. 是否出现无意义的 INIT_ONLY 式退化？
6. 与 P16 在相近成本下比较，AMOTA 是否有稳定优势？
7. 哪些结论已有证据，哪些只能作为假设？
8. 当前论文应该采用 V4，还是仍应以已经验证的 V2 MLP 为主？依据必须来自实际实验。

最后在 Codex 回复中只给出精炼状态报告：

```text
V3_ARCHIVE: VERIFIED / BLOCKED / PARTIAL
archive_path:
archived_models:
active_v3_jobs:

V4_IMPLEMENTATION: PASS / FAIL
budget_input_removed:
objective:
lambda_comm:
train_reference_R:
preflight:

V4_TRAINING: NOT_STARTED / RUNNING / COMPLETED / BLOCKED
best_epoch:
checkpoint:

V4_VALIDATION:
seeds_completed:
feature_payload_BPS:
serialized_BPS:
mAP:
AMOTA:
P16_paired_comparison:

FILES_CREATED:
BLOCKERS:
NEXT_MINIMAL_ACTION:
```

## 9. 严格的范围与停止规则

- **不要先做 V4，再回头补 V3 归档。**
- **不要把 V3 的“已训练到第几轮”当成 V4 起始状态。** V4 模型输入/目标改变，应重新初始化 MLP；冻结的原生 CoopTrack 权重和现有数据缓存可复用。
- **不要因为用户想要论文结果就隐瞒失败**：BPS/AMOTA 变差、λ不合适、Shared/Fixed 更强，都要保留。
- 除先前已获用户授权的任务范围外，不下载新模型、不重建官方数据、不改数据切分、不重训原始 CoopTrack、不做多架构/多 λ 自动搜索。
- **如果 V3 正在训练，先停在安全门禁并请求确认。** 如果 V4 的一次候选试验效果不佳，提供证据并停止，不擅自开始第二套实验。

**执行顺序固定：现场核对 → V3 归档校验 → V4 最小实现 → 正确性自检 → 单个 λ 的训练 → full-val 配对评价 → 真实结果报告。**
