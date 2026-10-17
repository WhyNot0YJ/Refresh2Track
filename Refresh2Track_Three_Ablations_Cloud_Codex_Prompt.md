# Refresh2Track：三个正式消融实验——云服务器 Codex 执行提示词

> 将本文件放到项目工作目录后，整体交给 Codex 执行。**本次三个实验均为必做**：`w/o Explicit Delta`、`w/o Receiver-State Conditioning`、`w/o Communication Loss`。不要把第三个标记为可选。核心目标是完成三次可复现的独立训练与相同协议下的完整评估，而不是继续改造主方法。

## 0. 你的角色与首要约束

你是一名负责 CoopTrack / Refresh2Track 研究工程的执行 Agent。请先**阅读实际仓库**，再行动。项目的实际目录、Docker 容器、已有 V4 源码、Checkpoint、配置和产物路径以机器上检查到的为准；以下目录仅是命名建议，**不可假装路径已经存在**。

请遵守：

1. **不修改、覆盖、移动、删除原有 V2、V3、V4 训练产物、Checkpoint 和结果。** 当前 V4 Full 为正式主模型；不要重训它，也不要恢复 V3 Fixed/Shared。
2. 不改变冻结的 CoopTrack R50 主干、权重、原生时序推理、数据 split、INT8 量化、三组 latent、逐帧低维几何、接收端 cache、scene/session reset、predicted track ID 匹配、Hard UPDATE/REUSE 决策、STE、threshold=0.5、原生任务 Loss 和评估器。
3. 新增实验一律创建独立目录、独立配置、独立 Checkpoint、独立日志；新代码尽量采用显式配置分支，不要在旧脚本直接硬编码覆盖。
4. **不擅自调 λ、阈值、归一化常数、Batch Size、学习率、早停规则或模型结构**以追求更好的结果。必须与已经完成的 V4 Full 正式运行保持一致；任何无法核实的参数停止并报告，不猜测。
5. 默认三个实验均从头初始化一个独立 MLP，而不是在已训练 V4 权重上临时 mask 后直接评估。每个模型使用与 V4 Full 相同的训练 Seed 和初始化策略。
6. 不允许使用 official validation 来不断调 λ 或重新定义消融，不允许挑选最有利的一次结果。继承 V4 原有 checkpoint selection 协议，并明确记录 official val 是否参与了选模。
7. 先进行只读审计、GPU 资源检查和 CPU/微型功能测试；通过后才开始正式训练。云端若缺少必要的数据或 Checkpoint，明确停止，**不要自动从未知地址下载或重新生成完整数据**。

## 1. 三个正式实验（全部执行）

### A. `w/o Explicit Delta`

**目的**：检验显式 `current - last_received` 特征是否帮助 MLP 优化；这不会移除接收端信息本身，因为 current 和 last 仍在。

完整 V4 输入合同如果仍是 `N×2324`，屏蔽：

```python
features = features.clone()
features[:, 1536:2304] = 0.0    # 768-D explicit delta
```

其余输入、缓存、训练和通信机制保持原样，**从头训练**。

### B. `w/o Receiver-State Conditioning`

**目的**：检验显式感知“接收端实际缓存内容与新鲜度”是否有利于 UPDATE/REUSE 决策。

```python
features = features.clone()
features[:, 768:1536] = 0.0     # 768-D last actual received latent
features[:, 1536:2304] = 0.0   # 768-D current - last received
features[:, 2304] = 0.0        # 1-D elapsed since last actual UPDATE
```

原始有效信息为 `2324 - 1537 = 787` 维，但**MLP 输入张量依旧是 N×2324**；请保持其他层和参数量不变，以隔离输入信息差异。

**极重要**：这里仅遮蔽“调度器输入”，**禁止关闭真正的 ReceiverCache 或 Sender Mirror**。车端依旧必须根据真实 UPDATE/REUSE 决策使用最后收到的 INT8 反量化 latent；无缓存的新 Query 依旧强制 UPDATE。`confidence change`、`ref_pts delta` 是相对上一次 *observed* 的变化（非上次 *received*），目前保留；`track_age` 是轨迹生命周期，也保留。不要错误屏蔽当前的 `cache_motion_feats`——它是路端当帧生成的运动 latent。

### C. `w/o Communication Loss`

**目的**：检验联合优化中通信损失项的作用。

完整 V4：

```text
L_total = L_task + lambda_comm * L_comm
```

本消融：

```text
L_total = L_task
```

实现时应保留原损失计算、统计与日志接口，仅**在计算总损失时使通信惩罚权重严格为0**。不能移除 MLP、缓存相关输入、Hard Gate、STE，不能绕过真实报文解码直接把路端当前 latent 注入 CoopTrack。对该分支也重新初始化并正式训练。

**注意**：λ=0 不保证模型全部 UPDATE；若表现为几乎全部 REUSE、训练任务梯度很弱、或者在高通信下精度更好，全部如实记录，不要自动补上新的正则或重设阈值。

## 2. 首先核对 V4 Full 的真实输入合同

不要把历史 V1/V2 文档当作 V4 的真源码。逐项查找当前 V4 实际执行入口、数据拼接、调度器 `forward`、训练损失、eval 入口，并输出 `reports/ablations/v4_preflight.md`，至少包含：

- Git commit / git status；当前运行容器与 Python/PyTorch/CUDA 环境；GPU型号/显存。
- V4 Full 的最佳 Checkpoint 路径、SHA256、真实 epoch、λ、训练 Seed、数据 split、batch size、优化器、LR/scheduler、精度模式、checkpoint selection/early stopping。
- `feature` 的实际 shape、索引表、dtype、缺缓存时占位值与编码；检查 `N×2324` 是否仍然成立。
- 训练/推理的真实 Hard Gate 和 STE 梯度位置；Task Loss 和 Communication Loss 的实现与梯度路径。
- 序列化结构、每个可选 UPDATE 字节数、`feature_payload_BPS` 与 `serialized_message_BPS` 的计算实现。
- official train **46 scenes / 7362 cooperative samples**、official val **21 scenes / 3316 cooperative samples** 是否可访问；数据路径与原始文件是否完整。

如果 V4 的输入索引不同，**根据真实源码建立同义字段映射并调整消融 mask**，保留证据，不可照搬旧索引。若发现某个所谓 `delta` 不是接收端缓存差，先报告而不要运行错误消融。

## 3. 必须严格一致的训练配置

**继承主模型 V4 Full 的真实配置，以下仅是历史参考，不能覆盖 V4 实际文件。**

| 项目 | 处理 |
|---|---|
| CoopTrack R50 | 冻结、使用 V4 Full 同一权重和同一数据流水线 |
| Scheduler | 与 V4 Full 完全相同的 MLP 结构、输入形状和参数量 |
| Batch size | 与 V4 Full 的实际配置一致，历史参考为 4 |
| 训练 Seed | 与 V4 Full 一致；不要更换以求更好分数 |
| AdamW / LR / Weight decay / clip | 完全读取 V4 正式配置，历史参考 `1e-4 / 1e-4 / 5` |
| LR warmup / cosine | 原 V4 完整计划，不另建策略 |
| λ | A/B 保持 V4 原值；C 精确为 0 |
| Epoch / eval frequency / early-stop | 与 V4 完全一致；不可因为某分支早期表现差自行停止 |
| 官方验证集 | 与 V4 一致的完整21 scenes、3316帧、car-only评价 |
| 推理 Seed | 采用与 V4 / 正式基线相同的固定 Seed 0 和随机数初始化位置 |

云端显存不够时：首先确认是否可复用原 V4 的 gradient accumulation 或资源策略。**不可静默把 Batch 4 降为 Batch 1/2 然后宣称严格同训练配置。** 如果必须改变，应停止并提出差异及影响，等待批准；不要以硬件不同换取实验协议不同。

三个消融可串行训练，推荐**先 B、再 A、最后 C**，因为 B 最直接检验论文核心主张。支持单任务中断恢复，但恢复必须校验配置、模型 SHA 和 optimizer/scheduler 状态，不能误从 V4 Full Checkpoint 恢复为新的实验初始点。

## 4. 正式运行前正确性门槛

写小型独立测试并留下 PASS/FAIL 结果：

1. 三个输入模式在训练与推理都实际生效；shape、dtype不变，非屏蔽列逐元素相等。
2. A 仅屏蔽 delta；B 完整屏蔽 last/delta/elapsed；C 所有输入列和 V4 Full 相同。
3. A/B/C 均通过*同一个*接收端真实缓存重建和消息序列化管线；REUSE 时没有当前 latent 泄漏到 CoopTrack 下游。
4. 无缓存的新轨迹强制 UPDATE；scene/session reset 正确，预测 ID 不能用数组行号替代。
5. A/B 的 `lambda_comm` 与 V4 完全一致；C 的总损失确实没有通信项，同时保留任务损失的梯度和通信诊断统计。
6. MLP 在训练时的硬前向、STE反向与正式推理一致；不发生 NaN/Inf；非强制 Query 下至少验证一次任务梯度能到调度器。
7. 可验证的通信基础公式保持一致：`serialized_bytes = 16 + H + 61N + 780U`（只有在实测 V4 仍采用同样协议时才能使用；否则以 V4 的实际代码为准）。
8. 不要求三个分支全部优于 Periodic 或 Full V4。检查条件只关乎正确性，不关乎性能是否好看。

任一正确性门槛 FAIL 则不要开展正式训练。微型测试仅用于功能正确性，**不可把小样本指标当论文结果**。

## 5. 正式训练与评估

每一分支独立：

```text
read V4 full canonical configuration and frozen source identity
  → create fresh ablation config / output folder
  → seed and build fresh scheduler
  → load frozen CoopTrack, original sender & receiver pipelines
  → full official train: 46 scenes / 7362 cooperative samples
  → checkpoints / scalar losses / soft and hard communication / optional and forced counts
  → validation per V4's predeclared cadence (full official val if that is V4's protocol)
  → select model strictly by existing V4 selection rule
  → freeze selected checkpoint + hash
  → report full 21-scene/3316-frame evaluation and real payload
```

记录每个 epoch：

- `task_loss`, `comm_loss_raw`, `lambda_comm`, `total_loss`, LR, finite gradient/gradient norm。
- `hard_optional_update_count` / `optional_query_count`；forced UPDATE 数、REUSE 数；Soft 预计与 Hard 实际的 byte breakdown。
- training throughput（samples/s、epoch seconds）、GPU peak memory；Checkpoint epoch和SHA。
- 完整验证中的 AMOTA、mAP、MOTA、IDS、FP/FN（若 evaluator 支持并且取同一固定操作点），以及 `feature_payload_BPS`、`serialized_message_BPS`。

**通信主指标**（10Hz名义帧率，且三个消融都使用同一数据协议）：

```math
FeaturePayloadBPS = (10 / T) × Σ_t [52 N_t + 780 U_t]
```

`N_t` 为本帧实际发送的目标数，`U_t` 包括强制和可选的真正 UPDATE 数。52B 对应 `ref_pts` 和 `pred_boxes` FP32；780B 对应三组256维INT8高维特征及各自的FP32 scale。**不含 JSON Header / 16B prefix / ID / UPDATE flag / TCP/IP。** 同时保留 `serialized_message_BPS = 10 × mean(len(packet))` 作辅助。若 V4 Full 实际口径不同，先对齐而不是强行套公式。

不能把仅仅改变 `lambda_comm` 的训练阶段预计 soft bytes 冒充真实发送 bytes。

## 6. 消融主表、合理解释与公平性

至少输出如下表（所有行统一载荷 BPS 与 Seed 0 口径，V4 Full 使用已保存正式结果而非重训）：

| Method | Input dimensions (effective) | Feature BPS ↓ | Serialized BPS ↓ | Optional UPDATE % | mAP ↑ | AMOTA ↑ | MOTA ↑ | IDS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 Full (reference) | 2324 | 读取原档案 | 读取原档案 | 读取原档案 | 读取原档案 | 读取原档案 | 读取原档案 | 读取原档案 |
| w/o Explicit Delta | 1556 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| w/o Receiver-State Conditioning | 787 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| w/o Communication Loss | 2324 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

只允许下列保守解释：

- `w/o Explicit Delta`：`current` 和 `last` 本身已蕴含差值；其价值是显式特征工程/优化便利性，不能声称带来数学上的新信息。
- `w/o Receiver-State`：观察接收端状态信息对实际通信—精度权衡的影响，**在它通信成本也变化时不可只看AMOTA高低**。
- `w/o Communication Loss`：用于论证通信正则是否改善成本—任务权衡；可能升通信、可能下降性能，结果如实记录。
- 若所有消融比完整模型还好，则明确标记核心设计未获证实，不得改写结果、只挑有利 Seed，或自动开展架构调参。
- 当前 V4 Full 的历史 Epoch12参考：AMOTA约0.37533、mAP约0.36513、feature payload约16502 B/s；**必须读取最新正式冻结 Checkpoint 与产物，不得硬编码这些早期数值当最终基线**。

## 7. 服务器迁移与归档

首先只读盘点本地与云端目录。数据量较大时只传必要运行部分，优先复用云端已经存在且校验通过的数据/模型；不重传重复数据。通过 SHA256 校验 V4 Full Frozen Checkpoint、底座权重、源码和必要manifest一致性。

建议新增目录（可按项目既有结构调整）：

```text
experiments/v4_ablations/
  no_explicit_delta/
    config.json
    train.log
    checkpoints/
    val/
    summary.json
  no_receiver_state/
    config.json
    train.log
    checkpoints/
    val/
    summary.json
  no_communication_loss/
    config.json
    train.log
    checkpoints/
    val/
    summary.json
reports/ablations/
  v4_preflight.md
  input_mask_tests.json
  run_manifest.json
  three_ablation_results.csv
  three_ablation_summary.md
```

对比报告必须记录训练与评估是否共用官方val选模；如果使用了，就如实说明 selection bias。所有正式过程固定 Seed 0，不额外启动三 Seed；也不能声称跨Seed稳健性或统计显著性。

**停止条件**：任何数据/模型身份不符、缓存泄漏、输入列mask错误、任务梯度错误、官方帧覆盖不完整时停下并报告。GPU OOM须先报告真实峰值及最小可行措施；不得自行把训练规模缩小为mini split而继续宣称正式实验。

## 8. 最终交付

完成后只需给出以下摘要，并提供精确文件路径与SHA：

```text
V4_FULL_PRESERVED: PASS/FAIL
CONFIG_PARITY: PASS/FAIL
OFFICIAL_TRAIN: 46 scenes / 7362 frames CHECKED/NOT_CHECKED
OFFICIAL_VAL: 21 scenes / 3316 frames CHECKED/NOT_CHECKED

NO_EXPLICIT_DELTA:
  training: COMPLETE/FAILED
  best_epoch:
  checkpoint_sha256:
  Feature_BPS:
  AMOTA:
  mAP:

NO_RECEIVER_STATE:
  training: COMPLETE/FAILED
  best_epoch:
  checkpoint_sha256:
  Feature_BPS:
  AMOTA:
  mAP:

NO_COMMUNICATION_LOSS:
  training: COMPLETE/FAILED
  best_epoch:
  checkpoint_sha256:
  Feature_BPS:
  AMOTA:
  mAP:

ABLATION_TABLE: <actual file>
EVIDENCE_REPORT: <actual file>
ISSUES_AND_LIMITATIONS: <if any>
```

执行期间可以给出简短进度，但不要在还没有完成 full-val 时给出虚假的最终AMOTA。除三个已指定消融外，不再擅自安排 Transformer、GRU、多 λ、2Hz、UniV2X、V3复训或主模型重训。
