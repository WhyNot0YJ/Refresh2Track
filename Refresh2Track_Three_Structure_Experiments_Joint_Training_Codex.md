# Refresh2Track V4：GBA / Fusion Gate / GBA+Gate 三组联合训练实验（Codex 一体化执行提示词）

> **最终决策版。只开展 3 组新训练 **
>
> 面向可访问用户实际 CoopTrack_Audit 仓库、真实缓存与 GPU 的 Codex。开始前先审计现场源码和现有作业；本文中的源码路径、超参数、历史指标属于对照依据，不能代替现场核验。
>
> **用户已明确决定：一个模块增加了新结构，就把这个模块中真正参与计算的原有参数与新参数一起更新。** 不再采用“原 GBA / 原融合 MLP 冻结、只训练新增 Adapter”的方案。**其他 CoopTrack 模块仍冻结**，以免变成重训整个感知系统。

## 0. 实验目标和资源分工

研究问题：在不改变 Refresh2Track V4 的通信协议与发送端输入、不重新训练路端/检测主干的条件下，轻量增强**车路 Query 关联 GBA**、**匹配后 Query 融合 Aggregation**，能否比已有冻结版 V4 获得更好的实际通信—跟踪权衡？

三组从**同一原始预训练 CoopTrack R50 协同 checkpoint**和**从零初始化的 V4 Scheduler**独立训练，绝对不加载 V4 Full epoch12 的已训练 Scheduler 作为初值。

| 实验 ID        | 设备     | 仅新增的结构                         | 原 GBA 有效参数    | 原融合 5 个 MLP    | V4 Scheduler | 其他 CoopTrack |
| -------------- | -------- | ------------------------------------ | ------------------ | ------------------ | ------------ | -------------- |
| **A_GBA**      | 远程 GPU | GBA pair-state residual（约449参数） | **解冻，联合训练** | 冻结               | **从头训练** | 冻结           |
| **B_GATE**     | 远程 GPU | 三分支融合门控（约355参数）          | 冻结               | **解冻，联合训练** | **从头训练** | 冻结           |
| **C_GBA_GATE** | 本地 GPU | 上述两种均启用（约804参数）          | **解冻，联合训练** | **解冻，联合训练** | **从头训练** | 冻结           |

> 远程 A/B 可以在两张 GPU 或已验证足够显存时并发；**如果仅一张卡且显存不够，远程内部顺序排队**，本地 C 与远程独立执行。不要占用/停止用户已有实验进程。
>
> **总共只有 A、B、C 三次正式训练**；不可自行追加任何冻结/解冻对照、H 模型、Periodic 微调、不同 λ 的调参或多 seed 训练。

历史 V4 Full 仅用于性能参照：

```text
V2X-Seq-SPD; car-only; CoopTrack R50; nominal 10 Hz
train = 46 scenes / 7362 cooperative samples
val   = 21 scenes / 3316 cooperative frames
seed=0; Full V4 epoch12 best（训练已在epoch24早停）
AMOTA 0.375331; mAP 0.365131; MOTA 0.301300; IDS 249
Feature Payload 16501.53 B/s; Serialized 27258.58 B/s
Optional UPDATE 2503 / 42772 = 5.852%
V4 Full best.pth SHA256:
0a15216b74b3fc2ebce0570e8aae436c156421b9c7119ec45755527d6cfa4a5f
```

上述历史数值**不是新实验预期目标或保证**。若本机同名文件不匹配，应在报告中说明而不是重写历史结果。

## 1. 先读真实源码，锁定可复现基准

必查：

1. 项目真实根路径、Git HEAD/diff、独立 CoopTrack 源码树、执行容器与依赖、已有进程/显存/磁盘、目前全部实验目录写入者。
2. 原 V4 `scheduler` 输入 2324 维、输出、初始化、loss、AdamW、warmup、batch 和 checkpoint 机制；确认模型参数量历史为 71361；确认 base CoopTrack 协同 checkpoint、路端 checkpoint 和 frozen Query cache 的精确路径与 SHA256。
3. 发送端镜像与接收端 `ReceiverCache`、预测轨迹 ID、scene/session reset、INT8 编解码/STE、52B 的当前几何、780B 的高维 UPDATE、真实 packet 序列化。
4. **本地代码**是否与公开官方 `zhongjiaru/CoopTrack` commit `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399` 相同。用户本地实验报告也出现过本地提交 `212a08ee...`，不能将其当成可在上游仓库查到的同一提交，更不能未经确认用上游覆盖本机修改。

公开参考代码：

- `projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py`：`aggregation()`、`_learn_matching()`、`_query_fusion()`、`init_params_and_layers()`。
- `projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py`：路端 Query 注入、关联监督 `asso_loss`、track task loss。
- `projects/mmdet3d_plugin/cooptrack/modules/latent_transformation.py`：CAA 的空间 keep mask、对齐后 `ref_pts/pred_boxes/latent`。
- 配置：`projects/configs_spd_coop/cooptrack/tiny_track_r50_stream_bs8_48epoch_3cls.py`。

先输出 `PRECHECK.md`：参数白名单、当前任务进程、底座权重、数据/cache 身份、训练配置/已有文档之间的差异。若关键前提缺失，停止并写 BLOCKED，不盲目改代码。

## 2. 三组结构与具体修改位置

### A_GBA：缓存状态感知的 GBA 匹配评分

**插入位置：** `SpatialTemporalReasoner._learn_matching()` 计算出原始关联 **logits `attn` 之后**、`sigmoid(attn)`/Hungarian 之前；不替换原 GBA、CAA 或匹配算法。

```python
# 演示接口，实际依据本地源码整理；只处理合法车路候选pair
orig_logits = original_gba_logits  # [Nv, Ni]
pair_features = receiver_observable_pair_state  # [Nv, Ni, 12]
delta_logits = gba_state_adapter(pair_features).squeeze(-1)  # [Nv, Ni]
logits = orig_logits + delta_logits
```

新增：`Linear(12,32) -> ReLU -> Linear(32,1)`，**449 个可训练参数**。最后线性层权重和 bias 初始化 0，确保 step0 `delta_logits=0`，原匹配行为不变。

**精确定义 12 个输入，Codex 不得擅自使用 GT/未来/未发送最新 latent：**

| 索引  | 输入（原始物理量须 train-only 归一化并记录）                                  |
| ----- | ----------------------------------------------------------------------------- |
| 0     | `log1p(age_in_frames)`：路端此 Query 高维 latent 自最后一次实收 UPDATE 的帧龄 |
| 1     | 自高维 UPDATE 锚点至当前路端的三维位移范数（m，统一物理系）                   |
| 2     | 自高维 UPDATE 锚点至当前路端的 wrap 绝对 yaw 变化（rad）                      |
| 3     | 自高维 UPDATE 锚点至当前路端的速度向量变化范数（m/s）                         |
| 4,5,6 | 当前车端目标与**CAA 后路端目标**的 x/y/z 位置差（m，同一坐标系）              |
| 7     | 上述相对位置三维范数（m）                                                     |
| 8,9   | 当前车路 yaw 差的 sin/cos                                                     |
| 10,11 | 当前车路速度差 `dvx/dvy`（m/s，同坐标系）                                     |

如本机保存的 `pred_boxes` 使用编码后的 yaw 或速度索引，必须先用已有 `denormalize_bbox`/官方 box 工具正确解码；不得直接猜测张量索引。位置要做同坐标系变换；**上次 UPDATE 的 anchor** 不得被每次收到的新几何覆盖。时间单位统一为采样帧而不是墙钟秒。

**A 组准许训练：** 从零初始化的 Scheduler + 新 `gba_state_adapter` + 原 GBA **实际参与 `_learn_matching()` 的完整评分路径**：

```text
STReasoner.get_node_veh.*
STReasoner.get_node_inf.*
STReasoner.get_edge.*
STReasoner.wq
STReasoner.wk
STReasoner.we
STReasoner.ffn.*
```

`pool` 或 `get_dummy_*` 若未被实际 forward 使用，无可训练梯度，不要求为了“整个GBA”硬把死路径列为已更新。除此之外的 5 个融合 MLP/CAA/Tracker/Backbone 均冻结。

**关键：GBA 参数需要真实关联监督。** 公开 CoopTrack 原训练阶段虽计算 learned affinity，但实际用于融合的实例配对会切回几何规则，推理时才用 learned affinity/Hungarian；因此不能宣称最终任务 loss 一定穿过 Hungarian 反传给 GBA。

先检查原 V4 training loss 是否已包含与原协同训练等价的 `asso_loss`。若原生存在，**沿用完全相同正负标签、FocalLoss 和权重**；若原V4没有这一监督，A/C须复用**CoopTrack原生有GT监督的 `asso_loss` 分支**，保持A与C一致，并单独报告其增加，不可对B暗中添加或声称与基础V4是同一优化目标。为防止额外关联监督意外修改发送端原V4优化目标，可在关联监督专用调用中 detach 输入 latent 与 state 特征，使其梯度仅更新原GBA和新GBA评分网络；必须实测证明 Scheduler 的原有 tracking/comm STE 梯度未被切断。若任务代码中的梯度路由有变化，记录清楚，不可隐瞒。

检查：真实非空正负pair、`asso_loss` 梯度、原GBA与新增 adapter 的权重更新；pair-state 的 vehicle 轴必须与该函数训练态 `veh = veh[mask]` 过滤后的索引一致，路端轴与CAA空间过滤后的索引一致。若某 batch 无正例，改用 train 中有监督的真实 batch 做梯度检查，不得用 val GT 训练。

### B_GATE：匹配后路端高维特征的自适应门控融合

**插入位置：** `SpatialTemporalReasoner._query_fusion()`，GBA 已选出 `veh_idx/inf_idx` 后、补全 `_query_complementation()` 之前；原参考点平均、未匹配目标保留流程**不变**。

原始三个 MLP **仍然参与计算且与新增 Gate 一起训练**：

```python
# 对 matched pairs 逐项执行，注意clone避免 in-place 覆盖车端原始张量
orig_fused = original_fuse_mlp(torch.cat([veh_old, inf_recv], dim=-1))
g = 2.0 * torch.sigmoid(gate_logits)  # 3个独立分支，每分支一个标量
out = veh_old + g * (orig_fused - veh_old)
```

语义 `query_feats`、运动 `cache_motion_feats`、位置嵌入 `query_embeds` 分别应用 `g_sem/g_motion/g_embed`。**不是三个不同新MLP，只新增一个小 Gate，原三个融合 MLP 在其前面。**

新增网络：`Linear(7,32) -> ReLU -> Linear(32,3)`，**355 参数**。最后线性层全零初始化：`g=1`，step0 等于原融合输出。

7 维 Gate 输入全部为车端已经能观察到的信息（与 A 共享规范/归一化）：

```text
0 log1p(age_in_frames)
1 since-last-received-anchor delta position norm
2 since-last-received-anchor abs wrapped yaw delta
3 since-last-received-anchor velocity-vector delta norm
4 current matched veh-inf position distance norm
5 current matched veh-inf velocity-vector distance norm
6 current vehicle matched-query confidence
```

**B 组准许训练：** 从零初始化 Scheduler + 新 `fusion_branch_gate` + 原融合 **完整的5个活跃MLP**：

```text
STReasoner.fuse_feats.*
STReasoner.fuse_motion.*
STReasoner.fuse_embed.*
STReasoner.cross_domain_query.*
STReasoner.cross_domain_motion.*
```

原 GBA 整条评分路径和CAA、主干、时序、分类回归保持冻结。B 组使用原生 fused tracking/task loss + `0.05 L_comm`，**不新增关联 loss**（若原V4已包含关联loss且有固定实现，则保持其行为和数值但不更新GBA，并记录实际 loss 项，不能隐瞒）。

训练测试必须证明新Gate和五个融合MLP对有效matched pairs收到有限非零任务梯度、参数真实更新。对于空配对不要强求 Gate 有梯度；选有合法匹配的真实 train batch 做门禁。

### C_GBA_GATE：同时增强 GBA 与 Aggregation

**本地设备独立从原始底座+新Scheduler开始训练**，不能从远程A/B模型权重继续，也不能从 V4 best 调度器继续。

组合执行序列：

```text
Roadside V4 decoded message + Vehicle Query
                     ↓
  CAA（原始冻结坐标与特征对齐）
                     ↓
  原GBA + GBA state adapter（全部相关GBA参数联合训练）
                     ↓
     原匹配算法（不改变未匹配处理）
                     ↓
  原三个融合MLP + branch gate（联合训练）
                     ↓
    未匹配实例补全 + cross_domain 2个MLP（训练）
                     ↓
    原生轨迹细化、ID维护及task loss（冻结原参数）
```

**C 组准许训练：** 从零初始化 Scheduler + 新 GBA adapter + 原 GBA 活跃评分网络 + 新 Gate + 原融合5个MLP；其余冻结。

GBA 监督与 A 完全相同；Gate 的 task-loss 梯度与 B 完全相同。两边的接收端缓存元信息映射要共享**相同**定义，不能远程使用一个字段编码、本地用另一个编码。

### 原有GBA/融合模块之外，全部冻结

- 冻结路端全部参数与离线 Query 缓存；不重新训练任何路端模型。
- 冻结车端图像 backbone/FPN、BEV 编码、decoder、初始 query embedding、MotionExtractor、4-slot temporal/spatial Transformer、分类与位置回归头、CAA，以及所有未在上表白名单的 CoopTrack 参数。
- 全部非训练子模块 `eval()`，特别保持原始 BN 参数及 running mean/var 不变。原 GBA/融合被列为 trainable 时正常参与 forward + autograd，不因为整个主干需要冻结而使用包住整个 forward 的 `torch.no_grad()`。
- 轨迹状态、历史 Query 和 ReceiverCache 会在时间线上**正常更新**；这与“网络权重冻结”不同。

对应官方256维网络的**预计**可训练参数量（仅供检查）：

```text
A_GBA      = Scheduler 71361 + GBA原活跃路径 724225 + 新结构 449  = 796035
B_GATE     = Scheduler 71361 + 融合原5MLP 985600 + 新结构 355  = 1057316
C_GBA_GATE = Scheduler 71361 + GBA 724225 + 融合 985600 + 新结构 804 = 1781990
```

实际参数白名单、冻结参数/BN buffer哈希、梯度与 step 前后权重 diff 必须用本地 `named_parameters()` 生成，**禁止为了凑这个参考数字虚报**。

## 3. 通信协议绝对保持原V4

**所有三组都从零训练各自 Scheduler**（独立实验但采用相同 seed0 初始化规则、同样结构、参数数量、训练数据顺序），通信量**可能彼此不同**，不能直接假设同BPS。

必保持：

- 2324维 Scheduler 输入，包括 current/last_received/delta 与其他状态；无预算输入。
- 基于真实接收缓存的 Hard UPDATE/REUSE，阈值 `sigmoid(z)>=0.5`，训练前向硬、反向 STE；首次无缓存强制 UPDATE；不设 TTL、forced periodic 或 TopK 填额。
- 真实 ReceiverCache + 发送端镜像、预测 track ID 键、scene/session reset；不能因为排序/遮挡串号。
- 每个 Query 每帧继续发送当前 `ref_pts`/`pred_boxes` 低维字段（52B）；UPDATE 时发送三组256维 latent INT8+3 scales（780B），REUSE高维字段为空且接收端使用最后实际解码的路端高维字段。
- 禁止在 A/B/C gate 或 GBA 中读取路端“本帧未发送的最新高维特征”、未来图像、验证 GT，或把发送端 raw delta 当作接收端已有信息。
- 新增 age/上次高维 UPDATE 锚点只在接收端本地 sidecar metadata 计算；current geometry 按原协议发送；不增加线上的隐藏字段。第一次UPDATE后锚点记录对应时间/几何，REUSE仅更新当前几何而**不得覆盖锚点**。
- CAA对齐/空间过滤后，metadata必须对原路端 Query 用**同一个keep mask**；跨端匹配的 `inf_idx` 必须取对应路端metadata。`_init_inf_tracks()` 会把一些内部 `obj_idxes` 初始化为 -1，不能拿它当发送端真实原始预测ID；sidecar维护原始(scene,session,id)映射。避免类型/容器堆栈中偷偷放不可序列化字段。

通信计算：

```text
FeaturePayload_bytes_per_frame = 52*N + 780*U
U = forced_updates + optional_hard_updates
FeaturePayload_BPS = sum(FeaturePayload_bytes_per_frame)/num_frames * 10
Serialized_BPS = mean(actual_packet_length_bytes) * 10
```

其中 `N` 是实际消息 Query 数，`U` 是发送端实际UPDATE数；完整序列化长度包含ID/flag/header。必须保存逐帧(scene,session,frame,N,forced,optional,hard,feature_payload,packet_bytes)，累计真实计费。真实packet编码的UPDATE增量须恰好780字节（对同一字段布局）。

**两台设备同步性：** 不假设同seed在12GB RTX3080Ti与16GB RTX4060Ti上能做到bitwise完全一致；保证共用输入数据、代码协议、数值模式、R50、 checkpoint来源、数据顺序与seed，并记录实际差异。

## 4. 损失函数与训练超参数

沿用 V4 已通过验证的原生任务路径与通信目标，额外关联监督只按第2节严格处理：

\[
L_{\text{base}} = L_{\text{native tracking}}+0.05 L_{\text{comm}}
\]

\[
L_{\text{comm},t}= \frac{780}{R}\sum_{i\in O_t}p_{t,i},\qquad
R=12840.189523169913\,\mathrm{B/frame}.
\]

`O_t` 只含可选 Query，forced init 不产生此项梯度；**不要额外按可选Query数平均**；帧级项对有效帧均值。原生任务loss和关联loss边界先按现场实际实现复核，A/C需要可反传的监督以训练GBA；必须记载确切 loss 数值项、加权前后占比及其梯度作用范围。

| 参数                                 | 固定协议                                                                                      |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| Split                                | official train 46 scenes/7362；val 21 scenes/3316                                             |
| 频率、类别、主干                     | nominal10Hz、car-only、R50                                                                    |
| Seed                                 | 0；记录python/numpy/torch/CUDA RNG，并保证同样Scheduler初始张量哈希                           |
| Batch size                           | **4**，不擅自改变                                                                             |
| 精度                                 | **FP32**，不为了省显存改AMP                                                                   |
| Optimizer                            | **AdamW**                                                                                     |
| Peak LR（Scheduler和新增结构）       | **1e-4**，与原V4相同                                                                          |
| Peak LR（解冻的原GBA / 原融合5 MLP） | **1e-5**，因这些层已有预训练权重，采用较小学习率联合更新，三组遵守同一分组规则                |
| Weight decay                         | `1e-4`                                                                                        |
| Grad clip                            | `5`（按原 V4）                                                                                |
| Warmup                               | 第1 epoch各参数组线性升到各自峰值：Scheduler/新增结构 `1e-5 → 1e-4`；原已训练层 `1e-6 → 1e-5` |
| Decay                                | 后续按36 epoch cosine；Scheduler/新结构到 `1e-6`，已训练层到 `1e-7`（相同比例）               |
| Max epochs                           | **36**                                                                                        |
| Full validation                      | 每 **3** epoch，官方完整 3316 帧                                                              |
| Checkpoint select                    | 按 val **原始AMOTA** 严格提高才更新best；同分保留更早                                         |
| Early stop                           | 至少12epoch，连续4次完整验证没有更高AMOTA则停                                                 |

**原已有训练协议就是36epoch上限而非必须训练完36epoch。** 若解冻参数后训练出现数值不稳定、性能下降，先记录/排障而不是暗中把C/某一组单独换小LR得到更好结果；有必要改变LR时先向用户报告并将其明确标注为另一协议版本，不能冒充上述预注册三组公平对照。

**注意：不同参数组学习率不等于冻结某些参数。** 原 GBA/融合5个MLP在对应实验中与新结构一样每步更新，只是由于它们已有成熟预训练权重，预先固定使用较小峰值学习率 `1e-5`，避免用Scheduler的 `1e-4` 直接扰动整套预训练表示。三组一律遵守这一规则。先做真实数据的1-step/多step梯度与参数漂移预检；若发现梯度爆炸/明显破坏预训练表示，暂停后报告，而非继续耗费36ep算力。不能未经许可用验证集反复找有利LR。

## 5. 执行顺序（适合Codex自动操作）

### 阶段0 — 双设备预检与安全隔离

- 检查所有活跃任务PID/GPU占用，不能停止其他用户任务。
- 远程 A/B使用**独立源码工作树、配置目录、输出目录**；同时运行需要不同GPU或已验证足够显存。相同目录内不得同时修改 `spatial_temporal_reason.py`，可通过互不干扰的工作树/可配置Adapter实现。
- 本地C独立工作树，复制准确相同的新增模块源码和特征标准化常数，不依赖远程A/B训练结果。
- 记录原始训练路端cache完整性、train/val split不交叉、CoopTrack主checkpoint及基线V4 checkpoint身份。

### 阶段1 — 代码实现与单元测试

1. 三实验使用同一统一模块实现，通过配置开关 `enable_gba_state_residual`/`enable_fusion_state_gate` 控制，**A只启GBA、B只启Gate、C两者全开**；另配置 `trainable_modules` 仅包含预设各模块原有/新参数与Scheduler。
2. 定义接收端metadata构建器，状态规范、物理坐标归一化和sidecar测试跨设备一致；训练期间不能把新元信息当作真实传输字段。
3. 固定相同 test input、旧 V4 scheduler checkpoint **仅用作step0等价性测试，不做正式训练初始化**，确认 adapter residual=0/gate=1 使三种模型与原V4相同输入的得分、匹配索引、跟踪张量、通信字节一致（浮点设明确容差）；确认不会错误改变未匹配路端 Query 处理。
4. 边界测试：空接收/空Query/空pair、初始化UPDATE、REUSE多帧、ID重建、CAA空间过滤、当前位姿变换、设备切换、跟踪状态与receiver缓存不混淆。
5. **梯度预检**：对真实有optional Query与GT正负配对的连续小片段，验证Scheduler的`task/comm`梯度；A/C原GBA+Adapter受asso监督且真实更新；B/C原融合5个MLP+Gate受跟踪损失且真实更新。所有未列参数和BN buffers不变；不能把冻结模型整个forward置于`torch.no_grad()`。
6. 预检失败停在`PREFLIGHT_FAIL`，输出问题和建议，不得虚假启动训练。

### 阶段2 — 3组正式训练

每组从独立新Scheduler和相同预训练底座开始：

```text
remote  A_GBA      experiments/structure_v4_joint/A_GBA/
remote  B_GATE     experiments/structure_v4_joint/B_GATE/
local   C_GBA_GATE experiments/structure_v4_joint/C_GBA_GATE/
```

同样seed、batch、数据顺序、真实时序接收缓存、完整损失、训练日程与早停。每3epoch全val并保存best。断点续训必须恢复**optimizer、LR scheduler、RNG、真实train-cache时序状态**；不能从另一个实验checkpoint继续，不能将模型换初值后当resume。

如果本地16GB因为C同时解冻原GBA和5个融合MLP而OOM，**不要偷改batch/FP32以求完成**。记录显存与阻塞信息；可等远程资源释放后把C迁移过去，从checkpoint严格resume/重跑，并标明硬件/复现影响。当前三组仍是三组，不额外训练一条结构变体。

### 阶段3 — 完整评价（必须覆盖真实3316帧）

- 对每组按其val最佳 checkpoint，从新进程重新加载原始CoopTrack底座＋该组可更新原模块状态字典＋新增网络＋Scheduler；不能只存小网络却漏掉更新后的原GBA/融合权重。
- 每个scene/session起始重置真实ReceiverCache/SenderMirror/track query；记录正式评估顺序与随机补槽行为。
- 输出AMOTA、mAP、MOTA、IDS、FP、FN；实际FeaturePayload_BPS、Serialized_BPS、optional update rate、每帧bytes及P95；GBA正确配对分析（仅有合法GT时评估）、gate分支平均值/分位数及age分桶；推理FPS与峰值显存。
- 所有指标用当前可复现 evaluator、car-only及相同score/阈值协议；禁止在val上挑新匹配阈值来提分。
- 如果新方法AMOTA升但通信量也涨，只报告“新的工作点”，不声称全面优于V4。

### 阶段4 — 仅汇总已有4个模型，形成论文式对照

| Method     | GBA adapter | Gate | 原GBA训练 | 原融合训练 | AMOTA      | mAP        | IDS        | FeaturePayload B/s | Serialized B/s |
| ---------- | ----------- | ---- | --------- | ---------- | ---------- | ---------- | ---------- | ------------------ | -------------- |
| 旧V4 Full  | ×           | ×    | ×         | ×          | 0.375331   | 0.365131   | 249        | 16501.53           | 27258.58       |
| A_GBA      | ✓           | ×    | ✓         | ×          | `实际结果` | `实际结果` | `实际结果` | `实际结果`         | `实际结果`     |
| B_GATE     | ×           | ✓    | ×         | ✓          | `实际结果` | `实际结果` | `实际结果` | `实际结果`         | `实际结果`     |
| C_GBA_GATE | ✓           | ✓    | ✓         | ✓          | `实际结果` | `实际结果` | `实际结果` | `实际结果`         | `实际结果`     |

**方法学边界：** 因为本轮明确让“原GBA/原融合 MLP”与新增模块联合更新，所以旧V4与A/B/C的差异包含**新增结构＋原模块微调**；这四行支持比较三种“结构＋训练包”，不能单凭表格把全部增益严格归因于新增449/355个参数。用户只批准3组新训练，不为了隔离微调效应自动追加第4组。论文写法应准确。

不得仅按最高AMOTA挑胜者；按AMOTA/BPS、增幅/相对通信、运行复杂度联合判断。A/B/C相互比较时也因不同可训练参数集合而不能简单相加得出贡献效应。

## 6. 必须留存的文件与结束报告

每组新目录至少包含：

```text
configs/resolved_config.json
provenance/model_manifest.json
provenance/git_diff.patch
preflight/freeze_trainable_params.csv
preflight/grad_and_weight_delta.json
checkpoints/best.pth
checkpoints/latest.pth
logs/train.log
metrics/epoch_validation.csv
metrics/best_full_val.json
communication/per_frame.csv
metrics/association_or_gate_diagnostics.json
reports/FINAL.md
```

跨设备共享一份最终 `experiments/structure_v4_joint/COMPARE_THREE.md` 和 `COMPARE_THREE.csv`；统计包括每组 best epoch、实际运行进度、显存/速度和错误诊断。**未运行/失败填NA**，不可生成虚假结果或替换历史V4数据。

最终Codex回复必须简短列出：

```text
PRECHECK: PASS/FAIL
MODEL_BASE: actual checkpoint SHA256
SEED/SPLIT/PROTOCOL: resolved
A_GBA: RUNNING/COMPLETED/BLOCKED; best epoch, AMOTA, two BPS
B_GATE: RUNNING/COMPLETED/BLOCKED; best epoch, AMOTA, two BPS
C_GBA_GATE: RUNNING/COMPLETED/BLOCKED; best epoch, AMOTA, two BPS
TRAINABLE_A/B/C: exact module names, grads, counts
NO_RECEIVER_LEAKAGE: PASS/FAIL
FROZEN_WEIGHTS_AND_BN: PASS/FAIL
CODE_AND_REPORT_PATHS:
BLOCKERS_AND_NEXT_STEP:
```

## 7. 严格执行边界

- **只运行三组**；不做此前旧提示词中的A_F/U、B_F/U、C_F/U六配置，也不运行前一版`only-new-adapter+frozen-original`路线。
- **复用原底座，不继承V4 Full调度器best**；Scheduler必须重新训练。
- **同一个被修改模块的原有有效参数＋新增子网络一起更新**：A更新整条GBA评分；B更新融合5个MLP；C两者均更新。
- 不重训路端，不改CAA/主干/检测头/4帧历史建模，不加新的通信负载/预算机制，不修改场景标签或val GT。
- 不能把预期优势说成实验事实；如果不能确认真实训练路径，就停下并报告。
- **不触碰/删除用户既有 V4、三项消融、远端其他任务产物**；新实验目录独立，保留可复现性。

**执行顺序：真实源码/权重核对 → 双设备环境检查 → 新模块和sidecar metadata实现 → step0等价与梯度门禁 → 三次正式训练 → 全val复评 → 汇总。**
