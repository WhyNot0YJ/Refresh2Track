# CoopTrack V2：Periodic 极限扫描与自适应时序刷新完整执行提示词

> 执行对象：在本地项目中工作的 Codex  
> 项目根目录：`D:\CoopTrack_Audit`  
> 整合日期：2026-10-07  
> 任务范围：保留 V1，完成 Periodic 极限扫描、V2-MLP／V2-Transformer 实现与训练、内部 dev 选模、全 train 重训及 official val 最终评估。  
> 本文是执行指令，不是 V2 已完成的证明。历史事实须与本地记录核对，未运行的结果不得填写为 PASS。

---

## 0. 任务目标与执行方式

不要重新介绍 CoopTrack，不要重新选择 backbone，不要把任务退回数据准备或可行性研究。本次连续完成两件事：

1. **扫描 Periodic-N 的真正极限。** 找出固定刷新从基本保持性能到持续掉点的区间，并测量实际通信下限。
2. **实现真正可自主决定 UPDATE／REUSE 的 V2。** 移除 V1 的 `max_staleness=4` 强制刷新和 Budget Top-K 填槽，训练 MLP 与轻量 Transformer，检验学习式刷新是否在相近实际通信量下优于 Periodic。

同时落实：真正的多 scene batch、scene 内固定预算、最多 32 个 epoch、内部 dev 定期验证、通信约束选模与早停。

先阅读真实入口、锁定配置和已有报告，再修改最少必要代码。能由源码和本地文件解决的问题直接解决，不反复确认本文已经明确的配置。不得为得到“成功结果”擅自修改协议。

执行到全部结果完成，或遇到明确的资源／数据／实现阻断。被阻断时保留已完成结果，报告具体原因和最小下一步；不得假称任务仍在后台运行，不得把计划写成完成。

## 1. 必须继承的已有事实

### 1.1 代码身份与当前进度

历史记录中的 backbone 身份：

```text
project_root = D:\CoopTrack_Audit
backbone_repo = code/CoopTrack
backbone_commit = 29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399

Stage1 = STAGE1_SMOKE_PASS
Stage2 = FULL_DATA_READY
Stage3 = STAGE3_DIAGNOSTICS_PASS
V1 = 主实验完成，随后完成 gate / budget 源码审计
```

历史 V1 正式入口：

```text
scripts/run_main.py
scripts/main_train.py
scripts/main_eval.py
scripts/main_runtime.py
scripts/main_scheduler.py
scripts/main_message.py
```

已审计的关键类／函数：

```text
main_scheduler.Scheduler
main_message.select_updates
main_message.TemporalCache.prepare / commit
main_message.pack
main_message.ReceiverCache.unpack
```

这个 commit 是 backbone 的历史身份，不代表外层 `scripts/` 的 V2 版本。启动时分别记录 backbone HEAD、外层 Git 状态（若存在）和实际 V2 源文件身份，不得伪造 V2 commit。

### 1.2 数据与评价口径

| 数据项 | Train | Val |
|---|---:|---:|
| Cooperative samples | 7362 | 3316 |
| Infrastructure cache frames | 7445 | 3316 |
| Scenes | 46 | 21 |

固定：

```text
Dataset = V2X-Seq-SPD
Backbone = ResNet50
Frequency = 10 Hz
Evaluation = car-only
Infrastructure checkpoint = cooptrack_r50_inf.pth
Cooperative checkpoint = cooptrack_r50_coop.pth
```

7362 cooperative samples 和 7445 infrastructure cache frames 是不同计数，不能因此认定数据缺失，也不能截断任意一侧使二者相等。沿用已验证的 token／timestamp 映射。

### 1.3 已有 V1 代表性结果

以下是历史记录，不是本轮重新运行结果。mAP／AMOTA 使用 0～1 小数。

| 方法 | Mean B/frame | mAP | AMOTA |
|---|---:|---:|---:|
| FULL_FP32 | 42636.559 | 0.362345 | 0.356898 |
| FEAT_INT8 | 12161.645 | 0.361107 | 0.353343 |
| Periodic-2 | 7100.774 | 0.362819 | 0.352596 |
| Periodic-4 | 4559.423 | 0.360797 | 0.354025 |
| V1 APTU B1 | 4750.659 | 0.361420 | 0.355454 |
| V1 no-assoc B1 | 4743.367 | 0.362895 | 0.358268 |

继承的判断：INT8 与简单周期刷新已是强基线；不能只比较 FULL_FP32 就宣称学习式方法有效。新增 association-preserving teacher loss 尚未显示稳定收益，本轮不继续加入它。

### 1.4 V1 已知问题

历史审计确认：

- 无 cache 时强制 UPDATE；已有 cache 时 `elapsed >= 4` 也强制 UPDATE。
- Budget 转为可选更新容量，按分数排名尽量填满；没有真正的 `p >= 0.5` 更新决策。
- B1 共 12668 次 UPDATE，其中 11180 次是强制更新，约占 88.254%。旧日志未拆开 new-init 与 stale-force，不得假装知道两者各占多少。
- V1 训练逐帧随机预算、排名带 Logistic 噪声；eval 则固定预算、使用确定性排名。
- V1 双边平方 rate loss 同时惩罚超预算和少花预算。

本次不是给 V1 堆层数，而是修正决策空间和训练协议。

## 2. 严格边界：保护 V1，不返工

禁止重新下载／解压数据、重新运行 converter、重建已有完整 query cache、重做 Stage1/2/3 或大规模 CRC／SHA 审计。

禁止修改 GT／official split、换 R101、加 Griffin、加第二数据集、重新训练两端 CoopTrack、调整输入分辨率、缩小正式数据集或开始论文排版。

只读保留：

```text
experiments/main/**
reports/MAIN_EXPERIMENT_FINAL.md
已有 Stage1/2/3 与 R50 alignment 最终报告
query_cache/train/**
query_cache/val/**
checkpoints/**
正式 data_infos / datasets / data
现有 Docker 环境与 GPU 兼容 patch
```

不要再次清理整个项目，不自动删除用户改动。V1 数据、结果、权重无需复制一遍；通过独立输出目录和不覆盖旧代码保护。

V2 使用：

```text
configs/main_experiment_v2/
scripts/v2_*.py
experiments/v2/checkpoints/
experiments/v2/predictions/
experiments/v2/communication/
experiments/v2/metrics/
experiments/v2/figures/
reports/v2/
logs/v2/
```

“修正 shared cache”指修正 V2 使用的实现，不允许直接改掉 V1 默认行为。优先新增 V2 cache／selector；复用 helper 时使用显式参数并保留 V1 默认值。

启动时检查已有 V2 产物：只有协议、split、源码身份、variant 和阶段一致的完整产物才允许 resume。不一致时保留原结果，在 V2 下使用单个明确的新运行子目录，不混用 checkpoint，不生成大量 attempt/probe 目录。

集中记录到一份 `reports/v2/V2_IMPLEMENTATION.md`：实际入口、源文件身份、关键差异、必要检查。不要扩张成新的审计框架。

## 3. 统一通信协议与缓存语义

### 3.1 每帧发送什么

当前有效 query 的必要低维字段每帧发送：

```text
predicted track ID
UPDATE / REUSE flag
当前 ref_pts
当前 pred_boxes
必要 scene / session / token / timestamp / frame metadata
```

可复用的高维 bundle：

```text
query_feats         [256]
query_embeds        [256]
cache_motion_feats  [256]
```

`UPDATE`：三个 latent 按已有 INT8 格式一起发送。  
`REUSE`：不发送当前三个 latent；车端读取最后真正收到的 bundle，当前低维状态仍发送。

不做 query 删除、三个 latent 的分开门控、几何字段压缩、新 serializer、丢包或无序网络。保持原先可靠、有序、无丢包的评估假设，不额外引入未计费的 ACK 服务。

### 3.2 实际字节数

沿用已有应用层 serializer。历史审计格式：

```text
M = 16 + H + 61*N
c = 780 B / updated query
actual_packet_bytes = M + c*U
```

H 是真实 UTF-8 header 长度，N 是当前 query 数，U 是 UPDATE 数；780B 为 768B INT8 latent 加 12B scale。

最终依据是实际 `len(packet)`；公式仅用于一致性检查，不得用固定平均 header 替代逐帧计费。M 可由相同 metadata、全部 REUSE flags 的 packet 长度获得，但不要在无 cache 时把该测试 packet 当作合法消息交给 receiver。

统计包含本 serializer 的 header／ID／flags／geometry／latent／scale／CRC；不宣称它包含 TCP、IP、无线协议和重传开销。

历史 FEAT_INT8 与本轮 P1 可能存在消息包装差异。须检查 decoded latent／geometry 等价性，但不能为对齐旧 B/frame 修改包格式。V2 主比较以统一 serializer 下的本轮 Periodic 为准。

### 3.3 缓存键与两类历史

```text
cache key = (scene, native reset session, infrastructure predicted track ID)
```

禁止用 query 行号、GT ID、车端未来结果索引 cache。

严格区分：

```text
last_received:
    最后一次实际 UPDATE 并解码后保存的 INT8 反量化 latent
    仅 UPDATE 时改变

last_observed:
    上一次观察该 predicted ID 的 confidence / ref_pts 等
    每次观察都更新，包括 REUSE 帧
```

`confidence change`／`ref_pts delta` 对比 last-observed；`latent delta` 对比 last-received。

sender mirror 与接收端解码缓存一致，不使用未量化 raw latent 冒充已收值。REUSE 不推进 last-update frame index，不更新 latent mirror。

scene／native session reset 时清空相应状态。沿用 native reset 语义，不因 Period 更长就延长 session。缺帧导致 reset 时正常初始化并记录。

不得通过隐式 TTL 淘汰旧 cache，再把刷新伪装成“无缓存初始化”；不得因为 query 暂时未出现就无条件清缓存。ID 无效／冲突沿用已验证的 native 处理规则，检查不存在跨 session 串用。

## 4. 固定 train-core／dev 划分

从 46 个 official train scenes 中固定选 8 个 dev，剩余 38 个 train-core。按 scene 划分，禁止随机拆帧。

生成规则：

1. 从正式 train manifest 统计每个 scene 的 cooperative sample 数。
2. 按 `(sample_count, scene_id)` 排序，用 `numpy.array_split` 分成 4 个连续分组。
3. 使用 `numpy.random.RandomState(20261007)`，按分组顺序在每组无放回抽 2 个 scene。
4. 固化 scene ID、cooperative token 列表、cache 映射、样本数、种子和脚本身份。

输出：`configs/main_experiment_v2/dev_split.json`。

若已存在符合此前约定、带生成算法与种子的锁定 split，优先复用而非为追求上述具体 RNG 写法重新抽样，并记录所用算法。不得看过指标后换 split。

验证：scene 不交叉，cooperative token 不交叉，两部分并集覆盖全部 7362 个 train samples。

内部 dev 是本轮 scheduler 的开发划分。已有 backbone 权重可能使用过这些 official train scenes，不能称为对整个系统完全独立的测试集。

V1 official val 结果已经被看过。V2 从现在起只使用内部 dev 开发，不把 official val 描述成整个研究过程中“从未接触过的测试集”。

## 5. Periodic-N：独立于学习 gate 的唯一规则

对每个 predicted track ID：

```python
if not receiver_has_cache:
    update = True
elif mode == "init_only":
    update = False
else:
    elapsed = current_infra_frame_idx - last_update_infra_frame_idx
    update = elapsed >= period_frames
```

Periodic 不调用 scheduler、不调用预算 Top-K、不使用额外概率阈值。

P4 的正确例子：

```text
k     UPDATE
k+1   REUSE
k+2   REUSE
k+3   REUSE
k+4   UPDATE
```

以 native infrastructure frame index 计算 elapsed，不用 cooperative batch 的局部步数。禁止隐藏的 `elapsed >= 4` 提前刷新。

统一扫描：

```text
P1, P2, P4, P6, P8, P10, P12, P16
```

10Hz 下名义间隔为 0.1、0.2、0.4、0.6、0.8、1.0、1.2、1.6 秒。缺帧／新 track／reset 会改变实际间隔，额外报告 timestamp 间隔，不能把名义周期当成每条轨迹的实际间隔。

全部共用 INT8 quantization、mandatory fields、serializer、receiver cache、native model、reset 语义和 car-only evaluator。

### 5.1 扫描扩展条件

先在完整 8 个 dev scenes 上运行所有基础点。若：

```text
AMOTA_P1 - AMOTA_P16 < 0.005
且 mAP_P1 - mAP_P16 < 0.005
```

则继续：

```text
P24, P32, INIT_ONLY
```

0.005 是 **0.5 个百分点**，不是相对下降 0.5%。不为触发扩展条件修改阈值。

INIT_ONLY 在每个 native session 内为无 cache 的 ID 初始化一次，之后只复用；reset 后重新初始化。它是诊断端点，不是“整个数据集只发送一次”。

### 5.2 固定瓶颈定义

相对 P1，首次出现：

```text
AMOTA 下降 >= 0.010
或 mAP 下降 >= 0.010
```

且下一个更大的、已预设 Period 在同一触发指标上仍越过阈值，标记 `candidate_staleness_bottleneck`。

此规则只是描述性定位，不是统计显著性证明。若最大已扫描点才掉点，没有更大的预设点可确认，写 `UNCONFIRMED_AT_SWEEP_BOUNDARY`，不临时加有利周期。

若未找到瓶颈，写 `NO_BOTTLENECK_IN_SCANNED_RANGE`，照实展示曲线，仍完成已规定的 V2 实验，不制造“困难区”。

### 5.3 扫描输出与通信下限

每个方法报告：

```text
method / period_frames / nominal_period_seconds
sample_count / scene_count
mean_bytes_per_frame / p95_bytes_per_frame / bytes_per_second_at_10Hz
mAP / AMOTA / AMOTP / MOTA / IDS
query_count / update_count / query_weighted_update_ratio
new_init_count / periodic_refresh_count
mean_mandatory_bytes / mean_latent_update_bytes
staleness mean / median / p95 / max
scene_reset_count / native_session_reset_count
```

在同一无 TTL、predicted-ID 序列和 reset 协议下，可直接从 query manifest 统计“仅必要初始化”的通信下限。这不需要训练，也不要求在未触发扩展时运行 INIT_ONLY 的完整模型评估。

区分 `mandatory-only floor` 与 `mandatory + necessary initialization floor`，前者不能当作完整通信量。如果后者已超过某档预算的 1.05 倍，记录结构性不可行，不擅自提高预算。

输出：

```text
experiments/v2/metrics/periodic_dev.csv
reports/v2/PERIODIC_SWEEP_DEV.md
```

给出完整表、Bytes/frame–AMOTA／mAP 曲线及名义 Period–AMOTA 曲线。INIT_ONLY 单独标注，不能赋一个虚假的有限 Period。

## 6. V2 学习式 gate：移除两个限制

V2 学习式方法的 hard decision：

```python
if no_receiver_cache:
    UPDATE
else:
    UPDATE if sigmoid(logit) >= 0.5 else REUSE
```

只有无 cache 初始化是强制更新。scene／session reset 清空 cache 后，首次出现自然落入这个条件。

必须删除 V2 路径中的：

```text
elapsed >= 4 → force UPDATE
elapsed >= N → force UPDATE
按预算计算 slots 后不管分数高低都填满
Logistic / Gumbel ranking noise
按 val 结果调整 threshold
```

保留 staleness 作为输入，不能裁剪成“最多 4”，也不能通过 cache TTL 恢复固定刷新。B4 很宽松但没有 query 值得更新时，允许不花满预算。

本轮是 **确定性阈值 gate + 软通信约束**，不是硬预算调度器。禁止发现超预算后偷偷补一个 Top-K 截断层；那会改变已定义的方法。

## 7. V2 输入与两种模型

### 7.1 复用经过审计的输入接口

当前三个 INT8 反量化 latent 拼成 768 维。每个 query 的输入为：

| 列区间 | 含义 | 维数 |
|---|---|---:|
| 0:768 | current latent | 768 |
| 768:1536 | last-received latent | 768 |
| 1536:2304 | current − last-received | 768 |
| 2304 | elapsed／staleness | 1 |
| 2305 | 当前 confidence | 1 |
| 2306 | confidence change | 1 |
| 2307:2310 | 当前 ref_pts | 3 |
| 2310:2313 | ref_pts delta | 3 |
| 2313:2323 | 当前 pred_boxes | 10 |
| 2323 | track_age | 1 |

合计 2324 维。Budget 是另一路 scalar，不在这 2324 列中。

`track_age` 沿用当前 frame index 减本 session 首次观察该 ID 的 frame index；delta 的时间参照遵守第 3 节。不要根据字段名猜测原生 tracker 语义。

所有模型使用相同数据、量化、特征构造和归一化。保留已验证的归一化；若本地代码没有某种归一化，不为某个模型临时新增。检查没有把长 staleness clip 到 4 或把当前 latent 偷写进 last。

禁止将 GT、未来帧、query 行号、predicted ID 的数值当作 gate 输入。GT 只用于原生训练任务损失和离线评估／分析。

### 7.2 V2-Transformer

复用已验证的轻量结构，不增加层数寻找优势：

```text
shared latent projection: 768 → 64
current / last / delta 分别投影
concat(3 × 64, 20维状态): 212 → 64
LayerNorm + ReLU

budget embedding: 1 → 64 → 64
budget token 与当前 scene 的 query tokens 共同参与 attention

TransformerEncoder:
    layers = 2
    d_model = 64
    heads = 4
    dim_feedforward = 128
    dropout = 0
    activation = ReLU

discard budget output token
output head: 64 → 1
```

没有基于 query 行号的位置编码。使用本地现有 PyTorch 能支持的接口，不为新版便利 API 升级整个环境。

### 7.3 V2-MLP

优先复用现有、确实被 V1 正式调用的 MLP variant。保留相同的 current／last／delta 投影、20 维状态、budget 条件和单 query 输出；不得用没有 Budget 输入的弱 MLP 充当对照。

若没有可复用的 MLP 分支，采用以下固定补全方案并在模型配置中记录：

```text
query projection 与 Transformer 相同，输出 h_i [64]
budget embedding 与 Transformer 相同，输出 e_B [64]
concat(h_i, e_B): [128]
Linear(128, 64) → ReLU
Linear(64, 64) → ReLU
Linear(64, 1)
```

MLP 不执行跨 query 的池化或 attention。两种模型必须共享 gate、损失、训练数据、训练上限、选模规则和评估协议。

报告真实参数量及 Budget 注入方式。Budget token 与逐 query 条件注入是不同的具体实现，不得把它们写成“参数完全匹配、唯一一行差异”的严格消融。首轮不增加第三种网络。

## 8. 真实发包与 straight-through 梯度路径

训练和 eval 的 **gate 前向规则**一致；训练用 ST 近似反传。序列化、反序列化和 cache commit 使用 detached 的真实 hard flags。

下面是要实现的逻辑示例，函数名不是在声称现有代码已经具备这些接口：

```python
p = torch.sigmoid(logits)
forced = valid_mask & no_cache_mask
optional = valid_mask & ~no_cache_mask

hard = forced | (optional & (p >= 0.5))
soft = torch.where(forced, torch.ones_like(p), p)
soft = torch.where(valid_mask, soft, torch.zeros_like(soft))

# 常规 deterministic straight-through gate 表达：
gate_st = hard.to(p.dtype) + (soft - soft.detach())
```

对每个高维字段实际采用与 serializer 对齐的 hard-forward：

```python
# C：当前 INT8 编码再反量化的值；L：发包前 last-received 值。
# C / L 均为冻结、detached 的观测；无缓存位置先用安全占位值。
# 无缓存由 forced 确保一定 UPDATE，不能让占位值流入真实 REUSE。
#
# packet = pack(current_raw_fields, hard.detach(), metadata)
# D = receiver.unpack(packet)  # 真实 UPDATE / REUSE 解码结果

s = soft.unsqueeze(-1) * C + (1.0 - soft.unsqueeze(-1)) * L
x_st = D.detach() + (s - s.detach())
```

必须验证 `x_st` 前向与真实解码 D 完全一致，并且发送端 mirror 与 receiver cache 保存的都是 D 的 detached 值。

**读取旧 L 必须发生在 commit 之前。** 不能先把当前更新写进 cache，再用被覆盖的 cache 构建 surrogate，否则可能把 `C-L` 错误地变成零。

其他要求：

- query_feats／query_embeds／cache_motion_feats 三个字段都走同一 hard-forward 逻辑。
- 低维字段始终使用当前帧值，不做 soft blending。
- no-cache 初始化必须先成功 pack／unpack，再更新 mirror。
- receiver 状态和 native 时序状态跨步按现有训练约定 detach，不无意保留整个 scene 的计算图。
- 本轮没有要求长序列 BPTT；不能声称已通过完整未来回报训练更新决策。
- padded query 不参与 hard gate、通信、loss、cache 或 UPDATE 比例。
- 空 query／全部 forced 的帧仍正常推进时序；没有 scheduler 梯度时可跳过 optimizer step，不得跳过业务帧。

权重冻结不等于 `torch.no_grad()` 包住整个车辆模型：依赖 `x_st` 的原生任务计算必须保留输入梯度。验证 eval 才可使用相应 no-grad 路径。

native association 自身已有训练与推理分支差异；本轮不另改 native matching 协议，也不把“gate 前向一致”夸大为“整套系统训练推理完全一致”。

## 9. 四档 Budget 与单边 rate loss

固定沿用：

```text
B1 = 3474 B/frame
B2 = 6382 B/frame
B3 = 9290 B/frame
B4 = 12198 B/frame

R = 12840.189523169913
budget input = B / R
```

R 来自历史 `budgets.json` 中 **完整 train 的 full_int8_mean_bytes**。启动时读取配置核对，不能替换成 val 的 12161.645、最大预算 B4 或本轮 dev 均值。配置与历史值冲突时先检查来源，不自动覆盖。

定义：

```text
M = 当前帧真实 mandatory packet bytes
c = 780
F = no-cache forced-init 数量
p_i = optional query 的 sigmoid(logit)
R = 固定的完整 train INT8 均值

S_soft = M + c*F + c*sum(p_i over optional queries)
B_feasible = max(B, M + c*F)

L_rate = (ReLU(S_soft - B_feasible) / R)^2
```

这是可微的 **soft cost surrogate**。确定性阈值下，`sum(p_i)` 不是实际 UPDATE 数，也不应被称为实际发包字节数的数学期望。

实际通信：

```text
S_hard = len(packet)
       = M + c*count(hard UPDATE)
```

同步记录：

```text
S_soft
S_hard
S_hard - S_soft
max(0, S_hard - B)
max(0, M + c*F - B)  # 不可避免的初始化成本超额
```

`B_feasible` 仅避免惩罚不可避免的初始化成本，不替代最终对名义 B 的通信约束，不代表可以把预算悄悄抬高。

单边罚的含义：

```text
少花预算：不罚
soft cost 超过可实现预算：罚
硬发包是否合规：另用 S_hard 评估
```

软罚不会保证每帧或数据集平均 hard bytes 必然达标。未达标须如实失败，不能在输出中用 S_soft 替换 S_hard。

## 10. 训练范围与总损失

完全冻结：

```text
cooptrack_r50_inf.pth
cooptrack_r50_coop.pth
原始 backbone / neck / query / temporal / fusion / matching 权重
```

仅训练本轮 scheduler 的 projection、Budget embedding、MLP／Transformer、output head。不加 query adapter 或 fusion adapter，不让某个 variant 多训练一部分主模型。

冻结权重与 BN 等持久统计，沿用已验证的 native `return_loss=True` 训练入口；不得因为冻结参数而切断 scheduler 所需梯度，也不得偷偷更新主模型统计量。

总损失：

```text
L_total = L_native_task + 1.0 * L_rate
lambda_rate = 1.0
lambda_extra_association_preserve = 0.0
```

`L_native_task` 使用当前原生 detection／tracking／GT association loss 及其原权重。

删除额外 teacher-preserving 分支及其多余 teacher forward，但不能删掉原生 GT association loss。本轮不加入新的特征 L2、模仿 Periodic 的标签监督、distillation、RL 或多目标额外损失。

`lambda_rate=1.0` 沿用审计中的初始权重并在 V2 固化，不自动进行超参数搜索。若训练后预算仍不合格，记录本方案未满足约束，不在 official val 上调整系数或阈值补救。

## 11. 必要工程检查：不能变成 mini-train 研究

正式训练前只做与本次改动直接相关的检查，集中记录于 V2_IMPLEMENTATION，不再建立阶段化测试工程。

### 11.1 Gate／cache／serializer

至少验证以下构造序列：

| 场景 | 必须结果 |
|---|---|
| 无 cache，即使 p=0 | UPDATE |
| 有 cache，p=0.49，elapsed=1/4/8/16 | 均 REUSE |
| 有 cache，p=0.50 | UPDATE |
| 有 cache，所有 p<0.5，B4 很宽松 | 不为了花满预算新增 UPDATE |
| 同一 ID 每帧出现的 P8 | k 更新，k+1～k+7 复用，k+8 更新 |
| INIT_ONLY 中长期存在的同一 ID | 本 session 首次初始化后不刷新 |
| scene／session reset 后相同 ID | 重新初始化，不复用旧 session |
| REUSE | latent mirror 不变，observed 状态正常推进 |
| 任意合法 hard flags | `len(packet) == M + 780*U` |
| ST hard-forward | 三个 `x_st` 与真实 decoded 字段一致 |
| query 重排，ID 与字段同步重排 | 缓存按 ID 对齐，输出决策随 query 一起重排 |

构造序列明确标注为单元检查，不包装成真实数据实验结果。

### 11.2 真正的任务梯度

在正式 train-core 管线出现有 optional query 且 `C != L` 的真实样本时，单独检查：

```text
L_native_task → scheduler parameters
L_rate        → scheduler parameters
```

必须证明至少一个有效样本的 native task gradient 为有限、非零，不能只证明总 loss 或 rate loss 有梯度。

必要时使用 `autograd.grad` 做不改变权重的一次检查，之后保持正式 optimizer 行为。某帧因无可选 query、特征相同或原生匹配缺失而梯度为零可正常记录；若始终没有任务梯度，标记 `TASK_GRADIENT_BLOCKED`，不得继续几十轮只靠通信罚训练。

同时检查 native frozen parameters 没有被 optimizer 收集，持久参数／统计量未被意外更新。

## 12. 真正的多 scene batch，目标 batch=4

目标：

```text
batch_size = 4
```

含义是四条不同 scene 的时序流并行：

```text
step t:
    slot0: scene A frame a
    slot1: scene B frame b
    slot2: scene C frame c
    slot3: scene D frame d

step t+1:
    slot0: scene A frame a+1
    slot1: scene B frame b+1
    slot2: scene C frame c+1
    slot3: scene D frame d+1
```

同一 scene 内不打乱时序。scene 间可 shuffle；某条流结束后，该 slot 清空并接入未处理 scene。每个 epoch 全部 train-core 样本恰好处理一次，尾部不足 4 条时按实际有效流数计算。

### 12.1 每条流必须隔离

独立维护：

```text
native vehicle temporal / track state
receiver cache
sender mirror
last-observed state
scene / session / timestamp state
budget
reset state
原生任务所需的历史 GT / matching 状态（若有）
```

共享模型权重，不复制四份完整 CoopTrack 冒充 batch。不得只隔离 Python receiver dict，却共享 native model 内部 track state。

优先复用或最小扩展原生 batch／stream adapter。在必须调整的 native 接口上保留 V1 默认行为，并记录改动。

不同 scene 之间禁止 mutual attention。可以 padding＋mask，也可以对 scheduler 做 per-scene forward；但必须写清 native 高成本模型前向是否真正 batched。

**仅把配置改成 4，或四次 batch=1 串行 forward 再累积梯度，不算 native batch=4 吞吐优化。** 若使用后者作诊断，只能标记 `microbatch=1, accumulation=4`，不能冒充目标完成。

### 12.2 Loss 与更新

一个有效 batch 对各流任务／rate loss 按同一规则聚合，按实际有效样本数归一；一次 backward／optimizer step 后再进行下一时刻。末尾小 batch 不仍除以 4。

流间必须独立；同一流跨时刻按原生训练约定 detach。开发期 dev eval 不得污染训练流状态或 RNG，验证时使用独立 state 并在结束后恢复训练状态。

### 12.3 Batch fallback 与性能记录

在正式完整数据管线上做一次必要的短程资源／吞吐检查，不训练独立 mini dataset、不用这些步骤选模型。

优先 4；只有出现真实 OOM、RAM／commit 不足、或实测吞吐明显不如 2，才锁定为 2。记录：

```text
requested / actual native batch size
effective optimizer batch size
samples per second
data / forward / backward / optimizer time
peak GPU allocated / reserved memory
CPU RAM / host commit（当前环境可取得时）
```

两种模型使用相同正式 batch。若 Transformer 必须降为 2，MLP 正式训练也采用 2；尚未运行的正式阶段遵守该锁定配置，不混用不同配置的最佳 checkpoint。

若 batch=2 也无法在保持协议下实现或运行，报告 `V2_BATCH_BLOCKED` 和具体瓶颈。不自动退回正式 batch=1，不缩小数据集，不自动租赁或购买算力。

与 V1 batch=1 比较时，若软件路径、硬件或额外 teacher forward 不一致，应注明比较边界。没有同条件测量就不能宣称“batch 使速度提升 X 倍”。

## 13. Scene 内固定预算，四轮平衡轮换

预算不再逐帧随机。一个 scene 从开始到结束使用同一 B；native session reset 不更换该 scene 的 B。

对全部 46 个 train scene IDs 生成固定 initial offset，记录在配置；train-core 阶段取对应子集，final retrain 复用同一映射。

```python
budget_index = (initial_offset[scene_id] + epoch_index) % 4
```

`epoch_index` 从 0 开始。连续 4 个 epoch 内，每个 scene 恰好经历 B1/B2/B3/B4 各一次。

生成 offset 时用固定种子并平衡数量，不使用 Python 未固定的内置 hash。相同 scene 的预算不受它位于哪个 batch slot 影响。

dev／official val 对每档预算独立运行，每次整个 scene 固定 B，且每个评估 run 从干净 cache／state 开始。

## 14. Optimizer、学习率与训练上限

固定配置：

```text
optimizer = AdamW
peak_lr = 1e-4
weight_decay = 1e-4
grad_clip_norm = 5.0
precision = FP32
max_epochs = 32
warmup_epochs = 1
warmup_start_lr = 1e-5
min_lr = 1e-6
cosine_total_horizon_epochs = 32
```

不要因为 batch 从 1 变 4 就机械把 lr 乘 4。不切换 AMP／TF32 改变已锁定的数值协议。

学习率按归一化 epoch 进度 u 定义：

```text
0 <= u <= 1:
    lr(u) = 1e-5 + (1e-4 - 1e-5)*u

1 < u <= 32:
    lr(u) = 1e-6
            + 0.5*(1e-4 - 1e-6)
              *(1 + cos(pi*(u - 1)/31))
```

记录每轮真实 step 数和 schedule 进度。该式保持 32-epoch horizon；不是 ReduceLROnPlateau，不根据 dev 动态改学习率。

V2-MLP 与 V2-Transformer 均从头初始化 scheduler，不加载 V1 scheduler 权重。原生两个 checkpoint 保持固定。

## 15. Dev 验证频率与 checkpoint 选择

从 epoch 2 开始，每 2 个 epoch 做一次完整 dev eval：

```text
epochs = 2, 4, 6, ..., 32
budgets = B1, B2, B3, B4
dev scenes = 锁定的完整 8 个
```

每次使用官方 evaluator 对该 run 全部预测聚合计算指标，不将单帧或各 scene 的 AMOTA 简单平均代替正式指标。

每档真实平均通信：

```text
mean_bytes(Bk) = sum(actual packet bytes of evaluated frames) / frame_count
```

若四档同时满足：

```text
mean_bytes(Bk) <= 1.05 * Bk, for every k in {1, 2, 3, 4}
```

该 checkpoint 才有资格参与正式 best 选择。5% 是提前固定的 soft-budget 容差，不因表现好而放宽。

对合格 checkpoint：

```text
A(e) = mean(AMOTA_B1, AMOTA_B2, AMOTA_B3, AMOTA_B4)
P(e) = mean(mAP_B1, mAP_B2, mAP_B3, mAP_B4)
R(e) = mean(mean_bytes_B1, ..., mean_bytes_B4)
```

固定比较规则：

1. 先找所有已评估合格 checkpoint 中最大的 A。
2. 与最大 A 差距小于 0.001 的 checkpoint 进入近似并列集合。
3. 集合内优先 P 更高，再优先 R 更低，再优先更早的 epoch。

按完整历史重新计算，不写成可能随成对比较顺序漂移的临时规则。四档预算共用该模型的同一个 best checkpoint；MLP 与 Transformer 可以各有自己的 best epoch。

禁止：

```text
B1 选 epoch 8，B2 选 epoch 12，B3 选 epoch 18
用 official val 选 epoch
用 target budget 或 S_soft 代替 actual bytes 选模
训练到 12 轮就不看 dev 强制停止
```

### 15.1 早停

```text
max_epochs = 32
earliest_stop_epoch = 12
patience = 4 次连续 dev evaluation
```

有合格 best 之后，连续 4 次完整 dev eval 未产生按上述规则更好的合格 best，且 epoch >= 12，允许早停。

保存 best、latest、历史选择记录及停止原因。若 best 恰好在很早的 epoch 出现，应如实记录，不擅自改成第 12 轮。

若一直没有合格 checkpoint，不把“没有 best”当成一个可比较的分数来伪装早停；在不存在结构性阻断时运行到预设上限并标记：

```text
DEV_BUDGET_CONSTRAINT_FAIL
```

可保留 normalized overflow 最小的 checkpoint 用于诊断：

```text
overflow(e) = mean_k(max(mean_bytes(e,k)/Bk - 1.05, 0))
```

不能把这个诊断 checkpoint 冒充正式 best，也不能以其结果宣称约束满足。结构性预算不可行、任务梯度断裂、数值发散或资源阻断可提前停止对应训练，须给出证据。

## 16. 选模后用全部 46 个 train scenes 从头重训

dev 阶段得到：

```text
MLP best_epoch = E_M
Transformer best_epoch = E_T
```

然后各自从头初始化 scheduler：

```text
MLP:
    all 46 official train scenes
    E_M epochs

Transformer:
    all 46 official train scenes
    E_T epochs
```

不从 train-core 的 best checkpoint 接着训，不把内部 dev 当额外几轮 fine-tuning。

保持同一 batch、场景预算映射、optimizer、loss、gate threshold、量化、serializer、FP32 和初始化协议。epoch 内步数随训练数据量变化，按第 14 节归一化 epoch 进度计算 LR。

**cosine horizon 仍是 32，而不是改成 E_M／E_T。** 否则虽写“训练相同轮数”，学习率轨迹已经换了。

final retrain 不使用 official val 选 checkpoint；达到 E_M／E_T 后，最后一个 checkpoint 就是对应正式 checkpoint。

若某个模型 dev 没有合格 best，标记该模型未获得正式 checkpoint；可以继续另一个模型和 Periodic 的已注册评估，但不得凭空为失败模型指定 E。

输出 final checkpoint 身份、训练 scene／sample 数、真实 epoch 数、batch 和训练配置摘要。

## 17. Official val：冻结方法以后只做最终评估

冻结 V2 源码／配置、两个正式 checkpoint、Periodic 扫描点和主比较配对后，评估：

```text
21 scenes / 3316 samples
10Hz / R50 / car-only
```

运行：

```text
Periodic:
    P1, P2, P4, P6, P8, P10, P12, P16
    若 dev 触发扩展，再运行 P24, P32, INIT_ONLY

V2-MLP:
    B1, B2, B3, B4

V2-Transformer:
    B1, B2, B3, B4
```

同一模型四档预算必须加载同一个 final checkpoint。每档从独立干净状态开始；一个 scene 内固定 B，按 native session 正确 reset。

official val 不用于临时添加 Period、调 gate threshold、增加层数、换 loss 权重、重新选 epoch 或选有利 checkpoint。

V1 旧表可作历史参考。除非 seed、serializer、reset、model／evaluator 身份均能核对一致，否则不能直接把旧 Periodic 数值拼成这轮受控对照。

如果发现实现 bug，说明 bug、影响范围和更正后的协议身份，重跑所有受影响方法；不得只重跑受益的方法或隐瞒旧结果。算法未胜出不等于实现 bug。

## 18. 主比较配对：使用真实通信量，不挑有利点

核心问题：

> 在 Periodic 开始失效的低通信区间，学习器能否让稳定 track 更久 REUSE、让困难 track 提前 UPDATE，以相近实际 Bytes/frame 获得更好的 AMOTA／mAP？

“相同 B1 名称”不等于相同通信量。所有图和主表以 `actual mean bytes/frame` 为横轴。

### 18.1 在 dev 上提前锁定关键比较

在 dev 扫描和两种模型选模完成之后、official val 之前：

1. 若有已确认瓶颈，以第一个 candidate bottleneck 对应的 Period 为关键 Period；若没有，以基础／扩展扫描中最大的有限 Period 作为低通信诊断点，并明确没有确认瓶颈。
2. 在 V2-Transformer 四档预算中，仅按 dev 实际 bytes 与该 Period 的距离，选最接近的一档；平局选较小预算，不按性能挑点。
3. 同时锁定 MLP 的对应预算，并记录其实际通信量是否可比；可以单独记录 MLP 按通信最近的预算作为辅助比较，但规则与选择结果须在 val 前固定。
4. 写入 `configs/main_experiment_v2/primary_comparison.json`。

近似等通信预设标准：

```text
abs(bytes_learned - bytes_periodic) / bytes_periodic <= 0.05
```

若 dev 没有任何预算落入该范围，记录 `NO_MATCHED_RATE_POINT_ON_DEV`，保留最近配对作描述性比较，不临时增加预算或调阈值去制造匹配。

### 18.2 Val 上的解释边界

仍评估完整曲线，但关键配对不变。若 dev 上接近的通信量在 val 上偏离，应如实标记 `RATE_MISMATCH_ON_VAL`，不换成另一个看起来更好看的点。

允许分别报告：

- 近似等通信时的 AMOTA／mAP 差异。
- 学习器通信更少且两项指标均不差时的数值 Pareto 支配。
- 通信更多换来更高精度，或 mAP 与 AMOTA 一升一降时的真实取舍。

不把插值后的虚拟点当作实际实验，不把近似等通信误写成严格相同预算。单次运行的 Pareto 位置也不是显著性证明。

## 19. 随机性控制与关键点补 seed

分别记录训练种子和推理种子。主配置：

```text
split_seed = 20261007
scene_budget_offset_seed = 20261007
training_seed = 0
primary_inference_seed = 0
```

固定并记录 Python、NumPy、torch CPU／CUDA、PYTHONHASHSEED、cuDNN deterministic／benchmark 和 TF32 的配置；沿用兼容的已验证数值设置，不为某个方法临时切换。

每个 eval run 在模型／scheduler 构造、加载完成后，开始数据流之前重新设定推理种子，避免 Periodic 少一次 scheduler 初始化就消耗不同的初始随机状态。dev eval 的 RNG 不能改变之后训练的 RNG 序列。

相同 seed 不保证含非确定性算子的全过程逐位一致，报告实际能保证的范围。

若预先锁定的关键 V2-Transformer 与 Periodic 配对在主 seed 下：

```text
abs(AMOTA_transformer - AMOTA_periodic) < 0.005
```

则对关键 Periodic、Transformer 预算和关键 MLP 对照，补做完整 official val：

```text
inference seeds = 2025, 2026
```

不重训对应 scheduler，不挑最好 seed。报告 0／2025／2026 的各项值、均值和标准差。

这检验的是 **推理种子敏感性**，不是三个独立训练种子的可重复性。不能因此宣称已证明跨训练稳定性或统计显著提升。单 seed 结果也不得直接写“稳定超过”。

## 20. 统一日志、诊断与指标定义

### 20.1 每个完整评估 run

记录：

```text
phase / method / architecture / budget_name / target_budget
period_frames / seed / checkpoint identity / config identity
split identity / sample_count / scene_count
mean / p95 actual bytes per frame
nominal bytes_per_second_at_10Hz = mean_bytes * 10
mAP / AMOTA / AMOTP / MOTA / IDS
query_count / update_count
new-track or new-session forced_init_count
optional learned_update_count / periodic_refresh_count
query_weighted_update_ratio
optional_update_ratio_over_optional_queries
staleness before decision: mean / median / p95 / max
actual inter-update time statistics
mandatory / init / optional-refresh bytes
overflow frame_count / max frame overflow
mean S_soft / mean S_hard / soft-hard gap（学习式方法）
scheduler / pack-unpack / end-to-end latency
peak VRAM
```

UPDATE ratio 用总 UPDATE 数除以总 query 观测数，不简单平均每帧 ratio。无分母的值写 N/A，不写伪造的 0。

AMOTP、IDS 等指标使用官方定义及单位，并标注方向；无法从真实输出获得的指标写明缺项，不编造。10Hz 的 Bytes/s 是名义速率换算，不是在线网络实测吞吐。

时间测量区分 CPU wall-clock 与正确同步后的 GPU 时间；只做轻量测量，不为打点大幅改变所有 run 的性能。缓存查询特征的实验不含路端图像 backbone 在线开销，端到端延迟须写清覆盖范围。

### 20.2 每个训练 epoch

记录 task loss 各分量、rate loss、total loss、LR、optimizer steps、skipped steps、梯度范数、UPDATE／init 比例、probability histogram、预算覆盖、样本吞吐、峰值显存和 epoch 时长。

dev 结果按 epoch×4预算记录，保留 budget eligibility、best 选择、early-stop counter。不要只保存最后一个 epoch。

### 20.3 门控行为分析

按以下 staleness buckets 统计已有 cache 的 query：

```text
[1,2), [2,4), [4,8), [8,16), [16,32), [32,+inf)
```

elapsed=0、无 cache、异常负值另行列出；新初始化 query 不混入 optional gate 概率直方图。

每个 bucket 给样本数、平均 p、实际 optional UPDATE 比例。概率不必随 staleness 严格单调，不要把未出现的单调曲线“修正”出来。

评估期可保存压缩逐 query 诊断：

```text
scene / session / timestamp / frame_idx / predicted_id
has_cache / forced_init / elapsed_before_action
logit / probability / hard_action
last_update_frame_before_action
confidence / confidence_change / track_age
latent_delta_norm / geometry_delta_norm
```

不保存所有训练步的大型 latent dump，不每帧保存完整 attention matrix。日志压缩大小不是通信量。

如发现优势，分析真实的困难片段；可以使用 GT 作离线诊断，但不能把 GT 反向接入 gate。若没有遮挡标签或证据，不把所有性能下降都归因于遮挡。

## 21. 中断恢复与阶段幂等

仅实现必要恢复，不开发新框架。

每个训练阶段至少保存：

```text
latest.pth
best.pth（dev 阶段存在合格 best 时）
scheduler state_dict
optimizer state
LR progress
completed epoch
Python / NumPy / torch CPU / CUDA RNG states
variant / phase / split / budget map / config identity
dev history / best selection state / early-stop counter
```

`state_dict` 中的 scheduler 指学习式调度网络；LR schedule 状态另行明确记录，避免重名混淆。

最简单安全恢复边界为完整 epoch。若不保存并恢复所有 stream cursor／native state／receiver cache，则中断后从上一个完整 epoch 恢复，重新运行未完成 epoch；不要仅凭 frame index 跳过前面的时序历史。

每个 eval 方法／预算／seed 独立保存结果。中断时可以从干净状态重跑未完成的该 run；若按 scene 恢复，必须保留完整预测并在最后重新聚合官方指标。不得把不完整 run 标记 complete。

结果先写临时文件，完成并核对样本数后原子发布。兼容的已完成阶段不重复训练，不对 V1 旧目录写入恢复状态。

## 22. 最终输出结构

至少生成以下内容；不为每一次检查创建独立报告：

```text
configs/main_experiment_v2/
    dev_split.json
    training_v2.json
    primary_comparison.json

experiments/v2/metrics/
    periodic_dev.csv
    v2_dev_history.csv
    periodic_val.csv
    v2_final_results.csv
    key_seed_repeats.csv（触发补 seed 时）

experiments/v2/checkpoints/
    mlp/dev/{latest,best}.pth
    transformer/dev/{latest,best}.pth
    mlp/final/final.pth
    transformer/final/final.pth

experiments/v2/communication/
    每个实际评估 run 的帧级通信记录及必要 query 诊断

experiments/v2/predictions/
    官方 evaluator 所需的完整预测与原始指标

experiments/v2/figures/
    periodic_seconds_vs_amota
    bytes_vs_amota
    bytes_vs_map
    staleness_vs_update_probability

reports/v2/
    V2_IMPLEMENTATION.md
    PERIODIC_SWEEP_DEV.md
    V2_MODEL_SELECTION.md
    V2_FINAL.md

logs/v2/
    train.log
    eval.log
    resource.csv
```

失败模型没有 best／final 时不要创建假文件。模型选择报告说明为什么缺失，并保留诊断 checkpoint 的独立标识。

`training_v2.json` 在正式训练前固化本文约定的模型、预算、loss、batch、LR、seed、早停、选模、量化、缓存和消息协议。运行身份记录于对应阶段 manifest，不重复复制数据。

图使用真实 measured bytes，不用目标预算代替横轴。清楚标注历史 V1 与本轮结果、dev 与 val，不能连成同一条未经区分的曲线。

## 23. 推荐统一执行入口

先在已有项目结构上完成最小实现，不要求机械拆成很多脚本。建议实现：

```text
scripts/v2_run.py
```

其统一接口可以是：

```text
--phase prepare
--phase periodic-dev
--phase train-dev
--phase final-retrain
--phase eval-val
--phase report
--phase all
--resume
```

这些是本任务要实现并验证的接口，不表示当前已经存在。

在现有可运行的 Docker／WSL／conda 环境内执行等价于：

```bash
python scripts/v2_run.py --phase all --resume
```

不要直接在未配置 CoopTrack 的 Windows 系统 Python 下盲跑。复用已有 `run_main.py` 中已验证的环境启动方式，记录最终真实命令；只有实际存在并完成验证的命令才能写入 README／最终报告。

不得默认安装最新依赖、升级 CUDA／PyTorch、修改旧环境或终止无关进程。

## 24. 严格执行顺序

```text
读取现有源码／配置／报告，确认可复用阶段
    ↓
只读保护 V1，建立独立 V2 输出与运行身份
    ↓
实现 V2 cache / Periodic selector / learned gate / ST
    ↓
锁定 38 train-core + 8 dev scene split
    ↓
必要 gate／cache／serializer 检查
    ↓
完整 dev Periodic sweep，按预设条件扩展
    ↓
输出固定刷新瓶颈与通信下限
    ↓
接通并验证 task 梯度与多 scene batch，锁定公共训练配置
    ↓
训练 V2-MLP；每 2 epoch 完整四预算 dev eval 与约束选模
    ↓
训练 V2-Transformer；采用相同验证、选模及早停规则
    ↓
记录 E_M / E_T，提前锁定关键通信配对
    ↓
用全部 46 个 train scenes 从头 final retrain
    ↓
冻结源码、配置与 final checkpoint
    ↓
完整 official val Periodic sweep + V2 两模型四预算评估
    ↓
必要时按预设规则补关键 inference seeds
    ↓
完整表、曲线、诊断、V2_FINAL
    ↓
STOP
```

如果某一分支阻断，保留其他分支能合法完成的结果并明确缺项；不要把不完整实验写成完整主比较。

## 25. 最终成功标准与研究失败处理

分别回答，不合并成一句“成功”：

### A. 工程正确性

V1 是否保持可复现？V2 是否真正移除了 elapsed 硬刷新与预算填槽？ST 前向是否等于真实解码？task gradient 是否可达？batch 状态是否隔离？

### B. 通信约束

四档最终 actual bytes 是否满足预设容差？是否存在不可避免初始化超额？soft 与 hard cost 偏差多大？

### C. 自适应刷新的价值

在 Periodic 低通信困难区，是否有真实近似等通信改善，或更少通信下的性能支配？若没有瓶颈或没有匹配点，应说明证据不足，而不是假定 adaptive 一定有效。

### D. Transformer 相对 MLP 的价值

是否确有超过 MLP 的结果？通信量是否可比？额外计算／延迟是否值得？同预算标签但实际通信差异大的结果不能直接证明 relation modeling 有价值。

### E. 训练改进

最终 batch 是多少？真实吞吐、显存、best epoch、停止原因是什么？没有受控测量不得声称固定加速倍数。

如果 V2-Transformer 没有建立超过 Periodic 的证据，写：

```text
LEARNED_SCHEDULER_ADVANTAGE_NOT_ESTABLISHED
```

如果没有超过 MLP，写：

```text
TRANSFORMER_ADVANTAGE_NOT_ESTABLISHED
```

如果实际预算未满足，写：

```text
COMMUNICATION_CONSTRAINT_FAIL
```

这些可以与“实验流程完整执行”同时成立。工程完成不等于研究假设成立。

不要自动增加网络层数、增加 loss、改变预算、挑 seed、重选 val checkpoint。下一研究方向可根据已保存的 Periodic failure frame analysis 提出，但本任务到此停止，不自动启动 V3。

## 26. 最终汇报格式

最终回答简洁，但必须给真实文件位置与真实结果：

```text
V1_PRESERVED
PASS / FAIL
evidence=

IMPLEMENTATION
backbone_commit=
v2_source_identity=
hidden_staleness_rule_removed=
budget_fill_removed=
hard_forward_matches_decoded=
task_gradient_verified=
native_batch_size=
effective_optimizer_batch_size=

DEV_SPLIT
train_core_scenes=
dev_scenes=
train_core_samples=
dev_samples=
split_identity=

PERIODIC_DEV_SWEEP
completed_points=
extension_triggered=
candidate_bottleneck=
mandatory_floor=
mandatory_plus_init_floor=
table_path=

V2_MLP_DEV
status=
batch=
epochs_completed=
best_epoch=
best_mean_AMOTA=
budget_constraint_pass=

V2_TRANSFORMER_DEV
status=
batch=
epochs_completed=
best_epoch=
best_mean_AMOTA=
budget_constraint_pass=

FINAL_RETRAIN
MLP epochs=
Transformer epochs=
checkpoints=

OFFICIAL_VAL
sample_count=
scene_count=
completed_methods=
full_results_path=

PRIMARY_COMPARISON
preselected_period=
preselected_budget=
actual_rate_match=
Periodic: bytes= / mAP= / AMOTA=
Transformer: bytes= / mAP= / AMOTA=
MLP: bytes= / mAP= / AMOTA=
seed_repeats=

VERDICT
engineering_correctness=
communication_constraint=
adaptive_refresh_advantage=
transformer_over_mlp=
measured_throughput=
limitations=

FINAL
V2_EXPERIMENT_COMPLETE
report=
```

只有规定分支实际完成时写 `V2_EXPERIMENT_COMPLETE`。存在未完成阶段时改写：

```text
V2_EXPERIMENT_BLOCKED
completed_stages=
exact_blocker=
evidence=
minimal_next_action=
preserved_results=
```

失败原因使用真实异常、日志或可验证统计，不能写“可能显存不够”而没有实际证据。

---

## 27. 本轮协议来源与新增实现澄清

历史事实依据已有项目材料：

- `CoopTrack 项目精简 + 主实验完整版执行提示词`：基线 commit、数据口径、冻结范围、V1 主实验要求。
- `APTU Gate and Budget Audit`（2026-10-07）：真实入口、2324 维输入、预算归一化常数、serializer 成本、强制更新占比、ST 与原生任务梯度路径。
- `CoopTrack_V2_Training_Periodic_Sweep_Prompt.md`：Periodic 扫描点、移除两个限制、V2 两模型、scene-level budget rotation、batch=4／2、32 epoch、每 2 epoch dev、早停与全 train 重训。

本整合版补齐实现歧义：soft cost 不等于实际发包期望或硬上限；native task 梯度必须单独验证；真正 native batch 与梯度累积分开报告；final retrain 维持 32-epoch cosine horizon；关键通信配对在 val 前锁定；推理补 seed 不冒充独立训练 seed。

旧文档、本文与本地现状出现差异时，区分“已发生的历史事实”和“本次要实现的行为”。历史事实以实际证据为准；本轮目标以本文为准，并明确记录偏差。禁止把新要求反写成 V1 原本就已实现。
