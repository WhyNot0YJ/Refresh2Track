# CoopTrack：无预算输入、刷新收益监督 MLP 的一站式实验提示词

> 执行对象：Codex  
> 项目根目录：`D:\CoopTrack_Audit`  
> 新实验命名空间：`budget_free/value_v1`  
> 任务：先整理项目索引和新增实验结构，保护全部已有成果；再实现、检查、训练并评估不输入 B1/B2/B3/B4 的刷新决策器。  
> 本文是新的探索实验协议，不覆盖旧 V2，不把尚未完成的消融当作已经完成。

## 0. 本次选定的方法与边界

**主方法：BF_VALUE_H8——无预算输入的 MLP，以短窗口 UPDATE/REUSE 反事实收益监督二分类刷新决策。**

不直接把旧模型的预算输入删掉后继续用 `L_task + L_rate` 训练。采用如下过程：

1. 在 official train 的真实时序状态中选择一个已经初始化的 predicted track ID。
2. 从完全相同的状态分别执行“本帧刷新该 ID”和“本帧复用该 ID”。
3. 两个分支使用冻结 CoopTrack 的真实 **eval 关联与融合路径**，向前运行一个短窗口。
4. 用仅在离线训练阶段可用的 GT 对两个分支的最终输出评分，得到刷新收益差。
5. 用收益差构造“值得更新 / 不值得更新”的监督标签和样本权重，训练小型 MLP。
6. 推理时 MLP 只看发送端当前信息及接收端缓存镜像；不看 GT、未来帧、车端私有状态、带宽或字节预算。

**对照：BF_VALUE_H1。** 使用同一批反事实分支的第一帧收益训练同结构 MLP，用于检查短窗口监督是否比即时监督更有用。不额外生成 H1 分支，不再训练 Transformer。

关键边界：

- 本次只优化高维 latent 的刷新，`ref_pts`、`pred_boxes` 仍逐帧发送。
- 不把本次与几何预测、Geometry-Only、2×2 刷新消融合并。若它们已存在，保护并索引；若未完成，不擅自启动。
- 这是**收益监督的动作分类器**，输出不是经过校准的物理收益值，也不保证是自然频率下的准确概率。
- 不是完整强化学习，不做 BPTT，不声称直接最大化 AMOTA，不继承 DAgger/AggreVaTe 的理论保证。
- 没有在线预算输入，也没有 `L_rate`；仍有预先固定的“最小有意义任务收益”和分类阈值。不要称为“无阈值、无超参数”。
- 不保证一定省通信或提高性能，不保证满足链路容量。结果不好照实保留。

---

## 1. 本轮为什么采用这个设计

先读取现有 `V2_BUG_AUDIT.md`、V2 正式报告及关联机器证据，以本地实际文件为准。

已知审计内容：

- REUSE 与实际融合输入在受控测试中符合协议，未确认关键高维旁路 Bug。
- 一个 cold-start 后第二个真实 batch 中，MLP task 梯度约 `6.62e-7`，rate 梯度约 `0.0237`。**不能推广为全训练过程的比例。**
- 原生训练用几何匹配执行融合，推理用 learned affinity 的离散匹配。
- 当前缓存/时序状态 detach，后续帧损失没有直接回传到前一个刷新 logit。
- 16 个新 V2 run 缺少生成时的预测内容 SHA，属于历史溯源缺口，不是已复现的预测错误。

本轮分别尝试：

- 用直接的标签监督绕开弱的原生 task→gate 梯度链；
- 用真实推理分支生成监督，避免把训练模式下的几何融合当成实际部署后果；
- 用有限窗口的后果构造标签，不靠跨帧反向传播；
- 通过一轮学习策略状态采样，减轻固定缓存历史与部署历史不一致。

以上是实验动机，不是本轮方案优于 V2 的预先结论。

---

## 2. 第一阶段：遍历目录、分类保护、整理结构

### 2.1 先扫描，不先删除

读取项目根目录全部实际目录结构、文件元数据、README、Git status、运行进程和实验索引。检查 Python、PowerShell、Shell、Docker、配置和报告中的引用。

- 不越过项目根目录跟随 junction/symlink；不对原始数据做无必要的全量重复哈希。
- `.git`、权重、数据、缓存只做必要元数据及已有关键哈希检查。
- 不因为目录叫 `old`、`temp`、`audit` 就判断可删除。
- 不擅自终止正在进行的其他实验，不关闭 Docker、不删除镜像、不升级环境。

### 2.2 本轮采用非破坏性整理

**用户明确要求不删除原来的结果。** 因此，本轮开始前已经存在的实验产物默认全部保留。

必须保护：

- Stage1/2/3、R50 对齐报告、V1 全部结果；
- V2 全部正式模型、预测、通信记录、指标、图表、日志、配置、source freeze、run manifest；
- Bug 审计代码和证据，包括尚未闭合的溯源说明；
- Geometry/Vehicle/High-Dim 或 2×2 消融的已完成结果、在研代码和记录；
- 官方 CoopTrack 工作树及本地兼容 patch、Git 历史；
- 官方权重、train/val query cache、原始数据及转换产物。

整理方式：

1. 原有冻结目录原地保留，不移动、不改名、不重写相对链接。
2. 新增或追加一个简洁的实验索引，标明旧 V1、旧 V2、审计、消融和本轮实验的入口与状态。
3. 仅合并**本轮新建**的重复 helper/临时文件，不能为了共用代码改动旧冻结脚本。
4. 如果旧目录确有可整理但有引用风险的部分，只在索引中归类，保留实际位置。
5. 只允许删除本轮白名单工作目录内可重建的临时文件；不删除既有 `.pyc`、日志、失败结果或历史 checkpoint。
6. 路径不确定时从真实清单解析，不能因为示例路径不存在而重建数据或误判原结果遗失。

### 2.3 建议新增结构

遵循现有合理结构，不强制迁移：

```text
scripts/budget_free/
    __init__.py
    run.py                 # 唯一编排入口
    policy.py              # 无预算 MLP、输入构建
    counterfactual.py       # 状态快照、成对推理、采样
    scoring.py             # 离线短窗口监督评分器
    train.py
    evaluate.py

configs/budget_free/
    value_v1.json

experiments/budget_free/value_v1/
    protocol.json
    state.json
    dataset/               # 特征、标签、样本身份、必要复现信息
    checkpoints/           # round0、final_h8、final_h1
    metrics/
    communication/
    predictions/
    figures/
    evidence/
    work/                  # 仅本轮可重建临时文件

logs/budget_free/value_v1.log
reports/budget_free/VALUE_V1_FINAL.md
```

模块可少于上述数量，不要建立新的实验平台、数据库、多 Agent 框架或复杂恢复系统。

若 `value_v1` 已有内容，先核对身份：同一协议且未完成才恢复；同一协议已完成则核验后复用；内容属于另一协议时报告命名冲突，不覆盖、不清空，不自动堆出 `final2/final3`。

---

## 3. 锁定旧实验身份与本轮基础条件

历史参考身份（实际路径从项目读取）：

```text
CoopTrack reference commit:
29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399

V2 MLP final SHA256:
d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c

V2 Transformer final SHA256:
58bda699962f479a7556e75c6ad3b0389d6a909ca7b798e5eece2ea7f2fc65a3
```

只读核对；若实际文件变化，报告，不覆盖、不从头重训旧模型。

本轮基础条件：

```text
official train: 46 scenes / 7362 cooperative samples
official val:   21 scenes / 3316 samples
R50 / car-only / inherited nominal 10 Hz protocol
frozen cooptrack_r50_inf.pth
frozen cooptrack_r50_coop.pth
actual application-layer packet bytes
```

不能把 Git HEAD 当作整个工作树身份。记录关键原生源码、已存在兼容 patch、V2 runtime/serializer/evaluator 的当前哈希。

---

## 4. 预先冻结本轮配置

在生成本轮反事实收益标签之前写入 `protocol.json`。记录这是看到旧 V2 结果后提出的**后续探索版本**，不能声称从未使用旧 val 结果启发设计。

默认配置如下；可在没有查看任何新 val 结果前修正纯实现错误，修正必须留痕：

```json
{
  "experiment": "budget_free/value_v1",
  "main_policy": "BF_VALUE_H8",
  "control_policy": "BF_VALUE_H1",
  "model_seed": 0,
  "sample_seed": 20261009,
  "rollout_seed": 0,
  "budget_input": false,
  "rate_loss": false,
  "geometry_policy": "every_frame",
  "latent_encoding": "inherited_INT8",
  "forced_update": "missing_cache_only",
  "horizon_frames": 8,
  "control_horizon_frames": 1,
  "score_distance_gate_m": 2.0,
  "score_localization_weight": 0.1,
  "minimum_task_gain": 0.01,
  "decision_probability_threshold": 0.5,
  "pilot_pairs": 256,
  "pilot_expansion_cap": 1024,
  "round0_label_target": 4096,
  "round1_label_target": 2048,
  "aggregation_rounds": 1,
  "supervised_epochs": 40,
  "supervised_batch_size": 128,
  "supervised_peak_lr": 0.001,
  "supervised_min_lr": 0.00001,
  "weight_decay": 0.0001,
  "grad_clip": 5.0,
  "teacher_max_frame_forwards": 160000
}
```

这里的批量 128 是**离线特征—标签行**的 batch，不是 128 条 CoopTrack 时序流；40 epochs 是小 MLP 的监督拟合，不是旧 V2 的原生整场景训练 40 轮。

`H=8` 指 `t ... t+7` 八个处理帧。使用真实时间戳记录窗口时长，不宣称精确覆盖 0.8 秒。

本轮不创建 `38+8` train/dev，不在 official val 选 epoch、阈值、窗口、loss 权重或 architecture。训练和标签生成均只用完整 official train 对应的数据范围。

---

## 5. 新模型输入：彻底取消预算条件

决策器只使用当前 sender 可获得的信息：

```text
当前三个 latent
接收端最后收到的三个 latent（sender mirror）
current - last_received
elapsed / staleness
当前及变化的 confidence
当前 ref_pts、ref_pts change
当前 pred_boxes
track_age
```

按审计过的实际构建器核对 `3×768 + 20 = 2324` 维含义。若本地实现不同，先记录真实字段，不能盲目按索引截取。

禁止输入：

- B1/B2/B3/B4、名义预算、剩余预算、预算 embedding；
- 为绕过签名而偷偷传常量 B1、0 或伪 budget token；
- GT、GT ID、未来帧、未来收益标签；
- 车端当前私有轨迹状态、真实关联矩阵或不可获得的车端置信度；
- predicted ID 的数值 embedding、scene ID、文件 token、query 行号。

predicted ID、scene、session 只用于身份与缓存索引。

复用 INT8 roundtrip 输入语义，sender mirror 必须表示**接收端实际解码得到的特征**。不把路端上一帧特征冒充 receiver cache。

可以做固定输入归一化，但统计量只能来自 round0 的 train 特征并保存。当前和缓存 latent 共用对应均值/尺度，delta 只除同一尺度，不单独减去不相容的均值；标量时间可先 `log1p` 再归一化。不得用 val 重估统计量。

---

## 6. 模型结构与部署动作

实现一个轻量、逐 query 的 MLP。预先固定：

```text
每个 768-d latent block：共享 Linear(768,64) + ReLU
current / cached / delta 三个结果拼接 + 20-d state → 212-d
Linear(212,128) + ReLU
Linear(128,64) + ReLU
Linear(64,1) → logit
```

不增加 Transformer、目标间 attention、RNN 或多个预测头。没有 Dropout 或 BatchNorm。

主模型、H1 对照使用同一结构与初始化规则，均从头初始化，不加载旧 MLP 或 Transformer 的参数。

部署：

```python
if missing_receiver_cache:
    action = UPDATE
else:
    p = sigmoid(policy(sender_observation))
    action = UPDATE if p >= 0.5 else REUSE
```

无 TTL、无固定 max_staleness、无 Top-K、无“花满预算”、无几何更新门控。首轮不要添加冷却期，避免引入额外方法变化。

**推理时只执行一次 MLP，不执行双分支、读取 GT 或向未来滚动。**

---

## 7. 标签生成必须使用真实推理链，而不是 `return_loss=True`

教师不是新训练的大模型，而是同一个冻结 CoopTrack 的成对推理后果。

- CoopTrack 处于 `eval()`，使用 `return_loss=False` 的真实推理入口。
- 明确检查 reasoner 的 learned affinity → Hungarian → fusion 与正式 V2 eval 一致。
- 评分器在模型输出之后运行，GT 只能进入评分器，不能修改推理输入、状态、匹配或融合。
- 允许训练阶段的离线评分器查看 `t...t+7` GT；学生输入严格截止当前帧 t。
- 不能用“全帧全部 UPDATE vs 全帧全部 REUSE”得到一个标签，然后复制给所有 query。

原生训练 loss 可以作为附加诊断，但不能替代本轮主标签的 eval-path 输出评分。

---

## 8. 反事实分支的精确定义

对已有缓存的候选轨迹 i，在处理当前帧 t **之前**保存完整前缀状态。

```text
同一个 prefix_state
      ├── U：仅将候选 i 在 t 的 latent 动作置为 UPDATE
      └── R：仅将候选 i 在 t 的 latent 动作置为 REUSE
```

保证：

1. t 的其他 query 动作完全相同。
2. 两个分支都逐帧发送当前几何。
3. t+1...t+7 内，候选 i 若仍属于相同 session/ID 且已有缓存，两个分支都暂不再次刷新它；新的合法 ID 仍初始化。
4. 其他 ID 在窗口内使用从未干预参考轨迹记录的同一动作日程，不在两个分支各自随意重算不同动作。
5. 候选轨迹消失后不伪造其观测；窗口仍继续，允许其影响车端状态。不得只筛选未来始终存活的目标。
6. scene/session 边界前不足 H 帧的候选不进入 H8 数据；不得跨 reset 拼窗口。
7. 除候选在 t 的这次更新外，通信动作计划一致。因此两分支窗口总 payload 差应恰为一个合法 latent 更新成本 780 B。

这样测得的是：

> **在固定后续动作日程、短窗口内不重复刷新该候选的条件下，本帧一次 latent 刷新相对继续复用的边际任务收益。**

它不是全局最优策略的 Q 值，不是多个候选联合更新的精确收益，也不等于任意实际策略的完整长期回报。推理不强制上述候选锁定窗口，必须在报告中披露这个标签—部署差异。

---

## 9. 状态复制与随机性：必须先证明两个分支可比

优先复用已有审计的状态快照方法，并扩展到多帧；不要只复制 receiver cache。

快照至少包括真实执行所依赖的：

```text
receiver cache / sender mirror
vehicle active tracks / Instances / hist_* / cache_*
prev_bev / prev frame information
runtime tracker current_id、生命周期状态
scene/session、姿态和时间状态
评分器的历史 GT→prediction 匹配状态
Python / NumPy / torch CPU / CUDA RNG state
```

权重共享只读；可变 tensor 独立 `detach().clone()`。不要复制两份完整权重到 GPU。

两分支分别恢复相同快照和起始 RNG；采样候选的 RNG 与模型推理 RNG 分离。教师分支结束后必须恢复未干预 roll-in 状态及 RNG，不得让标签生成改变后续采样轨迹。

必须测试：

- 相同动作 R/R 或 U/U 分支重复得到相同输入和输出；
- 交换 U/R 的执行顺序，标签不变；
- 干预前两个分支状态完全相同；
- 不同分支不得共享可变缓存引用；
- native 函数不会读取当前未发送 latent 旁路；
- 候选确实进入过当前或窗口内有效消费路径，不能拿两个全空下游输入相等当作充分测试。

若动作引起后续原生随机分支变化，应记录；不得把一个 paired seed 视为已对随机性取期望。选少量样本附加另一固定 seed 的敏感性检查，只作诊断，不挑标签较好的一次。

---

## 10. 短窗口任务评分：固定、可复现，不冒充 AMOTA

主标签使用独立的离线 tracking-error proxy。**不要在八帧片段上运行 AMOTA 并把它当作单 query 可靠监督。**

对冻结模型输出，按正式协议转换到同一物理坐标系，只评分 car，使用正式输出阈值（从冻结配置读取，不按效果调节）。

匹配与事件提取优先复用项目现有 evaluator/motmetrics 的实现，并保留共同前缀的匹配历史：

- 用 2.0 m 的地面中心距离作为本标签评分的匹配门限；
- 超门限匹配不可接受；处理未匹配和空集合时遵守匹配器语义；
- 稳定排序与平局规则固定；
- FN、FP、IDS 均从各分支自己的输出和共同初始化的匹配历史计算；
- IDS 不是简单比较两个分支的预测 ID 数值，也不是拿 predicted ID 与 GT ID 做数值相等判断。

本轮窗口成本固定为：

```text
E_h = FN_h + FP_h + IDS_h
      + 0.1 * sum_{matched pairs} min(center_distance_m / 2.0, 1.0)

J_H = (sum_{h=0}^{H-1} E_h) / H
Delta_H = J_H(REUSE) - J_H(UPDATE)
```

解释：正值表示这次刷新降低了短窗口错误代理。定位项只用于提供平滑的小收益信号。它是本轮新提出的训练代理，不是官方 metric 定义，不保证与整体 mAP/AMOTA 单调一致。

必须同时保存原始 `FN/FP/IDS/distance/matches/GT count`，不要只保存一个 Delta。

没有 GT 时仍计入 FP，匹配定位项为空时为 0；不能把空 GT 帧全部跳过。

同一八帧分支的第一帧可直接产生 `Delta_1`，不需要重复模型推理。

---

## 11. 标签、收益容忍度与监督损失

先做 R/R、U/U 的 null-action 重复测试，记录数值波动。定义：

```text
epsilon_H = max(0.01, 10 * P99(abs(null_Delta_H)))
y_H = 1 if Delta_H > epsilon_H else 0
```

`0.01` 是预先固定的任务代理收益容忍度，单位是每处理帧的加权错误数，不是字节预算。精确平局及小于容忍度的改善均标为 REUSE。

若 null_Delta 本应确定却出现明显差异，先修复测试状态隔离；不得单靠增大 epsilon 掩盖快照或随机数 Bug。null 波动异常项必须作为正确性失败报告。

标签文件保存 `Delta_H`、`epsilon_H`、`y_H`，以便复核。禁止将 `Delta<0` 错标为 UPDATE。

**损失：收益差加权的二元交叉熵。**

```text
m_H = Delta_H - epsilon_H
s_H = max(P75(abs(m_H) over round0 labels), 0.01)
w_H = clip(abs(m_H)/s_H, 0.1, 10.0)
L = mean(w_H * BCEWithLogitsLoss(logit, y_H)) / fixed_mean_weight
```

`fixed_mean_weight` 是当前训练数据的固定权重均值，不随每个 batch 随意变化。round0 的 `s_H` 锁定并应用于聚合数据。

- 不再加入原生 task loss、rate loss、entropy loss、预算项或 teacher association preserving loss。
- 不额外打开 `pos_weight`、正负各半重采样等，否则会再次改变权衡；需要改动时只能另开研究版本。
- BCE 输入 raw logits，不能先 sigmoid 再喂给 BCEWithLogitsLoss。
- 输出 `sigmoid(logit)` 是成本加权目标下的决策分数，不声称它是刷新真实收益或自然频率下的校准概率。
- 权重截断是优化设计，不能声称精确最小化无限精度反事实 regret。

这不是“最小化所有通信”。它只尝试拒绝没有足够任务收益的更新；可能仍频繁更新，需要用实际 bytes 评价。

---

## 12. 第一个停止点：先确认标签值得学习

只在 official train 上生成 256 个 pilot paired samples，覆盖至少 8 个 scene、多个缓存年龄和有/无有效下游消费的情况。

pilot 是实现及标签信号诊断，不是用于发布性能的小数据集主实验。

必须先通过：

- 快照恢复、null action、分支顺序、无 GT/未来输入泄漏；
- 780 B 成本差、INT8 实际解码、缓存独立；
- 评分器人工小例子的 FN/FP/IDS 和定位成本；
- 所有标签可追溯到 scene/session/token/predicted ID。

输出：正/负/近零标签比例，Delta 分位数，有效 query 数量与 Delta 的关系。

若正标签太少或样本覆盖不足，按同一事先冻结的采样规则补充至最多 1024 个；不能降低 epsilon、换指标权重、专挑能赢的帧。

完成最多 1024 个后，若仍不足 16 个有效 UPDATE 正标签，或者根本没有超过 null 噪声的动作效果：

```text
STATUS = LABEL_SIGNAL_INSUFFICIENT
```

保留诊断和实现，不继续昂贵标注与正式训练。结论仅是“该采样规模和代理定义下的信号不足”，不是高维特征没有价值。

如果能运行但与GT的匹配/状态无法确认，则 `BLOCKED_CORRECTNESS`，不得进入主实验。

---

## 13. round0 标签采样：覆盖完整 train，控制教师成本

在所有 46 个 official train scenes 上建立有效候选清单。可以使用缓存/metadata 做 CPU 预扫描，但不得伪造真实模型前缀状态。

pilot 样本只有在采用最终同一协议、通过正确性检查且 sample key 未重复时才计入 D0；否则作为独立诊断保留，不混入训练标签。

初始 roll-in 策略固定为：

```text
P8
P32
INIT_ONLY
```

这些仅用于**训练状态采样**，不是重跑它们的 official-val 正式实验。每条时序从原生 scene/session 起点按真实顺序推进，不能每个候选都从空状态启动。

目标：共 4096 个有效 paired samples，覆盖所有具有合法候选的 train scenes。

采样要求：

- 先保证 scene 覆盖，再分配剩余配额；
- 覆盖六类年龄桶：0–1、2–3、4–7、8–15、16–31、32+；
- 用稳定 hash 和 `sample_seed=20261009` 选择 token/ID，不能依赖 Python 进程随机 hash；
- 每个选定帧只干预一个 optional ID；同一轨迹设置合理的均匀抽样上限，避免被少量长轨迹占满；
- 不按 GT 难度、未来收益或 val 指标选择候选；不只挑高差值 query；
- 对天然不存在的年龄桶/候选不足 scene，记录缺额，再按固定 round-robin 规则重分配，不能伪造覆盖；
- 记录采样分布，本数据不是所有在线 query 的均匀总体，不声称决策分数已校准到部署总体。

可以流式处理 scene 并暂存少量候选快照，不保存每一帧的完整 GPU 状态，不复制整个图像数据集。

全部新教师前向（包括 roll-in、paired rollout、pilot、聚合轮次）默认上限 160000 个 frame-forwards。到达上限保存可恢复状态并报告 `TEACHER_LIMIT_REACHED`，不得无限自动扩张。

如果真实帧/query 数不足目标，在固定规则下使用全部可用样本并披露，不能回头改 official split。

---

## 14. 标签数据格式与学生信息边界

每个样本至少记录：

```text
sample_key
scene / session / cooperative_token / infrastructure_token / predicted_id
rollin_policy / rollin_policy_sha
current_time / prefix_state_digest / cache_age
sender_observation（学生唯一输入）
Delta_8 / Delta_1 / epsilon_8 / epsilon_1
label_8 / label_1
两分支逐帧任务分项和 payload bytes
effective infra query count（诊断字段，不直接加到学生输入）
source/config/model identity
```

Teacher 可见车端状态和 GT，用于生成标签；学生输入 schema 必须明确白名单。

- 不将标签、参考策略动作、未来框或 GT 匹配结果混入 feature NPZ。
- 归一化仅从 train 观察量计算，不从 teacher 特权字段计算。
- 保存 CPU FP32 特征与紧凑标签；磁盘上的特征数据不是路车通信，不能混入在线通信统计。
- 检查 exact duplicate sample key，不能把同一重复标签当新增独立样本。

---

## 15. 训练 round0 主模型

在 round0 的全部 train 标签上拟合 H8 MLP：

```text
40 supervised epochs
batch_size=128，允许尾批
AdamW lr=1e-3, weight_decay=1e-4
1 epoch linear warmup：1e-4 → 1e-3
后续 cosine，40 epoch horizon，到 1e-5
grad_clip=5
FP32
seed=0
```

- 只训练新 MLP，教师权重不进入 optimizer。
- 小 MLP 训练只读特征和标签，不能再次跑 CoopTrack forward/backward。
- 所有 46 个 train scenes 的合法标签都可用于训练，不留 8 个 scene 做性能选模。
- 固定使用 epoch40；训练 loss 只用于故障监控，不通过 val 挑 checkpoint。
- 训练多数类准确率没有说服力，同时记录 precision、recall、PR-AUC（单类时 N/A）、预测 UPDATE 比例、加权错误、相对 always-REUSE 的经验 regret。
- 这些是拟合数据上的诊断，不称为独立泛化验证。
- 若所有输出相同、梯度异常或无法胜过常量预测，记录，不隐瞒，不自动调学习率/标签阈值直到成功。

保存 `round0_h8.pth` 和全部训练配置/哈希。该文件用于后续状态采样，最终报告不把它与 final 混淆。

---

## 16. 只做一轮自身策略状态补充

固定 round0_h8 模型，在全部 official train scenes 上完整 roll-in，**真正按模型动作更新 receiver cache**。

从它实际访问的状态中按与 round0 一致、预先固定的 scene/age 采样规则采集最多 2048 个新增 paired samples。

反事实定义、H8/H1 评分、epsilon 不变，不用旧 B1 策略替代当前模型历史。

这一步仅用于减轻“在 Periodic 缓存状态上学习、部署时却进入另一种缓存历史”的分布偏移。只做一轮，不无限循环。

合并 D0+D1，重新从头初始化并分别训练：

```text
BF_VALUE_H8：Delta_8 标签
BF_VALUE_H1：Delta_1 标签
```

两者同样固定 40 epochs，同结构、同随机种子、同输入统计、同数据行集合和训练配置；不同的是标签和按各自标签计算的收益权重。

H1 共享的是 H8 策略采样的数据，不声称它具有独立的完全 on-policy 数据聚合过程。

round0 的输入归一化与 s_H 保持锁定。D0/D1 不覆盖，保存来源和 sample key。

不能只接着旧 V2 MLP 训练；不能只挑 round0/final 里 val 更好的那一个发布。

---

## 17. 对未来监督与通信成本的诚实边界

模型看未来仅发生在**离线训练标签制作**，在线不访问未来。

它学的是当前可观察信息对一个特定短窗口边际收益的预测。车端隐状态、遮挡和未来运动未完全包含在 sender 输入中，因此标签可能存在不可预测的条件差异。

本轮不能声称：

- 所有 oracle UPDATE 都可被 sender 精确预测；
- H8 等于充分优化长期跟踪；
- 单 query 边际收益可以线性相加成整帧最优动作；
- 同一个采样策略下测到的收益对所有部署轨迹都不变；
- 无预算输入就必然通信更低，或输出具有硬预算保证。

本轮改变了预算条件、监督形式及有限窗口标签。**它不是只删除 B 输入的单因素实验。** 与 V2 的差异不能全部归因于预算输入移除。

---

## 18. 正式评估前的核心一致性测试

在新的独立副本上测试，不修改原数据：

1. **No-B test**：模型 forward 无 budget 参数或隐藏 budget token，配置改 B 不能影响其行为，因为新协议根本不读取 B。
2. **Observation whitelist**：扰动 label/GT/未来数据字段，不改变学生当前输入。
3. **REUSE test**：固定 flags 并扰动当前未发送 latent，实际模型输入保持缓存值。
4. **UPDATE test**：实际模型输入等于 INT8 decode，不要求等于原始 FP32。
5. **Geometry test**：ref/box 每帧当前，未被本轮冻结或改成预测。
6. **Identity test**：scene/session/ID、目标重排、消失再现、初始化语义保持旧 V2 定义。
7. **Payload test**：保持原 packet 格式，每个 optional latent UPDATE 增加 780 B，无新增评分/标签字段传输。
8. **Frozen-model test**：CoopTrack params/BN buffers、旧 V2 权重和源码前后不变。
9. **No-teacher-at-deployment**：新评估中教师调用次数、GT参与决策次数均为 0。
10. **Artifact identity test**：feature schema、归一化、模型 checkpoint 与冻结评估 manifest 相互匹配。

不能因旧审计缺历史预测 SHA 而称整个项目有 Bug；也不能在新目录补写旧 SHA 来假装补齐历史记录。

---

## 19. 正式 official-val 运行范围

只对以下两个新方法各执行一次完整 primary seed0 评估：

```text
BF_VALUE_H8 final
BF_VALUE_H1 final
```

每个 run：21 scenes、3316 cooperative tokens、同一个 car-only evaluator。

- 模型和全部协议在第一次新 official-val forward 前冻结。
- 不运行 val checkpoint selection、early stopping、阈值扫描或标签参数搜索。
- 不根据 H8 结果修改 H1，不根据任一结果重训。
- 不添加 budget input/Top-K 来强行匹配旧 MLP B1 的 2630 B。
- 如果新策略通信过高、全 UPDATE、全 REUSE、精度退化或 H1 更好，也必须完整保留。
- 有实现性异常则停止受影响 run，留下失败状态；不能把改代码后的结果覆盖到原 run ID。

两个新模型的推理初始化和 RNG 重置位置应遵循经审计的 V2-native 路径；MLP 构造消耗的 RNG 不得意外改变基模型初始化或 eval 起始 RNG。记录实际位置，而不是只记录 `seed=0`。

不顺带启动多训练种子、多推理种子或全部旧实验复跑。这些可以作为下一项明确任务。

---

## 20. 旧结果只读复用，不重跑 P1/P2/P4

从真实文件读取旧数据，禁止仅把本提示词中的数值抄成新 measured results。

优先比较：

```text
INIT_ONLY
P8 / P12 / P16 / P24 / P32 / P6 / P10
V2 MLP B1-B4
V2 Transformer B1-B4
```

历史 P1/P2/P4、FULL_FP32 作为继承参考，保留 serializer/RNG 差异标签；**禁止重跑 P1/P2/P4。**

特别关注这些既有点：

```text
INIT_ONLY: 2137.094692 B/frame, mAP 0.351785, AMOTA 0.331364
P32:       2362.438480 B/frame, mAP 0.359524, AMOTA 0.357565
P16:       2670.816043 B/frame, mAP 0.360143, AMOTA 0.360109
P12:       2876.165862 B/frame, mAP 0.362203, AMOTA 0.361646
MLP B1:    2630.828106 B/frame, mAP 0.362306, AMOTA 0.369316
```

实际成本不等于名义预算。对每个新点展示最近实际 bytes 的 Periodic 点以及上下相邻点，保留全部指标差和 rate gap。

不要线性插值出一个未测“等通信性能”，也不把最近点必然称为严格等成本对照。gap≤5% 可标 `approximate_rate_match`，仍需给真实差值。

计算平均通信、mAP、AMOTA 三目标数值前沿，并单列 IDS/P95，不能因前三项被支配就忽略其他指标。

---

## 21. 必须输出的在线指标和行为诊断

每个新 run：

```text
mAP / AMOTA / AMOTP / MOTA / IDS
mean B/frame / P95 B/frame / total bytes
nominal B/s at 10 Hz（不是实测网络吞吐）
control+geometry bytes / forced-init latent bytes / optional latent bytes
forced-init count / optional count / optional UPDATE count
cache age mean/median/P95/max
six age buckets / ten-bin decision score histogram
age 0-1 的重复刷新占比
scheduler-only latency（注明计时方法）
```

图只生成必要的两张通信—AMOTA/mAP 图，以及一张新旧 MLP 的刷新年龄比较图。底层原始 CSV/JSON 必须保留。

在报告中区分：

- 少花通信，但任务退化；
- 同等或更少通信，任务改善；
- 多花通信换来任务改善；
- 被现有点支配；
- 单个 seed 的微小差异，稳定性未检验。

不要事先写 `ADVANTAGE_ESTABLISHED` 或用 PASS 代替性能判断。

---

## 22. 这次必须闭合新结果的内容身份链

在首次新 official-val 运行之前记录：

- 新代码、配置、特征 schema、归一化、score/epsilon 规则的哈希；
- 原生关键工作树源码及兼容 patch 哈希，而不仅是 Git HEAD；
- MLP final checkpoint SHA；
- train label dataset 和 sample manifest SHA；
- evaluator、serializer、split、RNG 配置。

生成预测后、正式 evaluator 消费前：

1. 原子保存原始预测。
2. 计算文件 SHA256，以及可稳定定义时的解压内容 SHA256。
3. 将这个内容 SHA、run ID、sample count 和模型 identity 写入 run manifest。
4. evaluator 显式读取该路径，验证哈希并把输入 SHA 写进 result identity。
5. 格式化 det/tracking JSON 同样记录 SHA，并由评估读取其对应身份。
6. 评估完成后再次核对输入内容未改变。
7. 无损压缩时保留压缩文件与解压内容的对应身份，不把 gzip 时间头变化误判为预测改变。

这能建立本轮的可追溯链，不声称可追溯性等于指标算法绝对正确，也不补造旧实验的生成时 hash。

---

## 23. 资源、恢复与停止规则

- 使用已有 Python/Docker/GPU，不自动购买算力、不上传原始数据或权重。
- native teacher 以 batch1 参考执行正确性为先；优化并行必须证明状态独立和输出等价。
- 不能为了节省成本共享两个分支的后决策 temporal tensor。只允许复用已证明不依赖动作的只读量。
- OOM 时先释放分支状态、顺序执行、CPU 暂存；不改分辨率、checkpoint、精度、样本 split 或 H 后悄悄继续。
- 文件写入使用 tmp→atomic rename；样本 key 去重；已完成的 label shard 不重新生成。
- `state.json` 记录完成阶段、配对数、前向次数、模型身份和可恢复位置。不要新建复杂恢复平台。
- 不要求保存全部临时 GPU 快照；保存复现所需前缀策略、token、seed、动作日程和证据摘要。
- 本轮结束可以删除本轮 `work/` 中已证明可重建的临时文件。**不能删旧实验，也不能删本轮最终模型、失败证据、标签、配置和正式预测。**

以下是有效停止结果，不得伪装完成：

```text
BLOCKED_CORRECTNESS
LABEL_SIGNAL_INSUFFICIENT
TEACHER_LIMIT_REACHED
BLOCKED_RESOURCE
TRAINING_NUMERICAL_FAILURE
EVALUATION_BLOCKED
```

如果失败，不自动切换成 `Task + comm penalty` 方案、不无限调 epsilon、不启动 RL/BPTT。保留已完成工作，精确报告原因。

---

## 24. 最终报告只写一份

`reports/budget_free/VALUE_V1_FINAL.md`

包含：

1. 旧成果保护核验与新旧目录索引。
2. 本轮无预算模型、字段更新协议和训练目标的定义。
3. 有限窗口反事实、固定 continuation、GT 只用于离线评分的说明。
4. Pilot、D0、D1 标签分布和可学习性诊断；训练数据覆盖与成本。
5. 两个新 MLP 的训练记录、最终 checkpoint 身份。
6. 新旧方法完整通信—任务表，全部有利和不利结果。
7. H8 vs H1：短窗口监督是否体现价值。
8. 与 V2 MLP B1、相近字节 Periodic 的比较。
9. 重复刷新、缓存年龄和额外通信的分析。
10. 新预测→evaluator→metrics 的哈希链。
11. 局限：单 seed、train 中教师已见场景、有限窗口、部分可观测、联合决策非线性、代理评分不等于 AMOTA、无硬带宽保证、标签—部署 continuation 差异。

明确说明：

- official train 生成标签属于正常训练，但教师已在 train 上训练过，训练内指标不能称为对整个系统 unseen 的泛化结果；
- H8 结果更好也不能单独证明“去掉预算输入”是原因；
- 不根据模型名字写胜负；不因为主方法没赢就删除结果。

---

## 25. 严格执行顺序

```text
扫描目录与读取原有事实
→ 保护旧文件身份、整理索引和新目录
→ 冻结本轮 protocol
→ 实现独立快照 / eval-path 成对推理 / score / no-budget MLP
→ null-action、顺序交换、信息隔离和计费正确性检查
→ train-only pilot，检查标签信号
→ D0：覆盖全部 train scenes 的 capped 反事实标注
→ round0 H8 小 MLP 固定轮数训练
→ 全 train 自身策略 roll-in，生成 D1（只一轮）
→ 从头拟合 final H8 与 final H1
→ 固定两个 final 及全部新评估身份
→ official val：只跑两个新方法
→ 只读继承旧 V2 / Periodic 结果进行比较
→ 原始预测与评估 SHA 链核验
→ 整理本轮临时文件，不删旧成果
→ 单份最终报告并停止
```

正常情况下从头到尾执行，不只输出建议或停在写代码。遇到本协议定义的阻塞条件则停止并留下准确证据，不随意换研究方案。

---

## 26. 最终终端回复格式

```text
BUDGET_FREE_VALUE_REFRESH

STATUS=COMPLETE / 明确的停止状态

OLD_ARTIFACTS
V1_preserved=
V2_preserved=
audits_preserved=
ablations_preserved=
old_files_deleted=0
old_frozen_files_modified=0

NEW_STRUCTURE
entrypoint=
config=
artifact_root=

PROTOCOL
budget_input=NONE
rate_loss=NONE
geometry=EVERY_FRAME
latent=INT8_ADAPTIVE
teacher_path=NATIVE_EVAL
main_horizon=8
control_horizon=1
minimum_task_gain=
probability_threshold=0.5
hard_bandwidth_guarantee=NO

LABELS
pilot_count=
round0_count=
round1_count=
train_scenes_covered=
positive_ratio_H8=
positive_ratio_H1=
null_test=
branch_order_test=
no_privileged_input_leakage=
teacher_frame_forwards=

TRAINING
round0_status=
final_H8_sha=
final_H1_sha=
supervised_epochs=40

OFFICIAL_VAL
H8: bytes=; mAP=; AMOTA=; IDS=; P95=
H1: bytes=; mAP=; AMOTA=; IDS=; P95=

COMPARISON
vs_V2_MLP_B1=
nearest_periodic=
rate_gap=
H8_vs_H1=
new_frontier_point=
statistical_robustness=NOT_TESTED

PROVENANCE
prediction_generation_sha=
evaluator_input_sha=
result_binding=

REPORT
reports/budget_free/VALUE_V1_FINAL.md
```

---

## 27. 方法依据与解释边界（不要当成已验证的新方法保证）

- 本地依据：正式 V2 报告、`V2_BUG_AUDIT.md`、真实 `v2_train.py/v2_scheduler.py/v2_eval.py` 与原生 CoopTrack 实际工作树。
- 成本敏感的动作监督可参考 Ross & Bagnell，*Reinforcement and Imitation Learning via Interactive No-Regret Learning*，arXiv:1406.5979。
- 学习策略改变后续状态分布、需要采集自身状态的动机可参考 Ross, Gordon & Bagnell，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，AISTATS 2011，PMLR 15:627–635。
- 加权二分类实现参考当前环境支持的 PyTorch `BCEWithLogitsLoss`；禁止因为新文档存在更高版本就升级旧 CoopTrack 环境。

上述文献支持一般思路，不证明本提示词中 H=8、评分系数、样本数量、MLP 结构或阈值最优。它们是本轮要验证的固定工程选择。

**最后原则：保留全部已有成果，用有限成本验证“真实推理后果监督的无预算刷新”是否值得继续。不要为了获得好看结果改旧数据，也不要在新方案失败时抹掉证据。**
