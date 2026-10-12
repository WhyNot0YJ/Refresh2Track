> **已废止 / RETIRED（2026-10-12）**：V3 Fixed/Shared 固定预算路线已停止，仅保留本文件作为历史思路，以下原执行要求不再生效，不得据此恢复训练或重建已删除产物。废止原因及最终清理范围见 [V3 思路与废止说明](reports/v3/V3_FIXED_SHARED_FINAL.md)。

# CoopTrack V3：Fixed 与 Shared 预算刷新模型完整训练、验证与选模

## 0. 任务目标

项目根目录：

```text
D:\CoopTrack_Audit
```

在完整保留现有 V1、V2、Periodic、2×2 消融、三推理 seed 稳健性实验及 H 路线退役归档的前提下，新增一组独立的 V3 实验。

本次训练四个新的 MLP Scheduler：

```text
Fixed-4K
Fixed-8K
Fixed-12K
Shared-4_8_12K
```

核心研究问题：

1. 固定预算专用模型是否优于一个同时适配多个预算的共享模型？
2. 4K、8K、12K 三种预算是否能形成不同的实际通信工作点？
3. 旧多预算模型的 B1 高分，是低预算专门化造成的，还是多预算联合训练带来的？
4. 在相同名义预算和相近实际通信量下，Fixed、Shared、旧 V2 MLP 与 Periodic 的通信—性能权衡如何？

本次只训练 Scheduler。原生 CoopTrack、图像编码器、检测器、跟踪器全部冻结。

---

## 1. 严格保护现有成果

不得修改、覆盖、删除或重新生成以下内容：

```text
experiments/main/
experiments/v2/
experiments/v2/robustness/
experiments/v2/ablations/
reports/MAIN_EXPERIMENT_FINAL.md
reports/v2/
audits/v2_bug_audit/
reports/budget_free/
query_cache/train/
query_cache/val/
checkpoints/
code/CoopTrack/
code/mmdetection3d/
```

特别保护：

```text
experiments/v2/checkpoints/mlp/final.pth
experiments/v2/checkpoints/transformer/final.pth
checkpoints/cooptrack_r50_coop.pth
checkpoints/cooptrack_r50_inf.pth
```

H1/H8 已正式退役：

```text
RETIRED = BF_VALUE_H8 / BF_VALUE_H1
```

禁止：

- 恢复 H1/H8 源码、标签、权重或预测；
- 重新生成 D0/D1；
- 新建 D2；
- 继续训练或评估 H1/H8；
- 从 H 路线复制已经删除的实现；
- 修改 H 退役归档。

本次所有新增代码和产物必须放入独立 V3 目录。

---

## 2. 开始前检查项目状态

在写代码和启动训练前，先完成只读检查：

1. 检查项目目录结构。
2. 检查是否存在仍在运行的训练、评估或 Docker writer。
3. 检查磁盘剩余空间。
4. 检查四个核心 checkpoint 的 SHA256。
5. 检查 official train、official val、query cache 和 infos 是否存在。
6. 检查原 V2 MLP 的真实训练、消息、缓存、量化和评估入口。
7. 检查 Git 状态，但不要清理或提交已有修改。
8. 检查 H 路线退役后，不存在 V3 对已删除 H 文件的依赖。

检查失败时，只报告准确阻塞项；不得自动重建数据、重新下载模型或恢复 H 路线。

---

## 3. 新实验目录结构

优先采用以下结构：

```text
D:\CoopTrack_Audit
│
├── configs/
│   └── v3_budget_specialization/
│       ├── common.json
│       ├── fixed_4k.json
│       ├── fixed_8k.json
│       ├── fixed_12k.json
│       └── shared_4_8_12k.json
│
├── scripts/
│   └── v3_budget/
│       ├── train.py
│       ├── evaluate.py
│       ├── scheduler.py
│       ├── plot_training.py
│       └── audit.py
│
├── experiments/
│   └── v3_budget_specialization/
│       ├── manifest.json
│       ├── fixed_4k/
│       ├── fixed_8k/
│       ├── fixed_12k/
│       ├── shared_4_8_12k/
│       ├── old_multi_at_new_budgets/
│       ├── comparison/
│       └── figures/
│
├── logs/
│   └── v3_budget_specialization/
│
└── reports/
    └── v3/
        └── V3_FIXED_SHARED_FINAL.md
```

允许根据现有目录做小幅调整，但必须满足：

- 不修改冻结 V2 源码；
- 不复制整个 V2 工程；
- 公共逻辑尽量复用；
- V3 的代码、配置、checkpoint、日志和预测完全独立；
- 不创建大量重复报告。

---

## 4. 数据划分

训练使用完整 official train：

```text
46 scenes
7362 cooperative samples
```

验证使用完整 official val：

```text
21 scenes
3316 samples
10 Hz
R50
car-only
```

本次明确：

```text
不划分 internal dev
不使用 38/8 split
不进行第二次 full-train retraining
```

四个模型从第一轮开始就使用全部 46 个 official train scenes 训练。

Official val 同时用于：

- 周期验证；
- early stopping；
- best checkpoint 选择；
- 最终性能报告。

因此最终报告必须明确：

> official val 被用于 checkpoint selection 和性能报告，并非完全独立、从未参与模型选择的测试集。

不得将其描述为：

```text
untouched held-out test set
```

---

## 5. 四个模型

### 5.1 Fixed-4K

```text
模型：原 V2 MLP Scheduler
预算：始终 4000 B/frame
训练时所有 scene：4K
验证时：4K
```

### 5.2 Fixed-8K

```text
模型：原 V2 MLP Scheduler
预算：始终 8000 B/frame
训练时所有 scene：8K
验证时：8K
```

### 5.3 Fixed-12K

```text
模型：原 V2 MLP Scheduler
预算：始终 12000 B/frame
训练时所有 scene：12K
验证时：12K
```

### 5.4 Shared-4_8_12K

一个共享参数的 MLP Scheduler，同时学习：

```text
4000 B/frame
8000 B/frame
12000 B/frame
```

一次 scene rollout 内：

```text
所有 Track 共享同一个全局预算
预算从 scene 第一帧到最后一帧保持不变
```

禁止：

```text
同一个 scene 内逐帧切换预算
同一个 Track 使用一个预算、另一个 Track 使用另一个预算
每个 query 独立随机预算
```

---

## 6. Shared 的平衡预算轮换

将 46 个 train scene 按确定性顺序排序。

为每个 scene 生成固定初始 offset：

```python
offset = sorted_scene_index % 3
```

第 `epoch` 轮预算为：

```python
budget_index = (offset + epoch - 1) % 3
budgets = [4000, 8000, 12000]
```

示例：

| Epoch | Scene A | Scene B | Scene C |
|---:|---:|---:|---:|
| 1 | 4K | 8K | 12K |
| 2 | 8K | 12K | 4K |
| 3 | 12K | 4K | 8K |
| 4 | 4K | 8K | 12K |

要求：

- 每个 scene 连续 3 个 epoch 恰好经历 4K、8K、12K 各一次；
- 预算映射保存到 manifest；
- 不使用 Python 内置 `hash()` 生成 offset；
- resume 后预算序列必须完全一致；
- 同一个 scene 的不同帧不能发生预算变化。

---

## 7. 预算单位和归一化

本次统一使用十进制字节：

```text
4K  = 4000 B/frame
8K  = 8000 B/frame
12K = 12000 B/frame
```

禁止混用：

```text
4096
8192
12288
```

沿用原 V2 的 Full-INT8 train 平均参考值：

```text
R = 12840.189523169913 B/frame
```

对应预算输入：

```text
4K  / R = 0.31152188157208
8K  / R = 0.62304376314416
12K / R = 0.93456564471624
```

Fixed 模型仍保留原预算输入分支，只是预算标量恒定。

不得为了 Fixed 模型删除 budget embedding，否则会同时改变模型结构，导致 Fixed 与 Shared 无法公平比较。

---

## 8. Scheduler 与输入

严格复用原 V2 MLP Scheduler 的网络结构和参数规模。

每个 query 的输入继续包括：

```text
当前三组 latent
接收端最后实际收到并解码的三组 latent
当前 latent - 接收端缓存 latent
缓存年龄 / elapsed
confidence
confidence change
ref_pts
ref_pts change
pred_boxes / motion
track age
预算标量
```

禁止输入：

```text
GT
未来帧
query 行号
数值 Track ID embedding
车端未来状态
official val 指标
```

predicted track ID 仅用于缓存索引，不作为数值特征输入。

---

## 9. 通信与缓存协议

严格沿用 V2。

### 9.1 几何字段

每帧发送：

```text
predicted track ID
ref_pts
pred_boxes
必要 metadata
```

### 9.2 高维 latent

三组 256 维字段：

```text
query_feats
query_embeds
cache_motion_feats
```

UPDATE 时：

```text
INT8 latent：3 × 256 B
FP32 scale：3 × 4 B
总计：780 B/query
```

REUSE 时：

```text
不发送当前高维 latent
车端使用最后实际收到并解码的缓存
```

### 9.3 Forced UPDATE

仅允许：

```python
if no_receiver_cache:
    UPDATE
```

scene/session reset 后缓存清空，新目标自然触发初始化。

禁止恢复：

```text
max_staleness
TTL
elapsed >= 4 强制更新
Top-K fill
预算槽位强制填满
周期强刷
```

---

## 10. Gate 和 Straight-Through

Scheduler 输出：

```python
logit = scheduler(...)
p = sigmoid(logit)
hard = (p >= 0.5).float()
```

训练时使用：

```python
gate = hard + p - p.detach()
```

要求：

```text
forward：使用真实硬 UPDATE / REUSE
backward：通过 p 近似传递梯度
```

验证时：

```python
UPDATE if p >= 0.5
REUSE  if p < 0.5
```

阈值固定：

```text
0.5
```

不得根据 official val 修改阈值。

---

## 11. Loss

沿用正式 V2：

\[
L_{\text{total}} = L_{\text{task}} + L_{\text{rate}}
\]

其中：

- `L_task`：冻结 CoopTrack 的原生任务损失；
- `L_rate`：单边超预算惩罚。

定义：

```text
M = 当前帧 mandatory 实际字节
c = 780 B / latent update
F = 无缓存的强制初始化 query 数
p_i = optional query 的 UPDATE 概率
R = 12840.189523169913
B = 当前全局预算
```

软通信估计：

\[
E=M+cF+c\sum_i p_i
\]

可实现预算：

\[
B_{\text{feasible}}=\max(B,M+cF)
\]

Rate loss：

\[
L_{\text{rate}}=
\left(
\frac{\operatorname{ReLU}(E-B_{\text{feasible}})}{R}
\right)^2
\]

固定：

```text
lambda_rate = 1.0
```

含义：

```text
超过预算 → 惩罚
低于预算 → 不惩罚
```

预算是软条件，不是硬网络限流，也不要求花满。

不得改成双边损失。

---

## 12. 冻结原生 CoopTrack

必须冻结：

- CoopTrack 全部参数；
- 图像 backbone；
- BEV encoder；
- detection/tracking heads；
- association/fusion 模块；
- BatchNorm 参数及运行统计。

Optimizer 中只允许包含 Scheduler 参数。

训练前输出并核验：

```text
total parameters
trainable scheduler parameters
frozen CoopTrack parameters
optimizer parameter names
```

若发现原生模型参数进入 optimizer，立即停止。

---

## 13. Batch 与时序状态

使用正式 V2 的真实多场景：

```text
batch_size = 4
```

每个 batch slot 对应一个独立 scene 时序流。

每个 slot 独立维护：

```text
vehicle temporal state
receiver cache
sender mirror
scene/session state
budget
RNG state
reset
```

要求：

- 同一个 scene 内帧顺序严格保持；
- 不同 scene 不共享缓存；
- 不同 scene 不共享 Track 状态；
- Shared 模型同一 batch 中不同 scene 可以使用不同全局预算；
- 但每个 scene 内所有 Track 必须共享该 scene 的预算；
- padding 不能计入 gate、loss 或通信量。

---

## 14. 训练配置

四个模型全部使用相同训练资源和最大训练轮数：

```text
max_epochs = 36
training_seed = 0
precision = FP32
batch_size = 4
```

Optimizer：

```text
AdamW
peak lr = 1e-4
weight_decay = 1e-4
gradient clipping = 5.0
```

学习率：

```text
epoch 1：
1e-5 → 1e-4 linear warmup

epoch 2–36：
cosine decay
1e-4 → 1e-6
```

四个模型必须使用相同的：

- optimizer；
- LR schedule；
- batch；
- training seed；
- 数据顺序规则；
- 最大 epoch；
- validation interval；
- early stopping 规则。

禁止：

```text
Fixed 训练12轮，Shared训练36轮
Fixed 与 Shared 使用不同 LR schedule
根据模型结果单独延长训练
根据 official val 修改 loss 或阈值
```

---

## 15. Official val 验证频率

每 3 epochs 进行一次完整 official-val 验证：

```text
epoch 3
epoch 6
epoch 9
epoch 12
epoch 15
epoch 18
epoch 21
epoch 24
epoch 27
epoch 30
epoch 33
epoch 36
```

验证必须覆盖：

```text
21 scenes
3316 samples
car-only
10 Hz
R50
```

模型选择阶段统一使用：

```text
inference_seed = 0
```

RNG 重置位置必须统一：

```text
after scheduler/model construction
after checkpoint loading
after iter(loader) creation
before first next(iterator)
```

同步设置：

```text
Python
NumPy
Torch CPU
Torch CUDA
PYTHONHASHSEED
```

一次完整 scene 内不得重新设置 seed。

验证运行必须完整完成 3316 帧，部分结果不得用于 checkpoint 选择。

---

## 16. Fixed checkpoint 选择规则

### Fixed-4K

每次只验证：

```text
official val @ 4K
```

选择分数：

```text
score = AMOTA_4K
```

### Fixed-8K

选择分数：

```text
score = AMOTA_8K
```

### Fixed-12K

选择分数：

```text
score = AMOTA_12K
```

更新 best checkpoint 的唯一规则：

```python
if current_amota > best_amota:
    best_amota = current_amota
    best_epoch = current_epoch
    save_best_checkpoint()
    patience = 0
else:
    # 包括完全相等
    do_not_update_best()
    patience += 1
```

必须使用原始完整精度 AMOTA。

如果 AMOTA 相同：

```text
不更新 best
不使用 mAP 破平
不使用通信量破平
不使用更早 epoch 破平
```

mAP、MOTA、AMOTP、IDS、通信量仍完整记录，但不参与选模。

---

## 17. Shared checkpoint 选择规则

每次 validation 使用同一个 Shared checkpoint 分别完整运行：

```text
official val @ 4K
official val @ 8K
official val @ 12K
```

计算：

\[
\operatorname{meanAMOTA}=
\frac{AMOTA_{4K}+AMOTA_{8K}+AMOTA_{12K}}{3}
\]

选择分数：

```text
score = mean_AMOTA
```

唯一更新规则：

```python
mean_amota = (amota_4k + amota_8k + amota_12k) / 3

if mean_amota > best_mean_amota:
    best_mean_amota = mean_amota
    best_epoch = current_epoch
    save_best_checkpoint()
    patience = 0
else:
    # 包括完全相等
    do_not_update_best()
    patience += 1
```

禁止：

```text
4K 选一个 epoch
8K 选另一个 epoch
12K 再选另一个 epoch
```

Shared 三档必须共用同一个 best checkpoint。

不使用：

```text
mAP tie-break
通信量 tie-break
归一化字节 tie-break
更早 epoch tie-break
```

实际通信量不参与 best checkpoint 选择。

若某个 checkpoint 的实际通信量超过名义预算，仍然如实记录；不得隐藏，也不得因此修改选模规则。

---

## 18. Early stopping

每个模型独立维护 patience。

最早：

```text
epoch >= 12
```

才允许停止。

规则：

```text
连续4次 official-val 验证
选择分数都没有严格超过历史 best
→ early stop
```

即最多连续约 12 epochs 无改善。

对于 Fixed：

```text
score = 对应单档 AMOTA
```

对于 Shared：

```text
score = 三档平均 AMOTA
```

完全相等视为：

```text
没有改善
```

示例：

```python
if score > best_score:
    best_score = score
    patience = 0
else:
    patience += 1

if epoch >= 12 and patience >= 4:
    stop_training()
```

不得因为训练 loss 继续下降而绕过 early stopping。

不得因为模型看起来“快要变好”而临时增加 patience。

---

## 19. 训练过程必须记录的原始数据

每个模型保存：

```text
metrics/train_history.json
metrics/train_history.csv
metrics/val_history.json
metrics/val_history.csv
metrics/best_selection.json
```

### 19.1 每个训练 epoch 记录

```text
epoch
samples
scenes
optimizer steps
task loss
rate loss
total loss
learning rate
scheduler gradient norm
task gradient norm
rate gradient norm
zero-gradient batch count
non-finite batch count
mean actual B/frame
P95 actual B/frame
optional UPDATE ratio
forced-init ratio
total UPDATE ratio
cache age mean
cache age median
cache age P95
cache age max
seconds
```

统计规则：

- loss 使用样本或帧加权均值，不使用简单 batch 均值掩盖尾 batch；
- Bytes/frame 按真实序列化 packet 长度统计；
- UPDATE ratio 按 query 加权；
- P95 来自逐帧实际字节；
- non-finite 出现时立即停止当前模型训练；
- 不得把 NaN 忽略后继续。

### 19.2 Shared 额外分预算记录

Shared 每个 epoch 除总体统计外，还要按：

```text
4K
8K
12K
```

分别记录：

```text
scene count
sample count
task loss
rate loss
total loss
mean B/frame
P95 B/frame
optional UPDATE ratio
forced-init ratio
cache age
```

不得只保存一个混合总平均。

---

## 20. 每次 validation 必须记录

Fixed：

```text
epoch
budget
AMOTA
mAP
MOTA
AMOTP
IDS
FP
FN
recall
mean B/frame
P95 B/frame
Bytes/s @10Hz
optional UPDATE ratio
forced-init ratio
cache age mean/P95/max
checkpoint SHA256
prediction SHA256
```

Shared：

分别记录：

```text
AMOTA_4K
AMOTA_8K
AMOTA_12K
mean_AMOTA
```

以及三档完整指标。

每次 validation 的原始预测与 evaluator 输入必须在评估前绑定 SHA256。

不得只保存汇总表而删除所有验证来源。

---

## 21. 必须生成的训练曲线

使用训练日志生成真实曲线，不得使用示意数据。

### 21.1 Loss 曲线

分别输出：

```text
figures/task_loss.png
figures/rate_loss.png
figures/total_loss.png
```

包括：

- 四个模型叠加比较图；
- 每个模型单独图；
- Shared 按 4K、8K、12K 分组图。

### 21.2 Validation AMOTA

输出：

```text
figures/val_amota_fixed.png
figures/val_amota_shared.png
figures/val_amota_all_models.png
```

Fixed 图：

```text
Fixed-4K AMOTA
Fixed-8K AMOTA
Fixed-12K AMOTA
```

Shared 图：

```text
AMOTA@4K
AMOTA@8K
AMOTA@12K
三档 mean AMOTA
```

在图中标记：

```text
best epoch
early-stop epoch
```

### 21.3 通信量曲线

输出：

```text
figures/train_mean_bytes.png
figures/train_p95_bytes.png
figures/val_mean_bytes.png
figures/val_p95_bytes.png
```

标出名义预算参考线：

```text
4000
8000
12000
```

同时注明：

```text
nominal soft budget
```

### 21.4 Gate 与缓存行为

输出：

```text
figures/update_ratio.png
figures/cache_age.png
```

记录并展示：

- optional UPDATE ratio；
- forced-init ratio；
- total UPDATE ratio；
- cache age mean；
- cache age P95。

### 21.5 优化过程

输出：

```text
figures/learning_rate.png
figures/gradient_norm.png
```

梯度至少包括：

```text
total grad norm
task grad norm
rate grad norm
```

如果每个 batch 分别计算 task/rate 梯度成本太高，可以预先固定每个 epoch 的若干诊断 batch。

但诊断 batch 规则必须在训练前冻结，不能训练后挑选。

---

## 22. Checkpoint 保存与磁盘控制

训练过程中只保留：

```text
latest.pth
best.pth
```

每次 validation 可以临时保存 candidate checkpoint；一旦验证结束：

- 如果成为 best，替换 `best.pth`；
- 如果不是 best，不保留永久副本；
- `latest.pth` 用于恢复。

训练完成并核验后保留：

```text
best.pth
last.pth
best_selection.json
```

删除：

```text
无用中间 epoch checkpoint
临时预测片段
.partial
.writing
.tmp
```

但必须先完成 SHA 和验证。

不得删除训练日志、验证历史和生成图表所需的数据。

---

## 23. 正确性预检

正式训练前，进行不用于性能结论的最小正确性测试。

必须验证：

1. Fixed-4K 所有 scene 始终为 4000。
2. Fixed-8K 所有 scene 始终为 8000。
3. Fixed-12K 所有 scene 始终为 12000。
4. Shared 同一 scene 内预算不变化。
5. Shared 连续三个 epoch 每个 scene 覆盖三档各一次。
6. 同一 scene 的所有 Track 共享同一预算。
7. 不同 batch lane 的 cache 与 budget 相互隔离。
8. REUSE 真实读取 receiver cache。
9. UPDATE 真实发送当前 INT8 latent。
10. 真实 packet 字节符合 serializer。
11. CoopTrack 参数和 BN 保持冻结。
12. Scheduler 存在非零任务梯度。
13. official val 不参与反向传播。
14. validation 后训练状态能正确恢复。
15. resume 后预算轮换、optimizer、LR 和 best score 不变。

预检只用于实现验证，不得根据预检指标调整模型。

---

## 24. 正式执行顺序

按以下顺序执行，减少并发和显存风险：

```text
1. Fixed-4K
2. Fixed-8K
3. Fixed-12K
4. Shared-4_8_12K
```

每个模型完成：

```text
训练
→ 每3 epoch official-val
→ early stopping / max36
→ best checkpoint 确认
→ 日志和曲线核验
```

一个模型完成后再启动下一个。

禁止四个训练进程同时运行。

---

## 25. 新模型训练结束后的统一比较

首先使用每个模型已经选择好的 best checkpoint 和对应 official-val seed=0 结果。

比较：

```text
Fixed-4K        vs Shared@4K
Fixed-8K        vs Shared@8K
Fixed-12K       vs Shared@12K
```

然后加载旧 V2 MLP checkpoint：

```text
experiments/v2/checkpoints/mlp/final.pth
```

使用本次相同评估协议，新增评估：

```text
Old-Multi@4K
Old-Multi@8K
Old-Multi@12K
```

旧 checkpoint 不重新训练。

最终同预算比较：

| Budget | Fixed | Shared | Old Multi |
|---|---|---|---|
| 4K | Fixed-4K | Shared@4K | Old-Multi@4K |
| 8K | Fixed-8K | Shared@8K | Old-Multi@8K |
| 12K | Fixed-12K | Shared@12K | Old-Multi@12K |

还要按实际通信量，与已有：

```text
old V2 MLP B1
P16
P24
P32
FULL_FP32
```

做描述性比较。

不能只按方法名称或名义预算比较。

---

## 26. 本任务不自动执行的内容

本轮禁止自动：

- 训练第二、第三个 training seed；
- 为所有模型补 inference seed 2025、2026；
- 修改 budget；
- 修改 gate threshold；
- 修改 loss；
- 延长到 36 epochs 以上；
- 增加 Transformer；
- 恢复 H 路线；
- 增加 Delta+32；
- 根据 val 结果再训练新版本。

先完成这四个模型的 seed0 训练、周期验证和统一比较。

后续只有最有竞争力的低通信工作点，才另行决定是否补多 inference seed。

---

## 27. 结果解释规则

不得预设：

```text
Fixed 一定优于 Shared
Shared 一定优于 Fixed
12K 一定优于 4K
通信越多 AMOTA 一定越高
```

允许出现：

- 低预算性能更高；
- 高预算实际通信未花满；
- Shared 与 Fixed 接近；
- Fixed 专用模型更好；
- 旧 Multi-Budget 更好；
- 三档实际通信量集中在相近区间。

所有有利和不利结果都必须保留。

如果 8K、12K 花费更多通信但没有提高 AMOTA，不得通过修改阈值或 loss 重新调到单调。

---

## 28. 最终报告

只生成一份主报告：

```text
reports/v3/V3_FIXED_SHARED_FINAL.md
```

报告至少包含：

### 28.1 实验协议

- 数据划分；
- official val 用于 checkpoint selection；
- 四个模型；
- 预算定义；
- 训练和验证 seed；
- max epoch；
- early stopping；
- best checkpoint 选择规则。

### 28.2 Best checkpoint

| Model | Best epoch | Stop epoch | Selection AMOTA |
|---|---:|---:|---:|
| Fixed-4K | | | |
| Fixed-8K | | | |
| Fixed-12K | | | |
| Shared | | | mean |

Shared 还要列出 best epoch 下三档 AMOTA。

### 28.3 主结果

| Model | Eval budget | Mean B/frame | P95 | mAP | AMOTA | AMOTP | MOTA | IDS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-4K | 4K | | | | | | | |
| Shared | 4K | | | | | | | |
| Old Multi | 4K | | | | | | | |
| Fixed-8K | 8K | | | | | | | |
| Shared | 8K | | | | | | | |
| Old Multi | 8K | | | | | | | |
| Fixed-12K | 12K | | | | | | | |
| Shared | 12K | | | | | | | |
| Old Multi | 12K | | | | | | | |

### 28.4 训练曲线

嵌入全部正式曲线。

### 28.5 结论

回答：

1. Fixed 是否优于 Shared？
2. Shared 一个 checkpoint 是否能接近三个专用模型？
3. 三档是否形成不同通信工作点？
4. 旧 Multi-Budget 是否仍然更有竞争力？
5. 低预算结果是否值得补多 inference seed？
6. 是否存在明显训练未收敛、过拟合或预算失效？

### 28.6 限制

必须写明：

- official val 用于选 checkpoint 和报告；
- selection bias 存在；
- 只有一个 training seed；
- checkpoint selection 阶段只有 inference seed=0；
- Budget 是软约束；
- 原生 CoopTrack 存在推理随机性；
- 当前结果不是统计显著性结论。

---

## 29. 最终机器可读汇总

生成：

```text
experiments/v3_budget_specialization/comparison/final_summary.json
experiments/v3_budget_specialization/comparison/final_summary.csv
```

JSON 至少包含：

```json
{
  "protocol": {},
  "models": {},
  "best_epochs": {},
  "validation_histories": {},
  "training_curves": {},
  "final_metrics": {},
  "communication": {},
  "checkpoint_sha256": {},
  "prediction_sha256": {},
  "comparisons": {},
  "limitations": {}
}
```

---

## 30. 最终终端回复

只输出精简汇总：

```text
V3_FIXED_SHARED

STATUS=PASS/PARTIAL/FAIL

PROTECTION
V1=
V2=
PERIODIC=
H_ARCHIVE=
DATA=
CHECKPOINTS=

TRAINING

FIXED_4K
status=
best_epoch=
stop_epoch=
best_amota=
checkpoint=

FIXED_8K
status=
best_epoch=
stop_epoch=
best_amota=
checkpoint=

FIXED_12K
status=
best_epoch=
stop_epoch=
best_amota=
checkpoint=

SHARED_4_8_12K
status=
best_epoch=
stop_epoch=
amota_4k=
amota_8k=
amota_12k=
mean_amota=
checkpoint=

COMPARISON
fixed_vs_shared_4k=
fixed_vs_shared_8k=
fixed_vs_shared_12k=
old_multi_vs_new=
best_low_rate_point=
needs_inference_seed_repeats=

ARTIFACTS
report=reports/v3/V3_FIXED_SHARED_FINAL.md
summary_json=
summary_csv=
figures=
```

---

## 31. 最终原则

1. 四个模型使用相同最大训练资源。
2. 每 3 epochs 在 official val 完整验证。
3. Fixed 只按单档 AMOTA 选择 checkpoint。
4. Shared 只按三档平均 AMOTA 选择 checkpoint。
5. 只有 AMOTA 严格提高才更新 best。
6. AMOTA 相同不更新。
7. 不使用 mAP、通信量或其他指标破平。
8. 不做第二次 full-train 重训。
9. 不根据结果修改预算、loss 或阈值。
10. 全部训练曲线、通信行为及不利结果必须保留。
11. 不覆盖任何旧 V1/V2 结果。
12. H 路线保持退役。

目标不是强行让 Fixed 或 Shared 获胜，而是用统一训练和选模协议，准确回答预算专用模型与多预算共享模型之间的实际差异。
