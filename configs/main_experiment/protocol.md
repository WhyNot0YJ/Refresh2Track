# CoopTrack 项目精简 + 主实验完整版执行提示词

> **执行对象：Codex**
>
> **项目根目录：`D:\CoopTrack_Audit`**
>
> 本任务分两部分连续执行：
>
> 1. **一次性精简当前项目结构，删除/合并非必须的中间产物；**
> 2. **立即进入正式主实验，直接使用完整 train / val，不再先跑 Example、32帧、单scene、mini-train 或逐步扩大。**
>
> 当前已经完成的数据准备、Baseline 对齐和 Stage3 诊断全部继承，不允许返工。

---

# 0. 必须继承的事实

当前已有结果：

```text
CoopTrack commit:
29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399

Stage1:
STAGE1_SMOKE_PASS

Stage2:
FULL_DATA_READY

Stage3:
STAGE3_DIAGNOSTICS_PASS
```

正式 V2X-Seq-SPD：

```text
vehicle train / val       = 7110 / 3316
infrastructure train / val = 7445 / 3316
cooperative train / val    = 7362 / 3316

train scenes = 46
val scenes   = 21
```

论文一致评价口径：

```text
10 Hz
ResNet50
car / vehicle-only
```

当前本地原生参考：

```text
FULL_FP32:
mAP   = 0.3623447993
AMOTA = 0.3568982508
```

论文 R50 / 10Hz：

```text
mAP   = 0.356
AMOTA = 0.346
```

强通信基线：

```text
FEAT_INT8:
mean payload = 12161.645 B/frame
mAP          = 0.3611074772
AMOTA        = 0.3533432787
```

当前结论：

```text
简单 Top-K Query 删除会明显伤害性能；
INT8 保留全部 Query 时几乎保持精度；
同一预测轨迹的高维特征存在明显时序冗余。
```

本次主实验必须在 **car-only** 口径下汇报。

---

# 1. 本任务禁止返工

不要：

```text
重新下载数据
重新解压数据
重新执行数据converter
重新做全量CRC/SHA审计
重新执行Stage1
重新执行Stage2
重新执行已有FP32/FP16/INT8/Top-K/Random全量实验
重新训练CoopTrack R50主模型
修改official split
修改GT
使用val调训练超参数
```

已有结果必须直接复用。

---

# 2. 第一阶段：只做一次项目结构精简

## 2.1 精简原则

当前根目录存在：

```text
artifacts
checkpoints
code
data
data_infos
datasets
docker
logs
patches
query_cache
reports
scripts
stage2
以及历史Prompt文件
```

目标不是继续建立新的 audit 框架。

这次只做**一次收敛**：

```text
长期实验需要 → 保留
论文最终证据需要 → 保留
一次性中间脚本/日志/过程文档 → 删除
重复副本 → 只留一份
```

不要再生成几十个 cleanup 报告和安全测试。

---

# 3. 清理后的正式结构

最终整理成：

```text
D:\CoopTrack_Audit\
│
├── README.md
│
├── code\
│   └── CoopTrack\
│
├── configs\
│   └── main_experiment\
│
├── checkpoints\
│
├── datasets\
│
├── data\
│
├── data_infos\
│
├── query_cache\
│   ├── train\
│   └── val\
│
├── experiments\
│   └── main\
│       ├── checkpoints\
│       ├── predictions\
│       ├── communication\
│       └── metrics\
│
├── scripts\
│   ├── main_train.py
│   └── main_eval.py
│
├── reports\
│   ├── STAGE1_FINAL.md
│   ├── STAGE2_FINAL.md
│   ├── STAGE3_FINAL.md
│   ├── R50_PAPER_ALIGNMENT_FINAL.md
│   └── MAIN_EXPERIMENT_FINAL.md
│
├── artifacts\
│   └── core\
│
├── docker\
├── patches\
│   └── 40xx_cusolver_compat.patch
│
└── logs\
    └── main\
```

---

# 4. 必须保留

禁止删除：

```text
datasets/**
checkpoints/**
正式 data_infos/**
正式 query_cache/val/**
official split
code/CoopTrack/.git
当前40xx兼容patch
Docker环境复现文件
```

必须保留最终报告：

```text
STAGE1_FINAL.md
STAGE2_FINAL.md
STAGE3_FINAL.md
R50_PAPER_ALIGNMENT_FINAL.md
```

必须保留主实验需要的核心身份信息：

```text
FULL_DATA_READY manifest
train/val split identity
checkpoint manifest / MD5
R50 alignment result
已有 communication baseline 数值
```

---

# 5. 可以删除/合并

全树引用确认后，删除：

```text
__pycache__
*.pyc
.pytest_cache
*.tmp
*.writing
*.partial

attempt1 / attempt2 / resume 中间日志
中途进度日志
安全测试逐项结果
一次性CRC/path-guard脚本
一次性数据解压脚本
一次性converter wrapper
Example专用配置
mini-train专用配置
32帧probe专用脚本
不再调用的Stage1/Stage2辅助脚本
重复配置
重复源码
重复checkpoint
重复report
历史Prompt副本
```

`stage2/`：

1. 找出仍被正式路径引用的少数文件；
2. 移到根目录对应正式位置；
3. 修改正式引用；
4. 删除其余中间产物；
5. 最终删除整个 `stage2/` 目录。

不要因为“以后可能有用”全部保留。

清理结束只生成一个：

```text
reports/CLEANUP_SUMMARY.md
```

内容只记录：

```text
删除了什么
移动了什么
最终结构
删除前/后文件数
主实验入口
```

---

# 6. 清理后只做最小验证

只允许：

```text
Python import
checkpoint load
正式 data_infos 路径读取
query_cache/val 路径读取
git commit检查
```

禁止：

```text
重新跑小数据推理
重新跑Example
重新跑32帧性能
```

验证通过后立即进入主实验。

---

# 7. 主实验方法：APTU

暂定方法名：

```text
APTU
Association-Preserving Temporal Query Update
```

定位：

> 在 CoopTrack 的 Query-level 协同跟踪中，不直接删除目标，而是在路端根据接收端镜像缓存、目标关系和带宽预算，决定高维 Query latent 是否需要在当前帧重新发送。

---

# 8. 主实验核心结构

## 8.1 路端保持原模型不变

```text
Infrastructure Camera
        ↓
frozen cooptrack_r50_inf.pth
        ↓
当前完整 infrastructure track queries
```

路端 CoopTrack 权重完全冻结。

---

## 8.2 Sender-side Receiver-Mirrored Cache

路端维护一份：

```text
last_sent_cache
```

它表示：

```text
接收端最后真正收到的latent状态
```

cache key：

```text
scene
+
native reset session
+
infrastructure predicted track ID
```

禁止：

```text
按query数组行号
使用GT ID
使用“路端上一帧完整特征”冒充接收端缓存
```

UPDATE：

```text
发送当前latent
更新 sender mirror
```

REUSE：

```text
不发送当前latent
sender mirror保持不变
```

scene/reset时：

```text
清空对应cache
```

第一版假设：

```text
可靠、有序网络
```

不研究packet loss。

---

# 9. 通信字段拆分

当前官方待传字段：

```text
query_feats
query_embeds
cache_motion_feats
ref_pts
pred_boxes
```

主实验将其拆成：

## 每帧必须更新的低维状态

```text
track ID / identity metadata
ref_pts
pred_boxes
update flag
必要scene/token/timestamp信息
```

## 可被时序复用的高维latent bundle

```text
query_feats
query_embeds
cache_motion_feats
```

latent bundle：

```text
UPDATE → 当前三个字段全部按INT8发送
REUSE  → 三个字段都不发送，车端继续使用last_received_cache
```

这样避免简单Top-K删除目标。

注意：

所有：

```text
ID
flags
header
shape
dtype
INT8 scale
timestamp
```

必须计入真实payload。

通信量统一：

```text
len(serialized_message)
```

禁止理论维度估算冒充真实通信量。

---

# 10. 轻量关系调度器：Budget-Conditioned Set Transformer

不用简单MLP作为最终方法。

实现一个轻量集合注意力模块：

```text
d_model = 64
num_layers = 2
num_heads = 4
ffn_dim = 128
dropout <= 0.1
```

新增参数目标：

```text
< 0.5M
```

不新增CUDA算子。

不更换PyTorch / MMCV版本。

---

# 11. 每个候选更新token

对于每个 infrastructure track/query，构造：

```text
current latent projection
last_received latent projection
latent delta
time since last update
confidence
confidence change
ref_pts
ref_pts change
box / motion summary
track age
```

先通过小 projection：

```text
→ 64 dim token
```

不要使用：

```text
query数组位置编号
GT信息
车端未来信息
```

作为调度器输入。

---

# 12. Budget Token

加入一个：

```text
bandwidth budget token
```

表示当前帧允许通信预算。

所有 query token 与 budget token 经过两层 set self-attention。

输出：

```text
每个query一个 update score
```

目标：

```text
不是独立判断“变化大不大”

而是在当前目标集合和预算下，
联合决定哪些latent更新更重要。
```

---

# 13. 关联保持训练

主实验不是只训练：

```text
feature L2
```

需要利用 CoopTrack 已有的跨端关联机制。

训练时：

## Teacher reference

```text
完整 current INT8 latent
→ frozen CoopTrack association module
→ reference affinity logits
```

Teacher：

```text
torch.no_grad()
```

不运行第二套图像backbone。

只在同一当前 vehicle state 下复制必要实例状态，计算 full-INT8 reference affinity。

---

## Student

```text
APTU UPDATE/REUSE后的latent
→ 同一个 frozen CoopTrack
→ student affinity
→ detection / tracking loss
```

原始 CoopTrack 参数全部：

```text
requires_grad = False
```

但 student downstream 不允许整体 `torch.no_grad()`。

梯度必须：

```text
task loss
→ frozen fusion/association计算图
→ APTU scheduler
```

---

# 14. Loss

主损失：

```text
L = L_task
  + λ_assoc * L_assoc_preserve
  + λ_rate * L_rate
```

其中：

## L_task

使用当前 CoopTrack 正式训练路径实际启用的：

```text
detection / tracking / association task loss
```

不得自己虚构新的监督。

## L_assoc_preserve

让 student 的跨端关联 logits 接近 full-INT8 teacher：

建议：

```text
BCE / KL / SmoothL1
```

具体选一种最符合现有logit形式的简单损失。

## L_rate

约束期望update payload接近目标预算。

不要直接优化最终 mAP / AMOTA。

---

# 15. UPDATE/REUSE 的可训练门控

训练时需要 hard forward：

```text
UPDATE or REUSE
```

可采用：

```text
Straight-Through Gumbel-Sigmoid
```

或等价最小实现。

要求：

```text
forward = 真正hard gate
backward = 连续近似梯度
```

不引入强化学习。

---

# 16. 最大陈旧时间

为了避免一个track永远不更新：

输入：

```text
time since last update
```

并设置一个简单安全上限：

```text
max_staleness = 4 frames
```

因为当前 CoopTrack 的历史特征主要使用4帧。

当：

```text
age >= 4
```

强制 UPDATE。

这是工程保护，不要包装成核心创新。

后续可做：

```text
2 / 4 / 6
```

消融，但主模型固定4。

---

# 17. 带宽预算：用完整 train 数据确定

不要用val选预算。

先在 **全部46个official train scenes** 上统计：

```text
INT8 baseline真实payload
mandatory-state payload
query count
P50/P95
```

根据完整train统计确定4个固定预算点。

原则：

```text
最低预算 >= 必要状态合理可承载范围
最高预算 < 或接近 full INT8
中间均匀覆盖通信-性能曲线
```

把确定后的预算写入：

```text
configs/main_experiment/budgets.json
```

一旦生成：

```text
锁定
```

禁止根据val结果修改。

---

# 18. 训练数据：直接完整训练

**本任务明确禁止先32帧、单scene、mini-train、Example训练。**

首先生成/确认：

```text
全部46个official train scenes
所需的 infrastructure query cache
```

如果完整train cache不存在：

```text
直接按scene原生顺序生成全部train cache
```

不要先生成一个scene试效果。

cache必须：

```text
从scene第一帧开始
遵守原生reset
保持10Hz
使用官方cooptrack_r50_inf.pth
```

生成完成后直接正式训练。

---

# 19. 正式训练设置

训练：

```text
全部46个official train scenes
所有7362 cooperative train samples
```

scene内部：

```text
严格时间顺序
```

scene之间可以shuffle。

基础模型：

```text
cooptrack_r50_inf.pth   frozen
cooptrack_r50_coop.pth  frozen
```

唯一训练参数：

```text
APTU scheduler
input projection
budget embedding
output head
```

优化器：

```text
AdamW
lr = 1e-4
weight_decay = 1e-4
gradient clip = 5.0
```

主模型参数学习率：

```text
0
```

训练精度优先：

```text
FP32
```

不要为了省显存改变baseline推理数学精度。

如果16GB实际OOM：

```text
记录OOM
然后优先租单卡RTX 5090 32GB继续同一实验
```

不要缩小数据集。

不要改分辨率。

不要减少模型结构。

---

# 20. 训练轮数

直接正式训练，最大：

```text
12 epochs
```

使用：

```text
全部train数据
```

不使用official val做early-stopping超参数选择。

允许根据：

```text
train loss
gradient finite
rate constraint
```

判断训练异常。

最终主checkpoint使用：

```text
最后一个稳定epoch
```

不要在val上挑“最好看”的epoch。

---

# 21. 单模型多预算训练

每个train frame随机采样一个已锁定预算点：

```text
B1 / B2 / B3 / B4
```

通过 budget token 条件化。

目标：

```text
只训练一个 APTU model
```

最终同一个checkpoint在不同budget下运行。

不要为四个budget各训练一套模型。

---

# 22. 正式 Full-Val 主实验

训练完成后：

**直接运行完整 official val：**

```text
21 scenes
3316 cooperative samples
10Hz
car-only
```

所有scene：

```text
从第一帧开始
严格原生顺序
正确reset cache/tracker
```

不要跑mini-val。

---

# 23. 主实验必须跑的条件

已有结果直接复用：

```text
FULL_FP32
FEAT_FP16
FEAT_INT8
CONF_50
CONF_25
RANDOM
```

不要重跑。

新增正式完整版：

## Baseline A — Periodic Update + INT8

```text
固定每2帧更新
固定每4帧更新
```

完整val。

---

## Baseline B — Delta Threshold + INT8

根据：

```text
current latent
vs
last_received latent
```

的变化阈值更新。

阈值只由完整train分布确定。

不得根据val调。

完整val。

---

## Baseline C — MLP Scheduler

保留一个强MLP对照：

```text
输入信息与APTU相同
加global mean/max pooling
参数量尽量匹配
```

正式全train训练。

正式full-val。

---

## Ours — APTU

```text
Set Transformer
+
receiver-mirrored temporal cache
+
association-preserving loss
+
budget token
```

正式全train训练。

正式full-val。

---

# 24. 主消融

只做两个核心消融，避免实验爆炸。

## Ours w/o Association Preserve

```text
去掉 L_assoc_preserve
```

其余完全相同。

完整train + full-val。

## Ours MLP

使用上面的参数量匹配MLP。

作为结构消融。

不再堆更多无必要模块。

---

# 25. 评价指标

所有结果使用：

```text
car-only
```

报告：

```text
mAP ↑
AMOTA ↑
AMOTP ↓
IDS ↓
MOTA ↑
```

通信：

```text
mean Bytes/frame ↓
P95 Bytes/frame ↓
Bytes/s @ 10Hz ↓
update ratio ↓
```

系统开销：

```text
scheduler latency
pack latency
unpack latency
cooperative inference latency
end-to-end pipeline latency
peak GPU memory
```

所有通信量：

```text
真实serialized bytes
```

---

# 26. 主结果表

输出：

```text
reports/MAIN_EXPERIMENT_FINAL.md
```

核心表：

| Method | Bytes/frame | Bytes/s | mAP | AMOTA | IDS | Scheduler ms |
|---|---:|---:|---:|---:|---:|---:|
| Full FP32 | existing | existing | existing | existing | existing | - |
| INT8 | existing | existing | existing | existing | existing | - |
| Periodic-2 | ... | ... | ... | ... | ... | ... |
| Periodic-4 | ... | ... | ... | ... | ... | ... |
| Delta | ... | ... | ... | ... | ... | ... |
| MLP Scheduler | ... | ... | ... | ... | ... | ... |
| APTU w/o Assoc | ... | ... | ... | ... | ... | ... |
| **APTU** | ... | ... | ... | ... | ... | ... |

并输出：

```text
communication-performance curve:
Bytes/frame vs mAP
Bytes/frame vs AMOTA
```

---

# 27. 成功判断

与强基线：

```text
FEAT_INT8
```

比较。

主实验成功的最低要求：

```text
至少有2个固定budget点：
APTU处于INT8 / Periodic / Delta / MLP的Pareto前沿
```

理想：

```text
在明显低于INT8平均通信量时，
mAP / AMOTA只有较小下降；
或者相同Bytes/frame下，
APTU明显优于简单周期/阈值/MLP。
```

不要人为规定必须达到某个百分比后修改val结果。

如结果不理想：

```text
如实报告
不要不断在val上试阈值和结构
```

---

# 28. 训练与验证日志

只保留：

```text
logs/main/train.log
logs/main/eval.log
logs/main/resource.csv
```

不要再生成大量：

```text
attempt1
attempt2
resume
probe
audit
safety test
```

过程文档。

---

# 29. 中断恢复

正式长实验需要可恢复，但只实现最简单的：

```text
checkpoint every epoch
latest.pth
optimizer state
scheduler state
epoch
RNG state
```

中断后：

```text
resume from latest
```

不要重新开发复杂的“恢复框架”。

---

# 30. 最终停止点

完整执行：

```text
项目精简
↓
full train query cache
↓
APTU full training
↓
MLP / w/o-assoc full training
↓
full-val baseline新增实验
↓
full-val APTU
↓
结果表与曲线
↓
MAIN_EXPERIMENT_FINAL
```

然后停止。

不要：

```text
换R101
加Griffin
加第二数据集
重新训练CoopTrack主干
开始论文排版
```

这些等待主实验结果出来之后再决定。

---

# 31. 最终回复格式

最终只需要汇报：

```text
CLEANUP
PASS/FAIL

ACTIVE_PROJECT_STRUCTURE
...

FULL_TRAIN_CACHE
PASS/FAIL
frames=...

APTU_TRAIN
PASS/FAIL
epochs=...
trainable_params=...
peak_vram=...

FULL_VAL
samples=3316
scenes=21

BEST_MAIN_RESULT
budget=
bytes/frame=
bytes/s=
mAP=
AMOTA=
IDS=

INT8_BASELINE
bytes/frame=12161.645
mAP=0.3611074772
AMOTA=0.3533432787

CONCLUSION
APTU better/similar/worse than strong baselines

FINAL
MAIN_EXPERIMENT_COMPLETE
```

如果失败：

```text
MAIN_EXPERIMENT_BLOCKED

exact blocker:
minimal next action:
```

不要自动退回小数据集实验。
