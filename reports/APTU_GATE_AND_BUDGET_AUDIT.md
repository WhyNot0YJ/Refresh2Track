# APTU Gate and Budget Audit

核查日期：2026-10-07。项目根目录：D:\CoopTrack_Audit。

本报告依据当前正式入口所导入的源码、正式运行日志、锁定配置及四份已经保存的 APTU full-val 通信记录。核查仅进行文件读取、标准库统计和示例字节计算；本次唯一新增文件为本报告。

## 1. Actual execution path

### 实际入口

```text
TRAIN ENTRY:
file: scripts/main_train.py
function: main() → train()
line: 322 → 329 → 113

EVAL ENTRY:
file: scripts/main_eval.py
function: main() → evaluate() → MainTemporalReceiver.__call__()
line: 482 → 503 → 297 → 350

APTU SCHEDULER:
file: scripts/main_scheduler.py
class/function: Scheduler.__init__() / Scheduler.forward()
line: 6 / 7 / 26

HARD UPDATE SELECTOR:
file: scripts/main_message.py
function: select_updates()
line: 249

CACHE / SERIALIZER:
file: scripts/main_message.py
function: TemporalCache.prepare() / commit(), pack(), ReceiverCache.unpack()
line: 206 / 240 / 48 / 117
```

主入口 [scripts/run_main.py:69–75](D:/CoopTrack_Audit/scripts/run_main.py:69) 使用 Docker 调用 `/mnt/d/CoopTrack_Audit/scripts/` 下的这些文件并记录命令；[121–129](D:/CoopTrack_Audit/scripts/run_main.py:121) 分别传入 `--variant aptu --epochs 12 --resume` 和 `--method aptu --budget B1/B2/B3/B4 --resume`。

实际训练链是：

```text
main_train.train
→ native.fields(query_cache/train/<inf token>.pkl)
→ int8_roundtrip(raw)
→ TemporalCache.prepare(current, metadata)
→ Scheduler(tokens, normalized_budget)
→ noisy sigmoid probability
→ select_updates(...)
→ pack(raw, flags)
→ ReceiverCache.unpack(...)
→ TemporalCache.commit(...)
→ straight_through_latents(...)
→ 原生车辆协同模型 return_loss=True
```

对应 [scripts/main_train.py:216–248](D:/CoopTrack_Audit/scripts/main_train.py:216)。这里导入的正是 `main_scheduler.Scheduler` 和 `main_message.select_updates`，见 [120–122](D:/CoopTrack_Audit/scripts/main_train.py:120)。

正式验证链是：

```text
main_eval.evaluate
→ pipeline 中用 MainTemporalReceiver 替换 LoadInfInformation
→ native.fields(query_cache/val/<inf token>.pkl)
→ int8_roundtrip(current)
→ TemporalCache.prepare(...)
→ Scheduler(tokens, normalized_budget) 得到 logits
→ select_updates(logits, ...)
→ pack / unpack / commit
→ decoded fields 进入车辆协同模型 return_loss=False
```

对应 [scripts/main_eval.py:350–389](D:/CoopTrack_Audit/scripts/main_eval.py:350)、[393–404](D:/CoopTrack_Audit/scripts/main_eval.py:393)、[428–443](D:/CoopTrack_Audit/scripts/main_eval.py:428)。这不是读取一个未被调用的旧调度器实现。

### 正式运行记录及证据边界

- [logs/main/train.log:482](D:/CoopTrack_Audit/logs/main/train.log:482) 记录正式 APTU 训练命令；[9405–9406](D:/CoopTrack_Audit/logs/main/train.log:9405) 记录第 12 个 epoch 完成和训练 PASS。
- [logs/main/eval.log:4888–4892](D:/CoopTrack_Audit/logs/main/eval.log:4888) 中 B1/B2 为 `reused=True`；这里复用此前完成的结果。B3/B4 的调用及完整验证完成记录见 [4894](D:/CoopTrack_Audit/logs/main/eval.log:4894)、[5241–5243](D:/CoopTrack_Audit/logs/main/eval.log:5241)、[5590](D:/CoopTrack_Audit/logs/main/eval.log:5590)。
- 项目另有 [experiments/main/run_aptu_validation_queue.ps1:6–20](D:/CoopTrack_Audit/experiments/main/run_aptu_validation_queue.ps1:6)：等待 B1 PASS 后，以相同 `main_eval.py` 入口依次调 B2/B3/B4。已有队列日志和产物支持这一入口存在，但不足以完整还原 B1 最初的启动命令。
- [aptu_train.json:188–202](D:/CoopTrack_Audit/experiments/main/metrics/aptu_train.json:188) 记录 12 个 epoch、134273 个 scheduler 参数及冻结原模型；四份 [aptu_B1.json](D:/CoopTrack_Audit/experiments/main/metrics/aptu_B1.json:1)、[B2](D:/CoopTrack_Audit/experiments/main/metrics/aptu_B2.json:1)、[B3](D:/CoopTrack_Audit/experiments/main/metrics/aptu_B3.json:1)、[B4](D:/CoopTrack_Audit/experiments/main/metrics/aptu_B4.json:1) 均记录 PASS、3316 samples、21 scenes 和同一个 scheduler checkpoint SHA256。
- 训练 identity 记录的是 `training.json/budgets.json` 哈希，见 [main_train.py:26–28](D:/CoopTrack_Audit/scripts/main_train.py:26)；验证 identity 记录预算、checkpoint 和官方模型身份，见 [main_eval.py:315–319](D:/CoopTrack_Audit/scripts/main_eval.py:315)。没有登记这些外围脚本的执行时源码哈希。因此运行记录能确认入口与产物身份，不能单独证明历史运行时的每一行源码与今天逐字节相同。本报告追踪当前实际入口的依赖，并用保存的全部通信行核验其硬选择数量及字节公式；没有根据旧 prompt 补写实现。

## 2. Query input construction

### 原始输入和历史状态

字段读取来自原生路侧缓存 `Instances.get_fields()`，见 [scripts/main_runtime.py:139–149](D:/CoopTrack_Audit/scripts/main_runtime.py:139)。合同由 [scripts/main_message.py:15–28](D:/CoopTrack_Audit/scripts/main_message.py:15) 验证：

| 字段 | Shape / dtype | 实际用途 |
|---|---|---|
| query_feats | N×256 / FP32 | current latent 第一部分 |
| query_embeds | N×256 / FP32 | current latent 第二部分 |
| cache_motion_feats | N×256 / FP32 | current latent 第三部分 |
| ref_pts | N×3 / FP32 | 当前几何状态及 ref delta |
| pred_boxes | N×10 / FP32 | 当前框状态，含速度分量 |
| scores | N / FP32 | 路侧预测 confidence |
| obj_idxes | N / int64 | predicted track ID，用于查历史缓存，不直接作为网络特征 |

`current['scores']` 是路侧检测置信度，不是 scheduler 的 UPDATE score。原生 confidence 来自分类 logits 的 sigmoid 最大值，见 [cooptrack.py:644–646](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:644)、[669–670](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:669)。

训练和验证都先对三个 latent 做 INT8 编码再反量化，然后构造 scheduler 输入；confidence、ref_pts、pred_boxes 保持 FP32。见 [main_message.py:32–45](D:/CoopTrack_Audit/scripts/main_message.py:32)、[main_train.py:219–221](D:/CoopTrack_Audit/scripts/main_train.py:219)、[main_eval.py:355–357](D:/CoopTrack_Audit/scripts/main_eval.py:355)。

历史字段的区别非常具体：

- `last`：该 predicted ID 最后一次实际 UPDATE 后保存的 INT8 反量化 latent；不是上一帧重新生成的 latent。
- `elapsed`：当前路侧 `frame_idx` 减最后 UPDATE 的 `frame_idx`。
- `confidence change`、`ref_pts delta`：当前值减该 ID **上一次被观察/处理时**的值；即使上一帧 REUSE，observed 也会更新。
- `track_age`：当前 `frame_idx` 减本 session 内第一次观察该 ID 的 `frame_idx`；不是原生 tracker 的 age 字段。

依据 [main_message.py:206–246](D:/CoopTrack_Audit/scripts/main_message.py:206)。

### N×2324 的精确拼接

令每个 latent 拼接为 768 维。实际输入顺序见 [main_message.py:190–195](D:/CoopTrack_Audit/scripts/main_message.py:190)：

| tokens 列位置 | 内容 | 维数 |
|---|---|---:|
| 0:768 | current 三种 latent 拼接 | 768 |
| 768:1536 | last-received 三种 latent 拼接 | 768 |
| 1536:2304 | current − last-received | 768 |
| 2304 | elapsed / staleness 帧索引差 | 1 |
| 2305 | 当前 confidence | 1 |
| 2306 | confidence change | 1 |
| 2307:2310 | 当前 ref_pts | 3 |
| 2310:2313 | ref_pts delta | 3 |
| 2313:2323 | 当前 pred_boxes | 10 |
| 2323 | track_age | 1 |

总计 `768×3+20=2324`。没有独立新增 motion-summary 张量：motion 由 `cache_motion_feats` 和 `pred_boxes` 提供。10 维框的编码为 `cx, cy, log(w), log(l), cz, log(h), sin(yaw), cos(yaw), vx, vy`，见 [core/bbox/util.py:6–18](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/core/bbox/util.py:6)。

`cache_motion_feats` 来源为原生 MotionExtractor 与历史 motion 更新，见 [motion_extractor.py:38–56](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/motion_extractor.py:38)、[cooptrack.py:1201–1203](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1201)、[spatial_temporal_reason.py:146–162](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:146)。ref delta 构造处直接相减，没有另加跨帧坐标变换。

预算没有塞进这 2324 列；它作为独立 scalar 输入下面的 budget token。predicted ID 用于缓存索引，没有把 ID 数值或 query 行号拼接进这些 token。

## 3. Budget encoding

### A. 数值来源和归一化

锁定预算来自 [configs/main_experiment/budgets.json:3–9](D:/CoopTrack_Audit/configs/main_experiment/budgets.json:3)：

| 名称 | B，bytes/frame | 网络输入 b=B/R |
|---|---:|---:|
| B1 | 3474 | 0.2705567541 |
| B2 | 6382 | 0.4970331620 |
| B3 | 9290 | 0.7235095700 |
| B4 | 12198 | 0.9499859779 |

其中 `R=full_int8_mean_bytes=12840.189523169913`，是**完整 train** 的 INT8 均值；不是 `max_budget`，也不是 val 的 INT8 均值 12161.645。

训练每帧 `random.choice(budgets)`，见 [main_train.py:222–226](D:/CoopTrack_Audit/scripts/main_train.py:222)。验证通过 B 名称的序号读取同一数组，见 [main_eval.py:305–307](D:/CoopTrack_Audit/scripts/main_eval.py:305)，同样传 `B/R`，见 [370](D:/CoopTrack_Audit/scripts/main_eval.py:370)。

网络看到的是名义预算 B 的归一化值，没有先减 mandatory bytes 或 forced UPDATE 成本。

### B. Budget embedding 和 attention

真实编码为：

```text
B/R
→ tokens.new_tensor(...).reshape(1,1)          [1,1]
→ Linear(1,64) → ReLU → Linear(64,64)         [1,64]
→ 在 N 个 query token 后追加 budget token    [N+1,64]
→ unsqueeze(1)                              [N+1,1,64]
→ 两层 TransformerEncoder self-attention
→ 丢弃最后一个 budget 输出 token             [N,64]
```

依据 [main_scheduler.py:14](D:/CoopTrack_Audit/scripts/main_scheduler.py:14)、[31](D:/CoopTrack_Audit/scripts/main_scheduler.py:31)、[38](D:/CoopTrack_Audit/scripts/main_scheduler.py:38)。

budget token **实际参与 self-attention**，不是算完 attention 才附加的特征。`[:-1,0]` 只丢弃最终 budget token 的输出；query 输出已经受到该 token 的 attention 影响。

### C. Budget 同时直接进入硬选择

**对应用户所列方案 2：Budget 作为 Transformer 输入，同时在最终 UPDATE mask 中根据真实字节成本限制可选 UPDATE 数量。**

第二条路径是 [main_eval.py:371](D:/CoopTrack_Audit/scripts/main_eval.py:371) / [main_train.py:230](D:/CoopTrack_Audit/scripts/main_train.py:230) 把未归一化的 B 直接传给 [main_message.select_updates:249–259](D:/CoopTrack_Audit/scripts/main_message.py:249)。它不是只靠网络“学会少发”，也不是只在 rate loss 中使用 B。强制 UPDATE 可以越过该预算限制。

## 4. Score generation

APTU 实际类使用 `nn.TransformerEncoder`；这里“set”表现为把 query 集合作为无额外位置编码的序列输入。不是调用另一个名为 SetTransformer 的旧类。

完整张量路径：

```text
tokens                                      [N,2324]
current / last / delta 三块                  各 [N,768]
共享 Linear(768,64)，分别调用三次             各 [N,64]
concat 三个投影 + 20维状态                   [N,212]
Linear(212,64) → LayerNorm(64) → ReLU         [N,64]
concat budget token / unsqueeze              [N+1,1,64]
TransformerEncoder：
  2 layers，d_model=64，4 heads，FFN=128，
  dropout=0，FFN默认激活ReLU
去掉 budget 输出 token                       [N,64]
Linear(64,1) → squeeze(-1)                   [N] 原始 logits z
```

源码：[main_scheduler.py:12–22](D:/CoopTrack_Audit/scripts/main_scheduler.py:12)、[26–39](D:/CoopTrack_Audit/scripts/main_scheduler.py:26)。`latent_projection` 只是 Linear，没有额外激活；最后 `output_head` 后也没有 sigmoid。

实际 Docker 环境中的 PyTorch 源码也经只读核对：

- `/home/yujie/miniconda/envs/cooptrack-smoke/lib/python3.8/site-packages/torch/nn/modules/transformer.py:285–287`：默认 `activation="relu"`、`batch_first=False`。
- 同文件 `320–326`：self-attention 的 Q/K/V 都来自完整输入，随后 residual、LayerNorm 和 FFN。
- `.../torch/nn/functional.py:4828`：softmax 用于 attention 权重；**这不是 UPDATE probability 的 softmax**。

训练在 logits 后增加随机 Logistic 噪声和 sigmoid；验证直接使用 logits。该区别在下一节给出。

## 5. Score → UPDATE/REUSE

### 不存在固定阈值

没有 `score>=0.5`、`logit>=0` 或“分数太低就不发”的判断。`select_updates` 只要求输入分数有限，然后进行排名；负 logits 也可以成为 UPDATE。

### 训练随机量与 ST

[main_train.py:227–228](D:/CoopTrack_Audit/scripts/main_train.py:227)：

\[
u_i=\mathrm{clamp}(\mathrm{rand\_like}(z_i),10^{-6},1-10^{-6}),
\qquad
p_i=\sigma\left(\frac{z_i+\log u_i-\log(1-u_i)}{\tau}\right).
\]

当前 `tau=gate_temperature=1.0`，见 [training.json:9](D:/CoopTrack_Audit/configs/main_experiment/training.json:9)。没有温度退火。源码没有调用 `gumbel_softmax`，也没有对每个 query 单独做 Bernoulli hard sample；hard flags 是对这些带噪声的 p 做预算排名选择。

真实 serializer 使用 detached hard flags。训练的 differentiable latent 则为：

\[
q_i=\begin{cases}1,&force_i\\p_i,&\text{otherwise}\end{cases},
\quad S_i=q_i C_i+(1-q_i)L_i,
\quad X_i=D_i+(S_i-\mathrm{detach}(S_i)).
\]

C 是当前完整 INT8 反量化 latent，L 是最后收到的 latent，D 是当前包实际解码的 UPDATE/REUSE 结果。前向 `X=D`；反向使用 S 的梯度。三个 latent 分别执行这一操作，并检查前向逐元素等于 D，见 [main_scheduler.py:42–51](D:/CoopTrack_Audit/scripts/main_scheduler.py:42)、[main_train.py:230–243](D:/CoopTrack_Audit/scripts/main_train.py:230)。

对于非强制 query，局部 surrogate 导数为：

\[
\frac{\partial X_i}{\partial z_i}
=(C_i-L_i)\frac{p_i(1-p_i)}{\tau}.
\]

这是 soft surrogate 的梯度，不是对 hard 排名操作求导；强制 query 的 q=1 不经 p，其这一梯度为零。即使某可选 query 此帧 hard REUSE，surrogate 仍可能向其 score 传梯度。

### 精确硬选择：同成本的预算 Top-K

每个 UPDATE 都是 c=780B。真实实现不是为不同 query 逐一计算不同成本的 knapsack，而是先直接算整数容量，再取排名前 slots 个非强制 query：

```python
# 完全对应当前 select_updates 的行为
flags = force.detach().clone()
slots = max(
    0,
    int((float(budget) - mandatory) // per_update)
    - int(force.sum())
)
if slots:
    order = np.lexsort((
        np.arange(len(scores)),             # 同分时按输入行号升序
        -scores.detach().cpu().numpy()      # score降序
    ))
    chosen = [int(i) for i in order if not bool(force[i])][:slots]
    flags[chosen] = True

update_mask = flags
reuse_mask = ~flags                         # 解释性名称；源码直接使用 flags
```

依据 [main_message.py:249–259](D:/CoopTrack_Audit/scripts/main_message.py:249)。所有强制 query 先保留，不会因预算不足撤销。非强制候选没有 score 门槛，只要有名额就选择；候选少于名额时全部选择。同分的行号 tie-break 是 hard selector 的规则，行号不是 scheduler 输入特征。

## 6. Forced update rules

### FORCED UPDATE CONDITIONS

1. **本 scene/session 内没有该 predicted ID 的已收 latent**：`saved is None`，强制 UPDATE；包括首次出现的 track。见 [main_message.py:223–225](D:/CoopTrack_Audit/scripts/main_message.py:223)。
2. **距离最后 UPDATE 的路侧帧索引差 >=4**：已有 cache 时 `elapsed = frame-saved['frame_idx']`，`elapsed>=4`。见 [227–228](D:/CoopTrack_Audit/scripts/main_message.py:227)。
3. **scene 或 native reset session 改变**：先清空 cache/observed/first_seen；随后出现的每个 ID 都落入条件 1。见 [208–211](D:/CoopTrack_Audit/scripts/main_message.py:208)。session metadata 在首次、scene 切换、时间倒退或时间间隔大于 500000µs 时重建，见 [main_runtime.py:157–171](D:/CoopTrack_Audit/scripts/main_runtime.py:157)。
4. **训练新 epoch / 验证起始**：新建 TemporalCache 和 ReceiverCache，相当于空 cache；首批 query 因条件 1 强制 UPDATE。见 [main_train.py:200–202](D:/CoopTrack_Audit/scripts/main_train.py:200)、[main_eval.py:339–340](D:/CoopTrack_Audit/scripts/main_eval.py:339)。

优先级：`prepare` 先产生 force；`select_updates` 先把 force 全部置 True，再用余下名额选择 scheduler UPDATE。

### 容易误解的几种情况

- **track ID change**：没有额外“检测 ID 变更”分支。新 ID 在 cache 中不存在才强制 UPDATE；若它已经是本 session 的已缓存 ID，则按现有 cache 和 elapsed 处理。
- **invalid cache / malformed packet**：有限性、ID、CRC、schema、顺序或无缓存 REUSE 检查失败会抛异常；没有自动改成 UPDATE 的 fallback。见 [main_message.py:15–28](D:/CoopTrack_Audit/scripts/main_message.py:15)、[118–156](D:/CoopTrack_Audit/scripts/main_message.py:118)、[170–184](D:/CoopTrack_Audit/scripts/main_message.py:170)。
- **geometry / required state**：ref_pts、pred_boxes、ID 和 UPDATE 标记每帧都发；这不是将所有 query 的 latent flag 强制为 UPDATE。REUSE 也携带当前几何。见 [64–71](D:/CoopTrack_Audit/scripts/main_message.py:64)、[167–182](D:/CoopTrack_Audit/scripts/main_message.py:167)。
- query 暂时缺席没有触发 cache 删除；重新出现时仍按该 ID 的历史计算 elapsed/track_age，见 [206–246](D:/CoopTrack_Audit/scripts/main_message.py:206)。

### max_staleness=4 的准确作用

当前 `4` **写死在 `TemporalCache.prepare` 中**。`training.json.max_staleness=4` 虽然同值，但 prepare 没有读取这个配置项，见 [training.json:10](D:/CoopTrack_Audit/configs/main_experiment/training.json:10)、[main_message.py:228](D:/CoopTrack_Audit/scripts/main_message.py:228)。

没有 `staleness += 1` 的计数器：每次 prepare 用当前帧索引减最后 UPDATE 帧索引重算。

- 新 track：`elapsed=0`、`track_age=0`、last latent 为零，仍因无 cache 强制 UPDATE。
- UPDATE：commit 覆盖三个反量化 latent，记录当前 `frame_idx`；该帧成为新的零距离参考时刻。下一连续帧 prepare 的 elapsed 为 1。
- REUSE：不改 latent cache，不改保存的 UPDATE 帧索引；但 observed confidence/ref 会更新。
- 若在帧 k UPDATE，k+1/k+2/k+3 可以 REUSE；到 k+4 已满足 `>=4`，必须 UPDATE。不是先完成四个 REUSE 再在第五帧强制。
- elapsed 是**路侧时间轴索引差**，不是连续 REUSE 次数；目标缺席或 cooperative 配对跳帧同样增加差值。metadata 与训练配对来源见 [main_runtime.py:157–171](D:/CoopTrack_Audit/scripts/main_runtime.py:157)、[main_train.py:150–151](D:/CoopTrack_Audit/scripts/main_train.py:150)、[216–221](D:/CoopTrack_Audit/scripts/main_train.py:216)。

Receiver 不另算 staleness；它按收到的 flag 更新或复用。其 scene/session reset 和 REUSE 重建见 [main_message.py:132–135](D:/CoopTrack_Audit/scripts/main_message.py:132)、[177–185](D:/CoopTrack_Audit/scripts/main_message.py:177)。

## 7. Train vs Eval differences

| Mechanism | Train | Eval | Same? |
|---|---|---|---|
| budget input | 每帧随机从四个 B 选一个；输入 B/R | 每次 full-val 固定一个 B；输入 B/R | 同公式，选择方式不同 |
| score generation | 同一 Scheduler 结构，GPU FP32，得到 logits | 加载最终 checkpoint，CPU FP32，得到 logits | 同结构/权重定义，执行设备不同 |
| ranking score | 带 Logistic 噪声的 sigmoid probability | 原始 logits；无 sigmoid、无噪声 | **不同** |
| hard gate | 同一个 select_updates，先 force 再预算 Top-K | 同一个 select_updates | 同选择器，但排名可能不同 |
| forced update | 无 cache 或 elapsed>=4 | 同规则 | 相同 |
| staleness | 同 TemporalCache，原生路侧索引差 | 同 TemporalCache | 相同计算，历史随选择演化 |
| serialization | hard flags → pack/unpack | hard flags → pack/unpack | 相同 |
| differentiable receiver latent | decoded hard forward + soft ST backward | 直接 decoded；no_grad | 不同 |
| rate / association preserve loss | 计算，并反传到 scheduler | 不计算 | 不同 |
| scheduler mode | .train()，dropout本来就是0 | .eval()，dropout为0 | 没有 dropout 随机差异 |

源码：[main_train.py:159](D:/CoopTrack_Audit/scripts/main_train.py:159)、[222–256](D:/CoopTrack_Audit/scripts/main_train.py:222)；[main_eval.py:335–338](D:/CoopTrack_Audit/scripts/main_eval.py:335)、[368–379](D:/CoopTrack_Audit/scripts/main_eval.py:368)；[main_scheduler.py:20](D:/CoopTrack_Audit/scripts/main_scheduler.py:20)。

因此不能说 train/eval 的 `score→flags` 完全一致：容量、force 和 serializer 相同；训练的随机 ranking score 与验证的 deterministic logits 排名不同。验证没有额外第二套 threshold 或排序，唯一排序就在共用 select_updates 内。

此外，原生 association 下游也有训练/验证差异：训练计算 learned affinity 后，fusion 匹配使用几何规则；验证使用 affinity sigmoid 后的 Hungarian。见 [spatial_temporal_reason.py:238–244](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:238)、[310–359](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:310)。这不改变上述预算 hard selector 的公式。

## 8. Rate loss

定义：

- M：将当前帧所有 flags 置 False 后，**实际 pack 长度**。
- c=780：每个 UPDATE 的附加字节成本。
- N：当前 query 数。
- F：强制 UPDATE 数。
- T=M+780N：完整 UPDATE 的实际包长度。
- R=12840.189523169913：完整 train 的 inherited INT8 均值。
- q：强制位置为 1，其他位置为训练 noisy sigmoid probability。

真实公式为：

\[
E=M+780\sum_i q_i,
\quad B_{\mathrm{feasible}}
=\max\{\min(B,T),\,M+780F\},
\]
\[
L_{\mathrm{rate}}
=\left(\frac{E-B_{\mathrm{feasible}}}{R}\right)^2.
\]

源码：[main_message.py:78–85](D:/CoopTrack_Audit/scripts/main_message.py:78)、[main_train.py:250–256](D:/CoopTrack_Audit/scripts/main_train.py:250)。

结论：

- loss 输入是 soft probability（强制位置覆盖为 1），**不是最终 hard flags**。
- M 和 c 用真实 serializer 校准，但 E 是 soft surrogate；它不是 `len(payload)`，也不是预算 Top-K 在随机噪声下真实发包字节数的数学期望。
- 低于目标和高于目标均受平方惩罚，不是 `ReLU(E-B)` 的单边超预算罚。
- 预算超过完整包时目标降为完整包；强制成本超过名义预算时目标升为强制成本。
- `lambda_rate=1.0`，`lambda_assoc=1.0`，见 [training.json:7–8](D:/CoopTrack_Audit/configs/main_experiment/training.json:7)。总 loss 为 `L_task + lambda_assoc*L_assoc + lambda_rate*L_rate`；no_assoc 将新增 association 权重置 0，见 [main_train.py:194](D:/CoopTrack_Audit/scripts/main_train.py:194)。
- 全部 query 都强制或空集合时没有 optional action，跳过 optimizer step，见 [main_train.py:259–279](D:/CoopTrack_Audit/scripts/main_train.py:259)。

## 9. Why B1/B4 yield different update ratios

对固定的同一帧状态，令：

\[
K(B)=\left\lfloor\frac{B-M}{780}\right\rfloor,
\qquad
U=F+\min\{N-F,\max(0,K(B)-F)\}.
\]

这是 [select_updates:253–258](D:/CoopTrack_Audit/scripts/main_message.py:253) 的准确更新数量公式。

B 增大，直接使整数容量 K 增大，通常增加可选 slots；selector 从排名列表中再取更多 query。**同一帧在 N、M、F 固定时，UPDATE 数量与 score 高低无关，score 只决定具体选谁。** 这是显式 budget fill，不是仅由神经网络学习产生的更新比例变化。

budget embedding 仍能改变 ranking；不同 budget 更新不同 ID，又会改变未来各 ID 的最后 UPDATE 帧及 F。因此不同预算跑完整时序时，不能只凭局部公式保证每一帧或任意数据上的比例严格单调。这次四个完整结果的单调增加是保存记录中的实际结果。

### 对已有正式记录的只读统计

每个 budget 都是 3316 帧、44329 次 query 观测。四份通信文件的所有帧均满足上式及 `bytes=M+780U`：

| Budget | B | UPDATE数 U | 强制数 F | 可选选择数 U−F | UPDATE ratio | Forced ratio |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 3474 | 12668 | 11180 | 1488 | 0.2857722935 | 0.2522051028 |
| B2 | 6382 | 18198 | 8542 | 9656 | 0.4105213292 | 0.1926955266 |
| B3 | 9290 | 25314 | 5053 | 20261 | 0.5710482980 | 0.1139885854 |
| B4 | 12198 | 32647 | 3012 | 29635 | 0.7364704821 | 0.0679464910 |

统计来源：[communication/aptu_B1.json](D:/CoopTrack_Audit/experiments/main/communication/aptu_B1.json:1)、[B2](D:/CoopTrack_Audit/experiments/main/communication/aptu_B2.json:1)、[B3](D:/CoopTrack_Audit/experiments/main/communication/aptu_B3.json:1)、[B4](D:/CoopTrack_Audit/experiments/main/communication/aptu_B4.json:1)。query-weighted ratio 的原始计算见 [main_eval.py:454–462](D:/CoopTrack_Audit/scripts/main_eval.py:454)，不是各帧 ratio 的简单平均。

## 10. Why target budget != actual bytes

### 精确消息成本

依据 [main_message.py:11](D:/CoopTrack_Audit/scripts/main_message.py:11)、[48–85](D:/CoopTrack_Audit/scripts/main_message.py:48)：

\[
M=16+H+61N,\qquad
\mathrm{actual\ bytes}=M+780U.
\]

其中：

| 部分 | 字节数 |
|---|---:|
| prefix：MAGIC 8B、header length 4B、CRC32 4B | 16 |
| JSON header：scene、session、token、timestamp、帧索引、schema等 | H，按UTF-8实际长度 |
| predicted ID | 8N |
| UPDATE/REUSE flag | N |
| 当前 ref_pts FP32 | 12N |
| 当前 pred_boxes FP32 | 40N |
| 每 UPDATE 的三个 INT8 latent | 768U |
| 每 UPDATE 的三个 FP32 scale | 12U |

header 不含 budget 或更新数量；字段 shape 始终按总 query 数写。timestamp/scene/token 属于 header，没有另一份固定长度二进制 timestamp。除以上部分外，pack 没有再添加额外应用层 padding 或传输层开销；记录的 `len(packet)` 是该应用消息长度。

四个 budget 的 mean mandatory 都是 **1770.852232 B/frame**：

| mandatory 部分 | 平均 B/frame |
|---|---:|
| prefix | 16.000000 |
| header | 939.391134 |
| IDs | 106.945718 |
| flags | 13.368215 |
| ref_pts | 160.418577 |
| pred_boxes | 534.728589 |

### B1 为什么超过目标

| Budget | Mandatory均值 | 强制 latent+scale均值 | 可选 latent+scale均值 | 实际均值 | 超预算帧 |
|---|---:|---:|---:|---:|---:|
| B1=3474 | 1770.852232 | 2629.794934 | 350.012063 | 4750.659228 | 1813 |
| B2=6382 | 1770.852232 | 2009.276236 | 2271.314837 | 6051.443305 | 496 |
| B3=9290 | 1770.852232 | 1188.582630 | 4765.856454 | 7725.291315 | 11 |
| B4=12198 | 1770.852232 | 708.492159 | 6970.838359 | 9450.182750 | 6 |

B1 的强制更新均值中，INT8 latent 为 2589.336550B、scale 为 40.458384B；可选更新中分别为 344.627262B、5.384801B。B4 对应为强制 latent 697.592280B、scale 10.899879B；可选 latent 6863.594692B、scale 107.243667B。

这次 **所有超预算帧都精确对应 `M+780F>B`**，可选排名选择没有引入额外溢出；mandatory 单独超过预算的帧数在四个 budget 中均为 0。B1 的 88.254% UPDATE 来自 force，最大单帧溢出 15984B；B4 也有 6 帧强制溢出，最大 7260B。这由“强制先保留，即使预算不够也不撤销”直接产生，见 [main_message.py:250–254](D:/CoopTrack_Audit/scripts/main_message.py:250)。

已有记录只保存每帧 force 总数，没有把 new 与 stale 分开，也没有逐 query 的 ID/flags/logits。因此不能从这些记录继续精确拆分“新轨迹强制”与“陈旧强制”，或还原哪个可选 ID 被排名选中。

### B4 为什么没有花满

不是因为 score 低、阈值拒绝或 selector 故意留空。真实 selector 会把**能容纳的所有可选名额**填到候选列表里，直到：

1. 全部 N 个 query 已 UPDATE，没有更多候选；或
2. 余下字节放不下一个完整的 780B bundle。

统计如下：

| Budget | 低于预算帧 | 已全部 UPDATE | 尚有候选但剩余不足780B | 恰好预算帧 |
|---|---:|---:|---:|---:|
| B1 | 1481 | 115 | 1366 | 22 |
| B2 | 2820 | 836 | 1984 | 0 |
| B3 | 3305 | 1192 | 2113 | 0 |
| B4 | 3258 | 1686 | 1572 | 52 |

B4 在 1686 个“已全部 UPDATE”的低预算帧中，平均剩余 **5053.149B**；在另 1572 个低预算帧中，剩余 **1–723B**，平均 390.487B，无法再容纳一个 bundle。再加 52 个恰好预算帧和 6 个强制超预算帧，共 3316 帧。

因此 B 是可选更新的容量条件，不要求无候选时靠 padding 把包补到 B；更不是强制更新场景下不可越过的总字节硬上限。统计来源同第 9 节；实际 record 的 bytes、mandatory、updates、force、overflow 字段来自 [main_eval.py:383–387](D:/CoopTrack_Audit/scripts/main_eval.py:383)。

## 11. Association-loss gradient path

### Teacher / student 的真实计算

[main_train.py:53–63](D:/CoopTrack_Audit/scripts/main_train.py:53) 在原生 crossview alignment 前安装 hook：

- teacher 使用当前所有 query 的完整 INT8 反量化 latent 和当前 geometry，在 no_grad 下调用相同 alignment 和 `_init_inf_tracks`。
- student 使用当前实际 decoded hard UPDATE/REUSE 值，加上第 5 节的 ST surrogate。
- teacher 没有再执行一次路侧或车辆 backbone。

[main_train.py:68–84](D:/CoopTrack_Audit/scripts/main_train.py:68)：

- student：先调用原生 `STReasoner._learn_matching`，取返回的原始 affinity logits `result[2]`。
- teacher：复制**student 当前同一时刻**的车辆 `cache_query_feats/cache_motion_feats/cache_ref_pts`，使用相同 mask 和几何参数，在 no_grad 下调用相同 matching。
- 比较 `smooth_l1_loss(student_logits, teacher_logits)`，而不是对 sigmoid probability 做 loss。

原生 affinity 的输入为道路/车辆 query feature、motion feature、ref points，见 [spatial_temporal_reason.py:310–342](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:310)：

```text
concat(query_feats, motion_feats)
→ get_node_inf / get_node_veh
ref点差 → get_edge
nodes → wq / wk；edges → we
Q K^T + edge embedding
→ FFN → 原始 affinity logits [筛选车辆query数, 路侧query数]
```

matching 不直接读取 query_embeds；但 query_embeds 是 scheduler 的输入，仍可通过 UPDATE score 路径影响该 loss。teacher 的车辆历史来自此前 student 的演化历史，不是另起一条完整 INT8 teacher 轨迹。

实际 SmoothL1 默认 reduction=mean、beta=1，已经只读核对实际环境 `.../torch/nn/functional.py:2963–2969`。令 d=student−teacher：

\[
L_{\mathrm{assoc}}=\mathrm{mean}\left(
\begin{cases}\tfrac12d^2,&|d|<1\\|d|-\tfrac12,&\text{otherwise}\end{cases}
\right).
\]

当前 APTU 的 `lambda_assoc=1.0`，见 [training.json:7](D:/CoopTrack_Audit/configs/main_experiment/training.json:7)、[main_train.py:194](D:/CoopTrack_Audit/scripts/main_train.py:194)、[256](D:/CoopTrack_Audit/scripts/main_train.py:256)。空 student affinity 时，初始化的零 loss 保持为零，见 [72](D:/CoopTrack_Audit/scripts/main_train.py:72)、[245](D:/CoopTrack_Audit/scripts/main_train.py:245)。

### 能反传到哪里

```text
L_assoc
→ student affinity logits
→ 冻结的 native matching / alignment 运算
→ student ST latent
→ 非强制 query 的 noisy sigmoid probability
→ scheduler logits
→ output_head
→ Transformer relation / budget_embedding
→ input_projection / 共享 latent_projection
```

scheduler 参数组定义和前向连接见 [main_scheduler.py:12–22](D:/CoopTrack_Audit/scripts/main_scheduler.py:12)、[29–39](D:/CoopTrack_Audit/scripts/main_scheduler.py:29)。原模型参数冻结、optimizer 只收 scheduler 参数，见 [main_runtime.py:126](D:/CoopTrack_Audit/scripts/main_runtime.py:126)、[main_train.py:159–163](D:/CoopTrack_Audit/scripts/main_train.py:159)。hard sort、serializer 和 detach 的历史 cache 没有可微路径；强制 q=1、current与last相同、loss残差为零等情况下，局部梯度可为零。

这些是源码建立的梯度路径，不能据此声称每个参数组、每帧的 association 梯度都非零。已有 `task_gradient_verified=true` 和首次范数 0.008486712965062284，只验证至少一次 **native task loss** 到 scheduler 的 aggregate 非零梯度，见 [main_train.py:260–270](D:/CoopTrack_Audit/scripts/main_train.py:260)、[aptu_train.json:197–202](D:/CoopTrack_Audit/experiments/main/metrics/aptu_train.json:197)；它没有单独验证 SmoothL1 梯度。

新增 preserve loss 和原生 task loss 中的 GT association loss 是两件事；no_assoc 只将新增 preserve 的系数置零，原生 GT association loss 仍存在，见 [cooptrack.py:930–942](D:/CoopTrack_Audit/code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:930)。

## 12. One-frame worked example

下面是按真实选择器和序列化格式计算的假设 4-query 帧，不是新实验，也不是声称原始 cache 这一帧恰好只有 4 个 query。

使用现有 train manifest 中的真实 metadata 值：

```python
metadata = {
    "scene": "0000",
    "session": 0,
    "token": "000004",
    "timestamp_us": 1626155124316962.0,
    "frame_idx": 4,
    "native_frame_idx": 4,
}
```

来源：[query_cache/train/manifest.json](D:/CoopTrack_Audit/query_cache/train/manifest.json:1)。按 [pack 的 JSON header:56–63](D:/CoopTrack_Audit/scripts/main_message.py:56) 对 N=4 的真实字段 shape 编码，H=931B：

```text
mandatory M = 16 + 931 + 4×(8+1+12+40) = 1191 B
per_update c = 780 B
full-update packet = 1191 + 4×780 = 4311 B
```

给定传入 select_updates 的排名 score：

| Query | 排名 score | cache 状态 | force |
|---|---:|---|---|
| A | 0.95 | new track，无 cache | True |
| B | 0.81 | elapsed=1 | False |
| C | 0.32 | elapsed=4 | True |
| D | 0.10 | elapsed=2 | False |

这里 score 指选择器的排序输入：在 eval 中是 logits，在 train 中是已经加入噪声的 probability。没有把这四个数当作 `>=0.5` 的二分类结果。为隔离选择规则，此例保持四个给定分数不变；实际网络在两个 budget 下可以给出不同 logits。

### B1=3474

```text
force flags = [True, False, True, False]
F = 2
capacity = floor((3474−1191)/780) = 2
optional slots = max(0,2−2) = 0

A UPDATE：无cache强制
B REUSE：没有可选名额，即使score=0.81
C UPDATE：elapsed>=4强制，即使score=0.32
D REUSE：没有可选名额

packet = 1191+2×780 = 2751 B
剩余723 B，放不下第三个UPDATE
```

### B4=12198

```text
F = 2
capacity = floor((12198−1191)/780) = 14
optional slots = max(0,14−2) = 12
非强制排名列表 = [B,D]
选择该列表前12项，实际只有2项

A UPDATE：无cache强制
B UPDATE：排名候选有名额
C UPDATE：elapsed>=4强制
D UPDATE：有名额，即使score=0.10

packet = 1191+4×780 = 4311 B
剩余7887 B，所有query已经UPDATE，没有更多候选
```

所以本例中低分 D 仍在 B4 UPDATE，高分 B 仍在 B1 REUSE。两种预算的 slot 状态分别为“无可选槽位”和“所有可选候选都能发”，因此即使训练加入随机 ranking 噪声，也不会改变这个特定例子的 flags。任一帧若处在“只够发一部分可选 query”的区间，训练噪声就可能改变选中谁。

UPDATE 时三个新 latent 用 INT8+scale 发送并覆盖两端 cache；REUSE 时仍发送当前 ID/flag/geometry，从最后收到的 latent 重建。依据 [pack:64–71](D:/CoopTrack_Audit/scripts/main_message.py:64)、[ReceiverCache:167–182](D:/CoopTrack_Audit/scripts/main_message.py:167)、[TemporalCache.commit:240–246](D:/CoopTrack_Audit/scripts/main_message.py:240)。

## 13. Final concise conclusion

```text
CURRENT IMPLEMENTATION SUMMARY:

Budget affects UPDATE/REUSE by:
B/R → 64维budget token → Transformer attention → 改变query排名；
同时 B 直接进入 floor((B−mandatory)/780)，决定可选UPDATE名额。
训练中 B 还进入soft surrogate rate loss的feasible target。

A query becomes UPDATE when:
它无本session已收cache，或距离最后UPDATE的路侧帧索引差>=4；
或者它是非强制候选，并进入剩余预算名额的score降序前列。
强制UPDATE即使超过预算也保留。

A query becomes REUSE when:
它有可用cache、elapsed<4，并且没有进入该帧可选排名名额；
三个latent保持最后收到的值，当前ID/flag/ref_pts/pred_boxes仍每帧发送。

B1 sends fewer queries than B4 because:
B1可容纳的780B整数更新名额更少，B4有更多名额可按排名填充；
score不做0.5阈值过滤。完整时序还受此前UPDATE选择改变的force历史影响。
```

**当前实现是“预算条件网络排名 + 强制优先的整数预算 Top-K + 真实 UPDATE/REUSE 序列化”；训练用带噪声 sigmoid 的 soft ST surrogate，验证用 deterministic logits 排名。**

