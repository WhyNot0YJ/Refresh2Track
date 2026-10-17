# Codex 执行任务：CoopTrack 原生 FULL_FP32 vs V2 MLP B1，三推理 seed 稳健性实验

## 目标

项目根目录：`D:\CoopTrack_Audit`（容器对应 `/mnt/d/CoopTrack_Audit`）。

**请实际执行实验，不只写计划。** 要判断已有 MLP B1 的 AMOTA 高于 FULL_FP32 的现象是否在不同**推理 seed** 下仍然成立。

实验严格限定两种方法 × 三种推理 seed，不训练模型、不改模型结构、不跑 H8/H1、不自动启动 V3：

| 方法 | 模型与刷新协议 | 推理 seeds |
|---|---|---|
| `FULL_FP32` | 冻结官方 R50 coop checkpoint；全部当前路侧三组高维特征，每帧 FP32 输入；每帧当前几何 | `0,2025,2026` |
| `V2_MLP_B1` | 冻结同一官方 R50 coop checkpoint + **已经训练好的** `experiments/v2/checkpoints/mlp/final.pth`；原 V2 INT8/cache/codec，**名义预算 B1=3474 B/frame**；每帧当前几何 | `0,2025,2026` |

历史 seed0 参考（**仅作审查锚点，不得在未核验同协议前当作本次配对结果**）：
- 原 FULL_FP32：mAP `0.3623447993`，AMOTA `0.3568982508`。
- V2 MLP B1：mean `2630.828106` B/frame，mAP `0.362306438354`，AMOTA `0.369315594181`。
- 项目已有的 `artifacts/ids_diagnosis` 三 seed 原生结果属于**历史诊断**。只有证明其 checkpoint、val 顺序、推理初始化/RNG 重置位置、运行设置、实际预测身份与本轮完全相容，才可复用；否则重新推理，绝不能把不同运行协议的 seed0/2025/2026 拼成一个统计表。

## 1. 执行前只读核查

1. 检查 `D:\CoopTrack_Audit` 的真实目录与当前是否有 GPU/评估/训练任务；避免干扰正在运行的进程。不下载/转换数据。
2. 定位并核验原生 CoopTrack `cooptrack_r50_coop.pth`、`cooptrack_r50_inf.pth`、V2 MLP final checkpoint、官方 21 val scenes / 3316 cooperative tokens、已有 `query_cache/val`、原始 V2 执行入口与官方 evaluator。检查原有 final checkpoint SHA、源码/配置身份和必要路径。
3. 确认 FULL_FP32 处理输入的实际路径，不能将其偷换为 INT8、或者将原生 FULL_FP32 改成 vehicle-only。MLP B1 必须使用老的 V2 frozen checkpoint / 序列化 / receiver cache / 0.5 门控及预算输入，不允许重新训练或改阈值。
4. 原生的 `torch.randperm` 会随机选用于补足跟踪 slots 的**已学习检测 query 模板**；seed 也可能作用于 Python/NumPy/Torch 等其他随机操作，所以不能说「seed 只影响 randperm」。三次属于推理 seed 检查，**不属于三次独立训练**。
5. 不触碰已经退役的 H 专用目录、数据或代码，也不要再生成它们。

## 2. 公平的三 seed 实验协议（先冻结再运行）

- 推理 seed 分别为 `0,2025,2026`。
- **一个 case 一个全新独立 Python 进程**，独立模型/接收端缓存/本车时序状态；禁止多个 seed 共用未重置的模型状态，或把不同 case 的 prediction 混写。每个 case 逐 scene/session 正确清零缓存。
- 对每一个 case，在同一个明确定义且可审计的初始化点设置 Python、NumPy、Torch CPU、Torch CUDA RNG；`PYTHONHASHSEED` 在 Python 进程启动前设置。参照 V2 冻结设置，在模型构建/权重加载/DataLoader iterator 创建后 reset，同法作用于 FULL_FP32 和 MLP B1；不能一边按官方 `tools/test.py` 的「模型构建前设 seed」，另一边按 V2 的「构建后 reset」。
- 保持相同官方 coop 权重、车端模型、官方 val 顺序、R50 FP32 CUDA 环境、cuDNN/TF32 设置、单卡、car-only official evaluator、路侧**预生成的同一份 query cache**、时间/标定。除 Full FP32 对比方法本身的发送/量化/刷新机制外，不得引入其他未报告变化。
- 这次只改变**本车 CoopTrack 推理随机状态**；既有路侧 query cache 不因不同 seed 重生成。因此结论仅针对「固定路侧预测输入时，不同本车推理随机 seed 的稳健性」，不等于路侧与车端全链路一起变 seed。
- 每个 case 都要实际完成 **3316/3316** 帧，并由原生官方 car-only evaluator 完整评估；不接受 1816 帧等部分结果或用片段指标冒充官方结果。
- 保留 V1、V2、P1/P2/P4、Periodics、2×2、所有旧模型/报告不变；新产物独立写 `experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/`。
- 在首次推理前把新实验 identity/运行命令/seed 初始化位置/原生工作树哈希/配置与 metric 口径写入新 manifest。每个 run 完成 raw prediction 即刻写 SHA256，**评估前**记录 raw input SHA；结果与之绑定。

## 3. 执行策略

优先沿用现有经验证的 V2/main runtime、native eval/codec，不另造跟踪网络；必要的独立 wrapper 只能放在新实验目录或单独的新脚本，不改写冻结旧入口。

正式最多 6 个 complete cases（2 方法 × 3 seed），单个 case 3316 帧，**不复用情况下总计 19,896 帧模型前向**。可跳过某个 case 的运行，仅当该 case 有已经存在且**逐项证明可比**的完整预测、原始运行 manifest 和相同 RNG 时序、完整 3316 帧与原生指标；不能只凭方法名相同就跳过。

如果某 case 初始化、数据或 seed 对齐出错，停止该 case，记录具体阻塞；不得继续用不一致结果计算所谓三 seed 结论。不要动态改变 torch deterministic、batch、num_workers 或 CUDA 设置来凑结果；若不可避免，必须所有 case 同步一致，并另设可追踪的新运行身份。

## 4. 输出指标与分析

每个 case 输出：
- `seed`、方法、checkpoint SHA、official sample 数、评估来源与 prediction SHA。
- mAP、AMOTA、AMOTP、MOTA、IDS，以及可提取的 FP/FN、召回率。官方 IDS 发生在各自 evaluator 的 best-MOTA operating point，不把 IDS 差异直接等同于真实身份稳定性差异。
- 真实 mean/P95 B/frame 与 nominal 10Hz B/s；FULL_FP32 与 V2 INT8 不同的**应用通信表示**必须说明；不要只用 B1 名义 3474 B 代替真实通信量。
- 可选补充：仅利用已生成 predictions，在**预先确定的共同 score 阈值**下重算 FP/FN/IDS 作为解释性诊断；不得覆盖官方指标，不能从验证集上挑有利的阈值。

最终要给三行**同 seed 配对**：

```
seed | FULL_FP32 AMOTA | MLP_B1 AMOTA | Delta(MLP−FULL) | FULL_FP32 IDS | MLP IDS | FULL_BYTES | MLP_BYTES
0    | ...
2025 | ...
2026 | ...
```

另汇总：两方法 AMOTA 的均值/标准差、每 seed 配对差的均值/标准差、正差值次数 `wins/3`；mAP/MOTA 同样给差值。**只有 3 个推理 seed，不做显著性结论，不能宣称训练 seed 稳健性或泛化到新数据集。**

结论分级：
- `CONSISTENT_POSITIVE_INFERENCE_SEEDS`：三次配对 ΔAMOTA 均 > 0，表示这三个推理 seed 下数值一致为正，但**不等于统计显著**。
- `MIXED_INFERENCE_SEEDS`：有正有负，说明 seed 稳健性尚不足以支撑明确优越性。
- `CONSISTENT_NON_POSITIVE`：三次 ΔAMOTA 均 <= 0，旧单 seed 优势未在这组复核中出现。
- `INCOMPLETE`：有 case 未完成或不可比，严禁硬算均值。

必要时指出：即便多个 seed 下 MLP > FULL_FP32，也不能自动认定改善来自“去除最新高维噪声”，因为 FULL_FP32 vs MLP B1 同时改变了量化、缓存和刷新机制。若要隔离这种原因，需要另做**同协议全更新 INT8**对照，那是另一项实验，本任务**不要自动扩大范围**。

## 5. 最终检查、报告和终端输出

- 独立生成 `reports/v2/MLP_B1_VS_FULL_FP32_3SEED.md` 以及 `experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3/metrics/summary.csv`。
- 保留六份原始方法/seed 结果、原始预测文件、评估指标、run manifest 和 SHA，不覆盖旧结果。
- 核对原 V2 MLP 和 official 权重 SHA 与执行前一致，旧训练/评估结果与关键配置不变，query cache 未改变；明确标记核验范围。
- 如同 seed0 新评估与历史值不符，列出差异，检查 seed reset、完整输入、原生工作树、数值环境，不能偷偷回填历史值或挑选有利运行。
- 本任务**不**训练 V3、MLP、Transformer 或 H 模型，不创建任何新 checkpoint；不修改已有模型预测阈值、预算、loss 或 train split。

终端摘要：

```
THREE_SEED_INFERENCE_TEST
STATUS=PASS/PARTIAL/BLOCKED
CHECKPOINTS_UNCHANGED=PASS/FAIL
SEED_PROTOCOL_IDENTICAL=PASS/FAIL
FULL_FP32: seed0=..., seed2025=..., seed2026=...
MLP_B1: seed0=..., seed2025=..., seed2026=...
DELTA_AMOTA_BY_SEED=...
MEAN_DELTA_AMOTA=...
WINS_OVER_THREE=...
COMMUNICATION_COMPARISON=...
VERDICT=CONSISTENT_POSITIVE_INFERENCE_SEEDS/MIXED_INFERENCE_SEEDS/CONSISTENT_NON_POSITIVE/INCOMPLETE
REPORT=reports/v2/MLP_B1_VS_FULL_FP32_3SEED.md
```
