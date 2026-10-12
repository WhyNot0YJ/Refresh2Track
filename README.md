# Refresh2Track

Receiver-Cache-Aware Feature Refreshing for
Communication-Efficient Cooperative Tracking

Refresh2Track learns when to transmit roadside instance features
using a mirrored receiver cache, while keeping geometric states
updated for cooperative tracking.

This implementation builds on CoopTrack.

## Repository contents

- `scripts/v4_budget_free/`: the current receiver-cache-aware feature refreshing implementation.
- `code/CoopTrack/`: the underlying CoopTrack model and compatibility fixes.
- `code/mmdetection3d/`: the pinned OpenMMLab source dependency.
- `configs/`, `docker/`, `patches/`: experiment settings and environment recipes.
- `reports/`, `audits/`: Markdown documentation and audit source code.

## Reproduction requirements

Datasets, pretrained weights, roadside query caches, checkpoints, logs, and experiment outputs are distributed separately and are excluded from this repository.

The current scripts use the existing pinned Docker environment and experiment provenance checks. A fresh clone requires the data, weights, preprocessing files, caches, and matching manifests, together with path and container configuration. See [the Docker rebuild notes](docker/REBUILD.md) and [the V4 method documentation](reports/v4_budget_free/V4_METHOD_AND_AUDIT.md). The rebuild recipe also requires the separately supplied wheels and CUDA header archives.

The commands in the historical notes below describe the existing local experiment environment.

## Existing experiment notes

> 当前新增实验：`v4_budget_free`。**V3 Fixed/Shared 已废止（2026-10-12），仅保留思路文档**；其源码、配置、数据、checkpoint、日志和归档副本均已清理。详见 [V3 思路与废止说明](reports/v3/V3_FIXED_SHARED_FINAL.md)。V1/V2/V4 共用的数据、缓存、原始权重与 Docker 环境继续保留。

# CoopTrack APTU 主实验

继承官方 CoopTrack `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`、已有 FULL_DATA_READY、Stage1/2/3 与 R50 car-only 对齐结果。正式实验使用 10Hz、ResNet50、46 个 train scene / 7362 个协同训练样本，以及 21 个 val scene / 3316 个验证样本。

从 PowerShell 在项目根目录运行：

```powershell
python scripts/run_main.py --resume
```

入口复用固定 Docker 镜像，无需安装或重建环境。执行顺序为完整 7445 帧路侧 train cache、基于完整 train 锁定预算、APTU / MLP / 去关联保持损失的三个 12-epoch 训练、Periodic-2 / Periodic-4 / Delta 与学习调度器的完整 val，以及最终表格和通信性能曲线。已有 FP32 / FP16 / INT8 / Top-K / Random 结果直接复用。

中断后执行同一命令。缓存生成重放尚未完成的完整 scene；训练从每个方法的 `latest.pth` 恢复 epoch、优化器和 RNG，不在 val 上选 epoch。预算写入后锁定，禁止按 val 修改。

| 路径 | 用途 |
|---|---|
| `code/CoopTrack` | 官方源码及兼容补丁 |
| `code/mmdetection3d` | 现有 Docker 环境依赖的原生算子和源码 |
| `configs/main_experiment` | 正式配置、锁定预算与执行协议 |
| `checkpoints` | 原始官方 R50 权重 |
| `datasets`, `data`, `data_infos` | 已准备的数据与官方 infos |
| `query_cache/train`, `query_cache/val` | 正式路侧查询缓存 |
| `experiments/main/checkpoints` | 三种调度器的 epoch / latest 权重 |
| `experiments/main/predictions` | 新条件完整 val 预测 |
| `experiments/main/communication` | 逐帧真实消息字节数与延迟 |
| `experiments/main/metrics` | 训练、评测和执行状态 |
| `artifacts/core` | 数据身份、checkpoint 身份及继承的最终证据 |
| `logs/main` | `train.log`, `eval.log`, `resource.csv` |
| `reports` | 精简说明、各阶段最终报告和主实验最终报告 |
| `docker`, `patches` | 环境复现与 40xx 兼容补丁 |

单阶段入口在 `scripts/main_train.py` 和 `scripts/main_eval.py`，仅在固定 Docker 环境中运行。`main_scheduler.py` 实现集合注意力与参数匹配 MLP；`main_message.py` 实现按预测轨迹身份的发送端镜像和接收端缓存。所有候选保留，每帧传输必要状态；三个 latent 字段一起 UPDATE 或 REUSE，新轨迹与陈旧 4 帧的轨迹强制更新。字节数取真实序列化消息长度。

最终证据见 [MAIN_EXPERIMENT_FINAL](reports/MAIN_EXPERIMENT_FINAL.md)，执行状态见 `experiments/main/metrics/run_state.json`。在正式运行结束前，缓存/训练/评价状态以实际产物为准。
