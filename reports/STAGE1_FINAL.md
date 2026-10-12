# CoopTrack Stage 1 Example Smoke Decision

## A. Versions

| Item | Actual value |
| --- | --- |
| CoopTrack commit | 29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399 |
| Python | 3.8.20 |
| PyTorch | 1.9.1+cu111 |
| CUDA runtime | 11.1 |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| mmcv / mmdet / mmseg / mmdet3d | 1.4.0 / 2.14.0 / 0.14.1 / 0.17.1 |
| Docker image | cooptrack-stage1:29f1c52-official = sha256:22197870c646dcd6221210f814fe3494218e3e3fb22c6030faa47afc6dc649f8 |
| Docker container | cooptrack-stage1-smoke |

实际环境与 CUDA operator 证据：[environment.md](../artifacts/core/docker_provenance/docker_environment_gate.json)、[custom_ops.md](../artifacts/core/docker_provenance/docker_environment_gate.json)。Torch 自带 CUDA runtime11.1，独立编译器11.8；未用编译器版本代替 runtime 版本。

## B. Example Dataset

| Item | Actual value |
| --- | --- |
| Root | /mnt/c/Users/yujie/Desktop/CoopTrack/V2X-Seq-SPD-Example |
| Isolated converter/eval view | /mnt/d/CoopTrack_Audit/data/V2X-Seq-SPD-Example |
| Sequences | 0058 |
| Vehicle images | 250 |
| Infrastructure images | 251 |
| Vehicle velodyne | 250 |
| Infrastructure velodyne | 251 |
| Converted train infos V/I/C | 0 / 0 / 0 |
| Converted val infos V/I/C | 250 / 251 / 249 |

三侧跳帧均0、每侧13个/共39个 nuScenes 表通过实际核验；官方 split 保持原哈希。Example 下载字节0，原始数据只读绑定。[converter_report.md](../artifacts/core/stage1/converter_gate.json)。

## C. Environment

Official stack: **PASS**。40xx patch required: **YES**。

原始 GPU inverse 的 cuSOLVER 错误实际复现；单行 CPU inverse helper 与真实连续两帧验证通过，调用数为0/3。[40xx_compatibility.md](../artifacts/core/docker_provenance/docker_40xx_inverse_verification.json)。

Docker prefix 来自镜像层，host_prefix_bound=False；绑定 audit 与原始 Example。迁移后新宿主 prefix 已删除，400个本次来源可证明缓存目标、共911172109逻辑字节已清理；原 prefix 逻辑字节7326818544。Conda registry 仅删除该 prefix 一行，base/EMIFF、已有缓存、数据、官方 wheel/checkpoint、源码与报告保留。逻辑字节不表示实际磁盘回收量。

宿主 GCC11/G++11/ninja-build 已于2026-09-28安装，早于本次；本次宿主安装范围只读核对未发现新增系统 apt/pip 包，已有系统编译器/pip保留。Docker 内 APT 包另行记录。[local_cleanup.md](../artifacts/core/stage1/local_cleanup_manifest.json)、[environment.md](../artifacts/core/docker_provenance/docker_environment_gate.json)。

## D. Resources / Checkpoints

Auto-downloaded official resources: 官方 CoopTrack/MMDetection3D源码、PyPI与OpenMMLab依赖、必要 NVIDIA dev-header归档、Ninja1.11.1.3修正版、作者 Hugging Face 两份R50权重；NVCC/CUDA编译器归档复用已有缓存并核对官方hash，未重复下载。精确URL/hash和复用状态见[resource_download_log.md](../artifacts/core/stage1/resource_download_manifest.json)、[checkpoint_report.md](../artifacts/core/checkpoint_manifest.json)。独立[Docker重建配方](../docker/REBUILD.md)仅静态检查，未执行fresh build；本次实际验证的是迁移构建镜像。

| Checkpoint | Load/hash gate | Bytes | MD5 actual | Missing / unexpected / shape mismatches |
| --- | --- | --- | --- | --- |
| cooptrack_r50_inf.pth | PASS | 335324622 | 0657439da18e951724f6b4e5a623ea23 | 0 / 0 / 0 |
| cooptrack_r50_coop.pth | PASS | 367499139 | 12d4658b70f21931f3b4a793cd1bf033 | 0 / 0 / 0 |

Required upstream pretrain: **NOT_REQUIRED**。官方 test创建/加载链只加载完整 tracking checkpoint，不调用训练初始化；未下载R101/BEVFormer/vehicle训练初始化权重。[config_selection.md](../artifacts/core/stage1/stage1_final_gate_checks.json)。

## E. Inference

| Gate | Actual result | Evidence |
| --- | --- | --- |
| Infrastructure single sample / continuous2frames | PASS; ego calls0/3 | infrastructure_single_sample.md |
| Query cache | PASS;251真实生成文件，249完整cooperative依赖 | query_cache_report.md |
| Cooperative single sample | PASS;真实cached tensors与pipeline torch.equal | cooperative_single_sample.md |
| Official Example evaluation | PASS;249预测逐token匹配GT，det+track各一份metrics | example_official_inference.md |

官方 evaluation 总墙钟时间 `134.6043665289999` seconds；它包含模型加载、pipeline、forward与evaluation，未换算成模型FPS。Example指标只验证链路，不作为论文对比。

## F. RTX 4060 Ti 16GB

| Metric | Actual value |
| --- | --- |
| Peak allocated | 377686016 bytes (0.3517 GiB) |
| Peak reserved | 448790528 bytes (0.4180 GiB) |
| Mean latency | 297.68752544994186 ms |
| Median latency | 295.01910849990054 ms |
| P95 latency | 319.6420640497763 ms |
| FPS | 3.3592270905156107 |
| OOM | NO |

Latency/FPS来自batch1 FP32的5warmup+20正式cooperative帧；模型/输入dtype为float32、autocast=false，Torch官方默认TF32 matmul=true/cuDNN=true保持开启；未启用AMP/TensorRT/INT8。峰值取所有实际smoke帧与官方完整evaluation新进程峰值的最大值；allocated/reserved分别计数。OOM=NO依据这些实际执行完整成功，不从缺失字段推定。profile计时包含observer整数计数的微小开销；官方完整evaluation没有该model observer。[4060ti_example_profile.md](../artifacts/core/stage1/profile_coop.json)。

## G. Modality Audit

Inference camera input: **YES**。Inference LiDAR model input: **NO**。Depth supervision: **NO**。

LiDAR exact role: 3D GT的坐标系、图像投影/全局历史变换标定、vehicle/infra几何对齐及converter/evaluation metadata。官方converter的num_lidar_pts是占位值；所选感知pipeline不读raw point cloud，不使用DenseDepthNet/depth loss。[modality_audit.md](../artifacts/core/full_protocol.json)。

## H. Protocol Deviations / Patches

| File / scope | Change | Reason | Affects experimental protocol |
| --- | --- | --- | --- |
| projects/mmdet3d_plugin/cooptrack/modules/pf_utils.py | torch.linalg.inv(l2g_r2.cpu()).to(l2g_r2.device).type(torch.float)，单语义行 | 旧Torch/40xx cuSOLVER兼容；保留公式、原device和float32 | NO |
| scripts/generated_configs/stage1_{inf,coop}.py | Example/data_infos/cache路径；batch1；实际save/read合作字段；inference pretrained/load_from=None | 使用官方测试创建/加载路径和Example资源 | NO |
| scripts/stage1_smoke.py | 有限连续subset、真实tensor/key/finite审计、alias调用计数；wrapper原参数/返回保持 | smoke/性能与temporal补丁实际覆盖证明 | NO |
| scripts/official_eval_memory_hook/sitecustomize.py | atexit只读取新进程CUDA allocator peaks，无model hook/无重置 | 官方完整evaluation内存证据 | NO |
| scripts/run_official_example_converters.sh; tools/eval_tracking.sh | converter shell只在内存去CR；eval shell LF运行和pipefail | Windows checkout换行与退出状态兼容；官方脚本逻辑未改 | NO |
| data view; docker/*; cleanup_local_smoke.py | 原始资源只读视图、小metadata复制、Docker迁移与限定清理；保留审计输入 | 隔离运行且清理仅本次新prefix/cache | NO |

模型源码diff仅pf_utils一行；输入官方scale/pad、ResNet/BEV/query尺寸、类别/range、GT、原始数据与官方split均保持。仅运行Stage1，未训练/backward，未开始通信压缩或Stage2实验。

## I. Final State

```text
STAGE1_SMOKE_PASS

CoopTrack official Example pipeline works on RTX 4060 Ti 16GB.
Proceed to Stage 2:
query semantics + temporal redundancy + frozen-training feasibility.
```

本次执行止于Stage1；后续Stage2需要独立任务。所有最终Gate判断与源JSON SHA256：[stage1_final_gate_checks.json](../artifacts/core/stage1/stage1_final_gate_checks.json)。
