# Stage2 数据阶段最终报告

范围：**DATA_ONLY**；状态：**FULL_DATA_READY / PASS**。
完成时间：`2026-10-04T02:08:52.206284+08:00`。本次只完成 S2-0、D0–D5 和 D6.1；D6.2 query 缓存为 `READY_PENDING_INFERENCE`。

## A. 工程验收表

| Gate | 状态 | 证据/范围 |
|---|---|---|
| Stage1继承 | PASS | [JSON](../artifacts/core/stage2/manifests/stage2_context.json) |
| 本地归档/分卷/载荷CRC | PASS | [JSON](../artifacts/core/stage2/data/archive_integrity.json)；两组 Headers Error 保留；实际逐文件 CRC/大小/SHA 封口通过 |
| 完整原始目录与版本 | PASS | [JSON](../artifacts/core/stage2/data/raw_integrity.json) |
| 官方prepared视图与标签处理 | PASS | [JSON](../artifacts/core/stage2/data/prepared_completion.json) |
| 三侧train/val infos与nuScenes表 | PASS | [JSON](../artifacts/core/stage2/data/full_trainval_integrity.json) |
| FULL_DATA_READY | PASS | [JSON](../artifacts/core/full_data_ready.json) |
| 完整val路侧缓存覆盖 | NOT_RUN_STAGE_BOUNDARY | 计划 READY_PENDING_INFERENCE；尚未生成/加载实际 query |
| 原生完整val执行 | NOT_RUN_STAGE_BOUNDARY |  |
| R50数值参考对齐 | NOT_RUN_STAGE_BOUNDARY |  |
| 通信边界和字段 | NOT_RUN_STAGE_BOUNDARY | Stage2 接口审计未执行 |
| FP32真实消息无损回归 | NOT_RUN_STAGE_BOUNDARY |  |
| Example免训练扫描 | NOT_RUN_STAGE_BOUNDARY |  |
| 预测ID时序诊断 | NOT_RUN_STAGE_BOUNDARY |  |
| 真实train片段 | PASS | [JSON](../artifacts/core/stage2/manifests/train_smoke_manifest.json)；仅完整官方 train infos 派生的32帧 manifest；未执行梯度探针 |
| 任务loss到接口梯度 | NOT_RUN_STAGE_BOUNDARY |  |
| 16GB短步冻结探针 | NOT_RUN_STAGE_BOUNDARY |  |
| 完整val数值复现 | NOT_RUN_STAGE_BOUNDARY |  |
| 正式通信创新训练 | NOT_RUN_STAGE_BOUNDARY |  |
| 5090兼容与训练 | NOT_RUN_STAGE_BOUNDARY |  |

## B0. 完整数据处理账本

| 侧 | raw metadata | 官方选集 | 官方过滤 | train infos | val infos | GT sample | 必需依赖缺失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vehicle-side | 12252 | 10761 | 335 | 7110 | 3316 | 10426 | 0 |
| infrastructure-side | 11275 | 10761 | 0 | 7445 | 3316 | 10761 | 0 |
| cooperative | 10761 | 10761 | 83 | 7362 | 3316 | 10678 | 0 |

三侧各46个 train scene、21个 val scene；train/val token 和 scene 无交集。各侧13张 GT 表，共39张；GT引用、annotation token、类别、visibility、w/l/h与全局中心已对账。全量 GT yaw 数值核验不在本轮报告范围。

各层数量差异来自官方 test 选集、coop 配对及既有 invalid 清单，逐记录原因、split 和源码来源在 [排除账本](../artifacts/core/stage2/data/excluded_records.jsonl)；未新增任意删帧。官方选集为 `--sequences all` 的67个非test scene，`--freq 10 --update-label`；转换保留 `--forecasting` 与 `v1.0-trainval`。

归档总存储 `26789363360` B；解压 raw 新增逻辑字节 `28639083436` B、`187713` 个文件。8个物理归档已 SHA256 锁定，5组 SPD 实际载荷 CRC/大小/SHA 通过；无可用官方 hash，`official_hash=null`。额外图像包因兼容性未证明排除，仅登记/哈希，未混用。两组点云 `Headers Error` 警告未隐藏，封口状态 `HEADER_WARNING_PAYLOAD_VERIFIED`，根因未证明。详见 [归档与空间](../artifacts/core/stage2/data/archive_integrity.json)。

split 转换前锁定 SHA256 `0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd`；转换后 CPU pipeline SHA256 `0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd`，一致。必需依赖缺失0；未生成且当前 camera-only pipeline 不读取的 metadata `.bin` 路径逐条单列，不宣称所有 GT filename 文件都存在。原始 `.pcd` 载荷与必要图像已完整核验，全部 `23527` 张 JPEG 实际解码通过。详见 [三侧转换](../artifacts/core/stage2/data/full_trainval_integrity.json)。

原始标签只读使用；prepared 副本应用官方修正与首 track_id 去重，摘要如下：

| 侧 | label文件 | raw annotation | prepared annotation | 匹配官方修正 | 去重删除 |
|---|---:|---:|---:|---:|---:|
| vehicle-side | 10761 | 127440 | 127431 | 18 | 0 |
| infrastructure-side | 10761 | 176618 | 176618 | 132 | 0 |
| cooperative | 10761 | 250535 | 250535 | 0 | 0 |

官方修正表 SHA256 `a97d52fa9222d62eb396199e9bfaa4dc84da2d7dde734eed3feba5b0ea911667`；逐文件改动 ledger SHA256 `c3ec946fa31e7ae9a62917190623c0334a93e565fda4d8074a363f85d34c6066`。详见 [官方准备](../artifacts/core/stage2/data/prepared_integrity.json)。

mini-train 由完整官方 train infos 派生：scene `0000`，`32` 帧连续协同样本；沿用全量历史/未来监督，未对截断 raw 重转。0058仍是val，不进入完整train或片段；它此前作为Stage1开发诊断样本使用，不能单独称为独立泛化测试。

| 缓存计划用途 | 协同引用 | unique路帧 | 原生历史prefix帧 | scene | 实际生成/覆盖 |
|---|---:|---:|---:|---:|---|
| train_smoke | 32 | 32 | 32 | 1 | NOT_RUN_STAGE_BOUNDARY |
| val | 3316 | 3316 | 3316 | 21 | NOT_RUN_STAGE_BOUNDARY |

缓存身份包含真实 inf/coop checkpoint、三侧配置、source/image/data hash 与时间范围；只生成计划 manifest，尚未推理或检查实际消息。初始协同 CPU 数据形状检查仅使用显式空路侧记录，不是 query 覆盖或梯度证据。后续必须按原生scene顺序生成并从scene起点恢复，实际非空路侧消息通过后才能做梯度探针。详见 [D6计划](../artifacts/core/val_cache_manifest.json)。

## B. 通信结果表

| 方法 | B/frame | 压缩比 | mAP/AMOTA | 变化百分点 | 样本数 | latency | 原因 |
|---|---|---|---|---|---|---|---|
| 未执行 | null | null | null | null | null | null | NOT_RUN_STAGE_BOUNDARY |

## C. 研究结论范围

数据准备完成；完整val精度、通信收益、时序收益、loss到接口梯度、16GB训练可行性及5090兼容性均未验证。本轮不对这些问题作研究结论。

## D. 停止与恢复

已停在 `FULL_DATA_READY`。Stage1保护、raw/prepared只读服务与阶段写容器停止均通过；没有训练、模型加载或GPU推理。用户后续授权继续Stage2后，复用本轮产物与统一CLI，先实现/执行待推理缓存和原生完整val，再进入通信/训练诊断；当前 `run --resume` 保留边界提示，后续阶段尚未实现。

历史数据阶段入口已在一次性精简中移除；正式数据直接复用，不再重新准备。当前主实验恢复入口为项目根目录下的 `python scripts/run_main.py --resume`；模型仍在继承 Docker 镜像内运行。

最终 context 曾因成功的CPU导入输出诊断前缀而使编排器严格JSON解析失败；实际Docker诊断退出0，版本匹配，未修改框架或数据。导入诊断现由编排helper捕获到stderr，最终继承门禁通过后才生成本报告；[原始诊断证据](../artifacts/core/stage2/data/context_runtime_stdout_diagnostic.json)及历史失败保留。

完整门禁见 [stage2_gate_checks.json](../artifacts/core/stage2/stage2_gate_checks.json)，数据报告见 [FULL_DATA_READY.md](../artifacts/core/FULL_DATA_READY.md)，路径见 [paths.json](../artifacts/core/paths.json)，命令/UTC/退出码/hash见 [执行ledger](../artifacts/core/stage2/manifests/prepare_data_execution.json)。
