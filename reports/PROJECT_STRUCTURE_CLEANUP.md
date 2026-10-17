# PROJECT_STRUCTURE_CLEANUP

项目：`D:\CoopTrack_Audit`。审计与执行日期：2026-10-09，时间均为北京时间。

## 1. 本次实际结果

**FINAL = COMPLETE。** 基于此前审核的精确白名单，Codex 已直接执行 `scripts/cleanup_reviewed.ps1 -Apply`，真实删除 **116 个文件、232 个空目录**，移除文件的逻辑大小为 **2,056,555,438 B = 2.056555438 GB**。执行完成时间为 **02:44:50**；独立复查全部 348 个目标，残留文件和目录均为 0。

Dry-Run 于 02:43:11 全部通过。本次 Apply 未收到 `blocked by policy`，退出码为 0。此前被策略阻止的尝试仍保留在旧机器证据中，不能把那个历史状态当作此次执行状态。

删除后，973 个受保护文件的 SHA256 与本次操作前基线相同；10 个 checkpoint 和 17 个正式源码/配置绑定符合既有证据；18 个关键 Python 模块再次导入正常。02:47:21–02:47:56 的独立数据及结果检查也通过，详见第 5 节。

本次新增项目文件只有清理脚本；文档只更新本报告。读取本报告时其长度为 0 B，因此以下依据未改动的审计 JSON 与旧清单恢复已有清理结论，并记录此次真实执行结果。未重新扫描全项目、制定新清理方案、训练模型或运行 official val。

## 2. 采用的既有证据

精确清单仍来自 [cleanup_evidence.json](../audits/project_structure_cleanup/cleanup_evidence.json) 的 `merged_files`、`deleted_files` 和 `removed_directories`；文件元数据依据 [inventory_after.csv.gz](../audits/project_structure_cleanup/inventory_after.csv.gz)。这两个文件没有回写，脚本固定验证以下 SHA256：

| 文件 | SHA256 |
|---|---|
| `audits/project_structure_cleanup/cleanup_evidence.json` | `d4bd8a1f1c5b2d563a32d82ad8630b39306615af1562cf02df07e873c286652b` |
| `audits/project_structure_cleanup/inventory_after.csv.gz` | `00d68d4912fb266af358e1399526daa11d8fe97734c1d41c9d87ca840a98ad4e` |
| `scripts/cleanup_reviewed.ps1`，实际 Dry-Run/Apply 版本 | `eb9cc162da3422727c2a0a73a908d537c4b3e8a8bb9474bdbef1ca915837ec08` |

旧 JSON 的 `actual_execution.status=BLOCKED_BY_POLICY`、`status=PARTIAL` 及旧 CSV 都是此前尝试的快照；本次结果以本报告及实际执行输出为准。此前扫描记录为 233,527 文件/1,042 目录/60,549,420,990 B，先前拒绝删除后的清单为 233,535 文件/1,047 目录/60,927,726,578 B。它们不是当前全项目统计。并行实验持续产生结果，本次没有用旧总量减去清理量来伪造当前目录总量。

## 3. 审核清单与实际清理

| 类型 | 计划文件数 | 实际删除数 | 移除逻辑字节 |
|---|---:|---:|---:|
| Torch 重复下载分块 | 61 | 61 | 2,041,339,783 |
| 与现用运行时 SHA256 相同的 build `.so` 副本 | 11 | 11 | 5,992,792 |
| 无用编译中间文件 | 36 | 36 | 9,019,590 |
| Python 字节码缓存 | 7 | 7 | 202,249 |
| 下载探针 | 1 | 1 | 1,024 |
| **合计** | **116** | **116** | **2,056,555,438** |

232 个目录包括 191 个无内容的 evaluator 输出目录、40 个删除已审核生成文件后变空的 build 目录，以及 1 个字节码目录，全部实际移除。每个目录删除时均检查为空；没有递归删除非空目录。

释放空间按实际已删除文件的长度求和，单位 GB 为十进制；没有把并行实验写入期间全盘空闲空间的变化归因给清理，也没有声称测得 NTFS 分配空间的精确净增量。

没有新增删除目标或移动路径。现用 `.so`、完整 wheel 的既有路径保留，源码、配置及引用路径无需调整。原计划中的两个不确定项 `CoopTrack_V2_Periodic_Sweep_and_Adaptive_Refresh_Codex_Prompt.md` 和 `artifacts/core/stage3/state.json` 继续保留；下载 headers/manifest/log 等非 payload 文件也保留。

## 4. 脚本与安全检查

脚本：[scripts/cleanup_reviewed.ps1](../scripts/cleanup_reviewed.ps1)。默认模式为 Dry-Run；`-Apply` 才执行文件删除。`-ResultPath` 可选，仅允许在项目外保存机器结果，默认不产生额外项目报告。

脚本的检查及本次实际表现如下：

- 固定项目根目录和两份既有证据 SHA256；逐项检查精确白名单的路径、类型、数量和大小，拒绝重复项、越界路径及目标/祖先上的 junction、符号链接或其他 reparse point。
- 校验全部 11 个 build `.so` 与保留运行时文件的 SHA256 相等；按数值偏移检查 61 个分块的连续范围、长度，并流式计算拼接 SHA256，确认等于保留的完整 wheel；下载探针也核验既有 SHA256。
- 预检目录只读取直接子项，要求每个子项也已获审核；没有跟随链接或扩展白名单。不存在的目标明确记录为 `MISSING` 并停止，不推断替代路径。
- 检查 426 个 Windows 进程的命令行、所有运行容器的项目挂载；对相关容器读取 `/proc` 的命令行、cwd、fd 和 maps，并对每个目标文件作只读独占打开测试。目标引用与检查错误均为 0。
- Apply 在删除前再次检查进程；每个文件再次核验路径、大小、mtime、SHA256 和独占打开。文件使用 `Remove-Item -LiteralPath`，没有 `-Recurse`；目录使用 `[IO.Directory]::Delete(path, false)`，目录变为非空即失败。
- 任何预检或执行错误都会停止，不追加其他路径。Apply 后检查全部目标消失、保护文件哈希一致及模块导入正常。本次没有错误，也没有缺失目标。

进程快照显示 2×2 消融方法 D 与 `budget_free/run.py --resume` 正在运行。它们的输入、日志、预测及中间产物不在白名单中；没有中断这些进程。运行时实际使用的 CUDA `.so` 均保留。唯一停止的是本次自身因重复路径解析而耗时的只读 Dry-Run，随后完成了优化后的完整 Dry-Run。

预检开发过程中发现文件共享访问和 PowerShell 5.1 的 Docker JSON 数组解析问题，均在实际删除前修正。共享读哈希检查文件读取期间大小/mtime不变；删除目标仍须通过独占打开。最终脚本通过语法检查及完整 Dry-Run 后才执行 Apply。

## 5. 删除后真实验证

| 检查 | 结果 | 实际证据与范围 |
|---|---|---|
| 删除白名单 | PASS | 独立按旧 JSON 精确路径检查：116 个文件、232 个目录全部不存在；残留 0 |
| 受保护文件 | PASS | Apply 内 973 个既有保护文件逐项前后 SHA256 相等，包括原生源码、配置、正式结果元数据及 Bug 审计证据；没有将运行中的消融输出当作静态文件冻结，Git 历史没有纳入这 973 项内容哈希 |
| V1/V2 checkpoint | PASS | 10 个正式/官方 checkpoint 路径 SHA256 与旧证据一致，并通过删除后保护哈希检查；具体值见第 6 节 |
| 正式源码与 manifest | PASS | 17 个冻结源码/配置绑定与既有 SHA256 相同；`FINAL_PROTOCOL_MANIFEST.json` 的操作前后 SHA256 相同，为 `dd753608c90b75f8b7a316bb6207403c6d27bbd2f0eebb2a9878760c69968923` |
| 完整 Torch wheel 与运行时扩展 | PASS | 完整 wheel 保留，SHA256 为 `2546dcaae81ac74f3f88dd4b29f5ead28852a5e8e66ca85b4f46f2eea306033c`；11 个 canonical runtime `.so` 保留且哈希相同 |
| Python imports | PASS | Apply 前及删除后均实际导入 18 个模块；没有模型 forward、optimizer 或 evaluator 调用 |
| 数据与 query cache | PASS，元数据核验 | 删除后按旧 CSV 中 230,812 个已知精确文件路径逐项 stat：全部存在，大小和 Windows mtime 一致；未遍历目录或对全部数据重新计算内容哈希 |
| 既有结果可读性 | PASS，在明确范围内 | 删除后重新解析相同 97 个 metrics/communication/审计 JSON 或 CSV，失败 0；相同 3 个预测 gzip 完整读取到 EOF，CRC 校验通过 |
| 已知链接 | PASS | Apply 后对旧证据中的 27 个精确链接路径进行 lstat/readlink，全通过且未跟随；有历史 target 记录的 3 个链接与记录相同 |
| 官方 Git 身份 | PASS | 原生 CoopTrack HEAD 仍为 `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`；mmdetection3d HEAD 仍为 `f1107977dfd26155fc1f83779ee6535d2468f449`；未改写 Git 历史或恢复继承 patch |

删除后的数据核验于 **02:47:21–02:47:56** 完成，严格复用旧路径清单：

| 已知保护范围 | 文件数 | 保留逻辑字节 | 存在/大小/mtime |
|---|---:|---:|---|
| `datasets/` | 220,038 | 29,409,224,925 | PASS |
| `data_infos/` | 12 | 2,201,680,039 | PASS |
| `query_cache/` | 10,762 | 6,720,146,099 | PASS |
| **合计** | **230,812** | **38,331,051,063** | **失败 0** |

18 个导入模块为 `torch`、`numpy`、`mmcv`、`mmdet`、`mmdet3d`、`mmdet3d.ops`、`projects.mmdet3d_plugin`、`main_runtime`、`main_message`、`main_scheduler`、`main_eval`、`main_train`、`v2_runtime`、`v2_message`、`v2_scheduler`、`v2_eval`、`v2_train`、`v2_run`；均在继承容器中以关闭字节码写入的模式导入。

gzip 抽查为 `experiments/v2/predictions/official_val/mlp_B1_seed0/` 的 `predictions.pkl.gz`、`results_nusc.json.gz` 和 `results_nusc_det.json.gz`；没有 unpickle 或重新评估。其余 45 个 V2 gzip 未作全量 CRC 检查，旧清单中的 V1 正式预测没有 gzip。本次结果可读性检查不等于重新核算全部指标或重新认证历史预测身份。

并行任务造成的既有漂移单独保留：旧清单中的 `experiments/v2/ablations/refresh_2x2/metrics/independent_audit.json` 从 5,258 B 更新为 11,441 B，前后验证均可解析；两个消融报告/保护辅助脚本相对旧审计也已有修改，本次使用其当前操作前哈希作保护，并确认清理期间不变，没有恢复旧版本。新增并行输出不纳入本次旧清单统计。

## 6. 保留 checkpoint SHA256

以下 10 个精确路径均实际核验，V1 每种模型的 `latest.pth` 与 `epoch_12.pth` 两个文件分别检查，表中合并展示相同哈希：

| checkpoint 路径 | SHA256 |
|---|---|
| `experiments/main/checkpoints/aptu/latest.pth`、`epoch_12.pth` | `43a7203a3049607893fbbb92e8cef6a01eac1e01608937eebada84a65189e951` |
| `experiments/main/checkpoints/mlp/latest.pth`、`epoch_12.pth` | `70fd29401739ffaecf4faca42c0e815fada6a1e8bdc94bdc87d028c9b6d4dd0d` |
| `experiments/main/checkpoints/no_assoc/latest.pth`、`epoch_12.pth` | `c05f7e3d1bc46d5de0955aa5e31d15c8f00c02d21b05d061f4a4ef0a0ad761d9` |
| `experiments/v2/checkpoints/mlp/final.pth` | `d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c` |
| `experiments/v2/checkpoints/transformer/final.pth` | `58bda699962f479a7556e75c6ad3b0389d6a909ca7b798e5eece2ea7f2fc65a3` |
| `checkpoints/cooptrack_r50_inf.pth` | `a9ae9cecef72aa772666a1fb8aad4eb009d1f20ce1cb5b812de16e2945630f79` |
| `checkpoints/cooptrack_r50_coop.pth` | `de543ce784175dd62f62d4e9d6b17c5ae7757f00548efbe49f8671e98ef0c4e2` |

## 7. 执行记录与复核方式

实际执行使用 PowerShell 直接调用脚本，没有其他 Shell、Python 或 Docker 删除：

```powershell
& .\scripts\cleanup_reviewed.ps1 -ResultPath 'C:\Users\yujie\AppData\Local\Temp\cooptrack_structure_cleanup\dry_run_current.json'
& .\scripts\cleanup_reviewed.ps1 -Apply -ResultPath 'C:\Users\yujie\AppData\Local\Temp\cooptrack_structure_cleanup\apply_current.json'
```

机器结果只保存在项目外的临时目录，本报告记录其完成状态和内容哈希，未新增项目内重复报告：

| TEMP 文件 | SHA256 |
|---|---|
| `dry_run_current.json` | `03e66cc30345f1038d80ddcb74cd288a1004d0ce1e9fc81bdda2a077ff2e58b9` |
| `apply_current.json` | `1028cce1ab048ff30917cb872eea4b68dfd0eebcaaa1db28bd8d092333e74d40` |
| `protected_assets_current.json`，保留首次验证并含删除后复核 | `8a5700a02f8dc732c1dc74170223535c2abc3c1be94c0f61941933732bd5396e` |

清理已经完成。再次运行本脚本将明确记录已删除目标不存在并停止，不会猜测其他路径或扩大删除范围。检查容器进程及模块导入使用的 Docker/Python 都是只读验证，不承担删除操作。

## 8. 完成范围

本次 COMPLETE 表示审核清单已真实清理、规定的保留与可读性检查已完成。数据/cache 采用完整旧路径元数据核验，预测 gzip 为明确抽样；没有宣称全项目重新扫描、全部数据内容哈希、全部预测重评或全局无 Bug。

Bug 审计报告及证据保留；[V2_BUG_AUDIT.md](v2/V2_BUG_AUDIT.md) 的历史预测 SHA 绑定缺失 E-01 没有补造，也不因清理完成而变成通过。现有及并行实验结果没有被重新生成。

```makefile
PROJECT_CLEANUP

SCRIPT_GENERATED=PASS
DRY_RUN=PASS
EXECUTION=PASS

FILES_DELETED=116
DIRECTORIES_REMOVED=232
SPACE_RECLAIMED_GB=2.056555438

V1_PRESERVED=PASS
V2_PRESERVED=PASS
CHECKPOINT_HASHES=PASS
DATA_PRESERVED=PASS
SOURCE_MANIFEST=PASS
RUNTIME_IMPORTS=PASS

FINAL=COMPLETE
```
