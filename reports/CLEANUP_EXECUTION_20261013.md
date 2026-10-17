# 已确认清理项执行记录

2026-10-13，Asia/Shanghai。按用户“把确认可清理的清理了”的指令，执行前复核 [审计清单](../audits/cleanup_review_20261012/safe_candidates.csv) 的 SHA256、逐文件大小/mtime/内容、链接目标和当前使用情况，只处理已确认可清理项。

**STATUS=COMPLETE；删除后核验 PASS。**

| 项目 | 实际执行结果 |
|---|---:|
| 文件条目，包括链接 | 186 |
| 其中仅删除链接本身 | 14 |
| 删除文件逻辑字节 | 100,830,055 B，约 96.16 MiB |
| 移除空目录 | 42 |
| 移除已停止的历史容器 | 11 |
| 保留的 Git/正式数据结构空目录 | 8 |

清理了旧 Example 视图与 infos、源码树 Example 链接、Stage1 smoke 大预测、旧下载 headers/log、已完成的 V4 一次性交接脚本及相应空目录。Stage1 小型原始 metrics 和 Stage1/2/3 验收报告、证据保留。链接只删除其自身，没有跟随到外部原始数据。

11 个停止容器的 inspect 与日志已保存后移除；其写入层合计 516,906 B，日志为空。`cooptrack-main` 的容器 ID、运行状态及共享镜像 ID 保持一致，未删除镜像或卷，未停止训练。文件逻辑字节不等于实测磁盘净回收量，Docker 的共享约 7.94 GB 镜像层没有计入清理量。

执行前发现上次空目录分类把 6 个 Git 内部目录及 2 个正式数据布局目录列为可清理，已纠正分类并保留。其他没有文件但含空子目录的目录树，自底向上逐项确认实际为空后移除；没有递归删除非空目录。

删除后逐路径确认 186 项文件/链接、42 个目录和 11 个容器全部不存在。V4 的完整运行身份与操作前一致，1,417 个受保护文件大小/SHA256 核验通过，6 份完整 train/val infos 的元数据一致，Git HEAD 与已接受的合并记录一致，当前训练及协调器进程继续运行。

条件清理项未执行：V4 raw/gzip 验证副本、原始点云、完整 wheel/CUDA archive、正式 V1/V2 预测与权重、共享 query cache 均保留。

执行明细：[execution.json](../audits/cleanup_execution_20261013/execution.json)。逐文件计划及原 SHA：[verified_plan.json](../audits/cleanup_execution_20261013/verified_plan.json)。删除后核验：[verification.json](../audits/cleanup_execution_20261013/verification.json)。容器身份与日志：[verified_containers.json](../audits/cleanup_execution_20261013/verified_containers.json)。实际 PowerShell 执行脚本和准备脚本保存在同一执行归档目录。
