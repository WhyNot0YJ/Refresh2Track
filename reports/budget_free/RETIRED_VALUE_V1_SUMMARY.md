# H 路线退役记录

`H_METHOD_RETIREMENT STATUS=COMPLETE`。实际通过 Dry-Run 并执行 `-Apply`，删除 **12,468 个文件、48 个空目录**，回收文件逻辑字节 **1,575,407,364 B**（约 1.47 GiB）。项目位于 D 盘，删除前后卷可用空间为 **22,999,248,896 B → 24,596,766,720 B**；文件系统分配和并发写入影响卷差额，不能用它替代成功删除字节总和。逐文件结果与所有失败/成功尝试见 [H_RETIREMENT_MANIFEST.json](H_RETIREMENT_MANIFEST.json)。此前 BLOCKED 与 Dry-Run 均没有计入删除量。

原文保留 [VALUE_V1_FINAL.md](VALUE_V1_FINAL.md) 和 [VALUE_V1_DIAGNOSTICS.md](VALUE_V1_DIAGNOSTICS.md)，两份报告字节未变。[value_v1_results.csv](value_v1_results.csv) 为 2 行、226 列完整精度指标/身份/来源；[final_h8_training.json](final_h8_training.json)、[final_h1_training.json](final_h1_training.json) 均为源文件的完整 40 轮标量日志原字节副本。另保留 [协议快照](value_v1_protocol_snapshot.json)、[完整标量结果](value_v1_scalar_results.json)、[诊断聚合](value_v1_diagnostic_scalars.json)、压缩历史保护快照和旧终端报告。14 项归档载荷合计 **6,196,826 B**；含清理审计的整个归档目录实际字节总量另见文末和 manifest 的 `archived.archive_total_bytes`。

8 份副本（含 gzip 解压后字节）的历史 SHA 校验通过；两份原报告 SHA 不变；生成协议快照仅更新本次实际退役和可复现性状态，没有改写历史配置、训练身份或结果。见 [H_ARCHIVE_VERIFICATION.json](H_ARCHIVE_VERIFICATION.json) 和 [删除后独立核验](H_RETIREMENT_VERIFICATION.json)。

| 删除类别 | 文件数 | 实际删除字节 |
|---|---:|---:|
| H 专用源文件（含重复源码） | 35 | 645632 |
| H 专用配置 | 1 | 2064 |
| D0/D1 训练标签数据 | 12299 | 668901607 |
| H checkpoint | 5 | 1798533 |
| H 预测与逐帧通信 | 20 | 813984757 |
| H 中间诊断与大日志 | 39 | 66845764 |
| H 冗余证据文件 | 69 | 23229007 |

精确 [白名单](H_RETIREMENT_WHITELIST.csv.gz) 的全部 12,516 项已不存在；包括 H 教师/标签/学生/评估/诊断代码、配置、round0/final 权重、optimizer 状态、D0/D1、1816 帧 partial 输出、完整评估预测与通信、逐样本 NPZ、OOF 数组及中间拟合产物。H 范围内没有 KEEP_SHARED 或活动/身份不确定遗留项，没有策略拒绝项。共享 train/val query cache 保留原位置。

D0=4096、D1=2048、总计6144，Pilot 1024 已包含在 D0；H8/H1 正例470/69。H8 是八帧**平均**收益，H1 是首帧收益，epsilon 均为0.01。2324维输入为当前 INT8 往返 latent、最后实际解码镜像、差值各768维，加20维因果状态；没有预算、GT、未来或车端私有状态。教师冻结，学生 MLP 在 CPU FP32 下拟合40轮。

| 方法 | 平均 B/frame | P95 | mAP | AMOTA | AMOTP | MOTA | IDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| H8 | 2465.701447527141 | 4640 | 0.36223574386625923 | 0.35795131154675774 | 1.3221018321048212 | 0.28424754867130875 | 217 |
| H1 | 2145.79794933655 | 3801 | 0.3532862384962936 | 0.3334648702205824 | 1.3328284243169157 | 0.2679053573966179 | 292 |

H8 可选更新1397/42772=0.03266155428785186，含初始化为2954/44329=0.06663809244512621；H1 可选更新37/42772=0.0008650519031141869，含初始化为1594/44329=0.03595840194906269。其他既有标量，包括 cache age、延迟和完整比较，全部保存在 CSV/JSON；未只保留有利指标。

D0 五折20轮诊断的 H8 OOF PR-AP=0.15825611749176277、H1=0.023219567719326074。它们不是正式40轮学生模型的独立官方测试。H8 平均收益算术校验通过：6144样本、98304存储帧，E/LOC/gain 残差为零，组件最大残差约6.439e-15；416个 H8 正例的 H1 非正，后续帧代理收益确实存在。跨 scene 预测较弱，H1 阈值决策失败、OOF recall 为零；不能概括成“标签全错”或“网络完全没有学会”。

限制包括单 seed0、固定 continuation、教师/学生可观测信息差异；随机性控制只覆盖一名候选，独立等 payload UPDATE/REUSE 安慰剂控制未完成。旧160000计数为158184离线调用加1816帧 H8 partial，当时 H1 未运行；后续独立6632/7000完整评估为3316×2，从新计数开始，未回写旧计数。当前选择参考过 official val。

H8 final 历史 SHA：`111b14ae0b9a038b550e84d16cd665f9264dea9bcc670d40e868adafd5986d38`。H1 final 历史 SHA：`87044b5142e7cdd9407ad59d617bee46bcebd39606dbd649b22fab45116b39f1`。哈希和训练身份保留，权重本体已删除。**完整本地 H 重跑、独立复评与完整复现能力已主动放弃**。原报告中指向已删除代码、配置、标签、权重、预测、通信及中间诊断的历史链接将失效；保留结果表和 SHA 不代表仍可完整复现 H。

原 V1/V2 的 **130 个受保护 SHA 身份**均不变，**476 个受保护实验/配置/审计文件**的路径、长度、mtime 不变；Git 原有修改和未跟踪文件保持原状。原生源码、官方权重、原始/转换数据和共享缓存做了代表性元数据检查，没有重新做全数据哈希，也没有声称全盘 PASS。V2 MLP final SHA=`d55e67bd868696e01de10810bbb314559399d075438fad501b0b9ff3761e7b2c`；Transformer final SHA=`58bda699962f479a7556e75c6ad3b0389d6a909ca7b798e5eece2ea7f2fc65a3`。B1/B2/B3/B4、Periodic P1/P2/P4/P6/P8/P10/P12/P16/P24/P32、INIT_ONLY、FULL_FP32/INT8 参考、2×2 A/B/C/D 和 V2 Bug 审计均保留。删除后18模块导入、CLI、codec/cache/scheduler 自检通过，见 [H_PROTECTED_AFTER.json](H_PROTECTED_AFTER.json)。

历史全局保护 FAIL 原样保留：539个选定 SHA、348个旧路径缺失（116文件、232目录）、3个变化。历史另一轮12925/12925 PASS 也按独立历史记录保留。本次新增 `AUTHORIZED_H_RETIREMENT` 差异记录与定向保护结果，没有改写旧 FAIL，也没有把预期 H 删除当作 V2 损坏。

研究主线为 `ACTIVE=V2_BUDGET_CONDITIONED_REFRESH`；`RETIRED=BF_VALUE_H8/BF_VALUE_H1/VALUE_V1_DIAGNOSTICS`。一个原 V2 MLP 共享参数接收 B1/B2/B3/B4 条件，预算3474/6382/9290/12198 B/frame，实际字节以 serializer 为准。保留当前特征/实收缓存/差值/状态、Task+单边超预算 Rate、INT8 与几何逐帧更新。B1 只是现有 seed0 的 AMOTA—平均通信候选，未证明所有预算/种子/指标最优；B2–B4 不保证 AMOTA 单调增加，全部结果与 Transformer 对照继续保留。

取消尚未执行的 H8 额外 seed、H1/H8 比较及反事实正例复核。本地没有找到对应可修改任务队列，保存的是计划修订，未伪造队列操作。后续 B1 与相近通信 Periodic 的重复性、完整曲线和通信行为核验需另行实验指令。本次未开始新训练或 official val，未启动 Delta+32，未恢复 TTL/max_staleness/Top-K 或四帧强刷。见 [实验索引](../EXPERIMENT_INDEX.md) 与 [计划修订](H_RETIREMENT_PLAN_REVISION.md)。

用户已确认 V3 指 H1/H8，相关专用文件均已退役；没有独立的 H/V3 镜像、容器或卷，原 V1/V2 共用镜像保留。额外授权的 Docker **14.71 GB 构建缓存已清理**。用户又授权暂时停止并恢复其他容器；正常退出 Docker 时内置压缩实际将数据盘从160,829,538,304 B缩小至137,834,266,624 B，回收22,995,271,680 B（约21.4 GiB）。管理员 DiskPart 随后成功完成，没有额外回收。服务恢复后的空间和验证记录见 [DOCKER_SPACE_AUDIT.md](DOCKER_SPACE_AUDIT.md)。

archive_total_bytes=10237533 B（reports/budget_free 实际文件合计）。
