# VALUE_V1 现有数据集低成本诊断

诊断已完成；原成果保护检查：`PASS_DIAGNOSTIC_PRESERVATION`，12976/12976份文件SHA不变、差异0、原value目录新增文件0。本报告独立于 [VALUE_V1_FINAL.md](VALUE_V1_FINAL.md)，没有改写正式实验结论或结果。

最重要的发现：H8 确实使用八帧**平均**收益，没有求和放大八倍的错误。6144 条存档的评分、收益和标签算术均一致，存在明显的后续帧代理收益。当前更直接的弱点是学生的场景泛化：D0 五折留出 PR-AP 为 H8=0.158256、H1=0.023220，明显低于同轮数训练折指标；固定阈值下 H1 留出召回为零。既有随机性控制覆盖仅一个候选，不能据此估计全数据正例噪声率；独立等 payload 安慰剂没有留存证据。

```text
P0_GAIN_DEFINITION_AND_ARCHIVE_ARITHMETIC: PASS
P0_FULL_6144_DISTRIBUTIONS: COMPLETE
P1_D0_SCENE_GROUPED_5_FOLD_CV: COMPLETE
P1_D0_D1_DISTRIBUTION_AND_PROVENANCE: COMPLETE
P2_RETAINED_REPEAT_AND_BRANCH_ORDER_CONTROLS: PASS_WITH_LIMITED_COVERAGE
P2_EQUAL_PAYLOAD_UPDATE_REUSE_PLACEBO: NOT_FOUND_IN_RETAINED_EVIDENCE
DIAGNOSTIC_NATIVE_FORWARDS_ADDED: 0
DIAGNOSTIC_GPU_FORWARDS_ADDED: 0
NEW_LABEL_PAIRS: 0
D2_CREATED: NO
FORMAL_CHECKPOINTS_RETRAINED_OR_MODIFIED: NO
```

“只读”约束针对原始成果；用户明确允许的小 MLP 交叉验证及统计产物写入独立目录 [experiments/budget_free/value_diagnostics_v1](../../experiments/budget_free/value_diagnostics_v1)。只做 10 个诊断拟合、每个固定 20 epochs，未保存可部署 checkpoint。没有读取图像执行 CoopTrack、重做 matcher、生成反事实标签、运行 official evaluator，或依据诊断修改正式阈值、模型和协议。

**P0：收益定义与 6144 条完整核验**

冻结代码 [counterfactual.py](../../scripts/budget_free/counterfactual.py) 的 `deltas` 为：

```text
δk = E_REUSE(t+k) - E_UPDATE(t+k)
ΔH1 = δ0
ΔH8 = mean(δ0, ..., δ7)
E = FN + FP + IDS + 0.1 * sum(min(matched_center_distance / 2, 1))
label_H1 = int(ΔH1 > 0.01)
label_H8 = int(ΔH8 > 0.01)
```

E 没有除以 GT 数量；匹配距离严格小于 2 m。H8 为固定帧数平均，非按时间加权，也不是无限时域收益。H8 正标签等价于 `sum(δ0..δ7)>0.08`。若误用 `sum>0.01`，现有记录会得到 796 个正例；实际冻结定义是 470 个，未更改任何标签。

逐6144条/98304个已存分支帧重新核对距离、匹配数量、FN/FP/IDS、LOC、E、两分支帧对齐、H1/H8 收益和严格标签：全部通过。E、LOC、存储 H1/H8 的最大重算残差均为 0；四项差值相加的最大浮点残差为 `6.439293542825908e-15`。两个 NPZ 与 metadata 的 key、动作前 age、标签及来源 schedule 对齐也无差异。

| 集合 | H8=0,H1=0 | H8=0,H1=1 | H8=1,H1=0 | H8=1,H1=1 | H8正例 | H1正例 |
| --- | --- | --- | --- | --- | --- | --- |
| D0，4096条 | 3788 | 8 | 265 | 35 | 300 | 43 |
| D1，2048条 | 1871 | 7 | 151 | 19 | 170 | 26 |
| 总计，6144条 | 5659 | 15 | 416 | 54 | 470 | 69 |

H8 正例中 416/470（88.51%）在首帧不是 H1 正例。其中首帧原始差值负/零/弱正分别为 108/186/122；“H1非正例”表示 `ΔH1≤0.01`，并不等于原始收益 `≤0`。这 416 条的首帧平均差值 -0.006810176，后七帧总差值平均 +2.600378290；对 H8 平均值的贡献分别为 -0.000851272、+0.325047286，合计 +0.324196014。

该组按定义需要后续净收益来超过阈值；上述计算进一步给出真实存档中的幅度和组成，而不是另一项独立随机性控制。416 条的 H8 平均收益分解为 FN=0.010817308、FP=0.271033654、IDS=0.039062500、LOC=0.003282553，主要来自 FP 项。这里使用各分支自身的已存匹配结果，不能解释为同一物体的纯定位因果效应。

| 帧偏移 k | 全6144条 δk均值 | δk中位数 | δk P95 | 416条子集 δk均值 | 全体负 / 零 / 正 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.002452980 | 0.000000000 | 0.001463030 | -0.006810176 | 1044 / 3910 / 1190 |
| 1 | 0.007477233 | 0.000000000 | 0.004235469 | 0.107761733 | 1587 / 2623 / 1934 |
| 2 | 0.009862469 | 0.000000000 | 0.005246433 | 0.201957068 | 1638 / 2555 / 1951 |
| 3 | 0.012313601 | 0.000000000 | 0.007203718 | 0.359015917 | 1652 / 2499 / 1993 |
| 4 | 0.015404719 | 0.000000000 | 0.008599274 | 0.448349909 | 1707 / 2444 / 1993 |
| 5 | 0.012077382 | 0.000000000 | 0.010481188 | 0.416341395 | 1705 / 2406 / 2033 |
| 6 | 0.016626343 | 0.000000000 | 0.012129225 | 0.550210943 | 1728 / 2374 / 2042 |
| 7 | 0.009396452 | 0.000000000 | 0.011075494 | 0.516741325 | 1817 / 2327 / 2000 |

完整逐行两分支 E、FN、FP、IDS、LOC、八帧差值及 sample keys 在 [per_frame_decomposition.npz](../../experiments/budget_free/value_diagnostics_v1/p0/per_frame_decomposition.npz)，可读索引在 [sample_index.csv](../../experiments/budget_free/value_diagnostics_v1/p0/sample_index.csv)。全部分位数、原始正负交叉表、各帧正负贡献与组件统计在 [gain_diagnostics.json](../../experiments/budget_free/value_diagnostics_v1/p0/gain_diagnostics.json)。

**P0：收益分布及阈值附近样本**

| 收益 | 原始 Δ<0 | 精确 Δ=0 | 0<Δ≤0.01 | 正标签 Δ>0.01 | 全体均值 | P05 | median | P95 | P99 | min / max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H8平均 | 1737 | 2320 | 1617 | 470（7.65%） | 0.010701397 | -0.009954841 | 0 | 0.125588253 | 0.740953321 | -2.624985794 / 2.500326085 |
| H1首帧 | 1044 | 3910 | 1121 | 69（1.12%） | 0.002452980 | -0.000913796 | 0 | 0.001463030 | 0.012194439 | -1.903758581 / 1.971625483 |

下面使用预先固定的区间 `abs(Δ−0.01)≤容差`。括号内为区间中实际正标签数量；只统计现有数值，没有按新阈值训练或选择正式规则。

| 容差 | H8阈值附近条数（其中正例） | H1阈值附近条数（其中正例） |
| --- | --- | --- |
| 0.000001 | 0（0） | 0（0） |
| 0.0001 | 2（1） | 1（0） |
| 0.001 | 18（9） | 6（2） |
| 0.005 | 105（28） | 72（13） |

`abs(Δ)≤0.001` 的 H8/H1 分别有 4755/5478 条，它们都不是正标签。0.01 过滤了大量零附近微小收益；既有H1/H8 gain均无样本落在 `0.01±1e−6` 范围。但区间统计不是执行噪声分布，也不证明阈值最优或标签跨 seed 稳定。

470 个 H8 正例中，419（89.15%）有净整数 FN+FP+IDS 改善，八帧平均整数项至少为 0.125，单独足以超过 0.01。另51条净整数项为零、靠 LOC 超过阈值：45条八帧所有整数项均未变，6条有整数项变化但相互抵消。没有净整数负贡献被 LOC 扭转成正例的记录。

这表明大部分 H8 正例含离散匹配/检测/ID 项变化；这些变化也可能涉及后续关联分支，不能仅凭其幅度排除随机路径影响。

**P1：仅 D0 的按 scene 五折交叉验证**

规格在任何拟合前冻结：[spec.json](../../experiments/budget_free/value_diagnostics_v1/cv/spec.json)，SHA256=`b5f9d8f069cea8d402d52957ec9d58f4fe4bb8b6296979db30d0f4170261d0bb`。

46个 scene 按样本数降序、scene 字典序排序，依次分配到当前行数最少的折；分折不使用标签。H8/H1 使用相同折，留出行数813/847/812/812/812，scene无交集，每条D0行恰有一次 OOF 预测。D1 未进入交叉验证，因为它的 roll-in 学生见过全部 D0 场景。留出scene只针对每折学生MLP；冻结CoopTrack教师已在这些official train场景训练过，因此这不是整个系统的未见场景泛化测试。

每折训练部分单独计算 latent/state 归一化、P75收益 margin 权重尺度和固定 mean weight，没有加载全 D0 正式归一化。模型仍为2324维、84801参数、共享768→64投影和212→128→64→1；CPU FP32，seed0，batch128，AdamW wd1e−4，clip5，首轮1e−4→1e−3、其余19轮余弦至1e−5。每次从头拟合20轮，无early stopping、折内val选择、类别平衡或调参；0.5阈值固定。

| 模型 | D0正例比例 / 常量分数AP基线 | pooled OOF PR-AP | 五折宏平均AP | 同20轮训练折AP范围 | OOF precision | OOF recall | OOF UPDATE比例 | TP / FP / FN / TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H8诊断MLP | 0.073242 | 0.158256 | 0.165292 | 0.499958–0.568481 | 0.162587 | 0.310000 | 0.139648 | 93 / 479 / 207 / 3317 |
| H1诊断MLP | 0.010498 | 0.023220 | 0.030720 | 0.333562–0.559650 | 0 | 0 | 0.010498 | 0 / 43 / 43 / 4010 |

| 折（从0编号） | 留出行数 | H8正例 | H8留出AP | H8训练AP | H1正例 | H1留出AP | H1训练AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 813 | 61 | 0.152152 | 0.542185 | 15 | 0.035809 | 0.438038 |
| 1 | 847 | 51 | 0.123825 | 0.568183 | 2 | 0.009301 | 0.333562 |
| 2 | 812 | 59 | 0.180774 | 0.568481 | 8 | 0.032791 | 0.426491 |
| 3 | 812 | 75 | 0.226583 | 0.499958 | 12 | 0.054355 | 0.469528 |
| 4 | 812 | 54 | 0.143129 | 0.512380 | 6 | 0.021345 | 0.559650 |

AP 是 tie-grouped non-interpolated average precision，已与 sklearn 独立核对；不同折的分数校准可以不同，因此同时披露各折、宏平均和 pooled OOF。H8 正例分布于45/46场景，H1只有22/46场景；H1一折仅2个正例，单scene/单fold估计不稳定。全部46场景指标（无正例scene AP=null）见 [per_scene_metrics.json](../../experiments/budget_free/value_diagnostics_v1/cv/per_scene_metrics.json)。

定义 margin regret 为 `mean(max(Δ−ε,0)−UPDATE*(Δ−ε))`，只用于现有单query标签的离线决策诊断，不等于部署跟踪损失、通信成本或 official AMOTA。

| 诊断模型 | OOF margin regret | Always REUSE | Always UPDATE | 仅由各训练折选择的常量动作 |
| --- | --- | --- | --- | --- |
| H8 | 0.020700079 | 0.023154418 | 0.022493909 | 0.024649288 |
| H1 | 0.004351098 | 0.004248584 | 0.010866929 | 0.004248584 |

H8 常量动作由训练折 margin 决定：折0/1/2为REUSE、折3/4为UPDATE；H1各折均REUSE。没有用留出收益挑选常量。H8 pooled regret相对Always REUSE降低10.60%，相对Always UPDATE也较低，但前两折分别比Always REUSE差；H1 pooled比Always REUSE高2.41%，固定0.5阈值仅产生43个假正例、漏掉43个真正例。

因此不能说完全没有可预测信号：两者AP都高于常量分数基线，H8有一定OOF收益；也不能用训练内AP声称可靠跨scene泛化。较大fit/heldout落差及H1决策失败在不使用D1时已经出现，D1不匹配并非诊断中出现这些弱点的必要条件。固定20轮、单seed、小样本及稀疏正例限制仍在，不能证明标签在原则上不可学习，或把20轮学生诊断指标当作正式40轮checkpoint的official结果。

完整 stdout 为 [cv/run.log](../../experiments/budget_free/value_diagnostics_v1/cv/run.log)，逐轮loss/lr/grad记录在 `cv/folds/fold_*_h*.json`，结果和每行OOF概率在 [summary.json](../../experiments/budget_free/value_diagnostics_v1/cv/summary.json)、`cv/oof_h8.npz`、`cv/oof_h1.npz`。10个拟合共21.06秒（拟合及其结果输出阶段，不含整个诊断I/O）；速度正常，因为只处理预提取数值特征。

![同轮数fit与scene-heldout PR-AP](../../experiments/budget_free/value_diagnostics_v1/figures/scene_cv_pr_ap.png)

**P1：D0、D1 的年龄与来源策略**

age来自未归一化FP32 `features[:,2304]` 的 `elapsed_frames`：本帧基础设施frame_idx减上次实际UPDATE frame_idx，动作前统计。它不是最后一维track age、墙钟秒数或归一化值；6144条逐行与metadata一致。

| 集合 / 来源 | n | age mean | median | P95 | max | H8正例数 / 比例 | H1正例数 / 比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 全体 | 4096 | 13.996582 | 7 | 47 | 436 | 300 / 7.32% | 43 / 1.05% |
| D0 P8 | 1135 | 4.407048 | 4 | 8 | 16 | 42 / 3.70% | 2 / 0.18% |
| D0 P32 | 1501 | 11.822119 | 8 | 32 | 32 | 93 / 6.20% | 12 / 0.80% |
| D0 INIT_ONLY | 1460 | 23.686986 | 12 | 100.05 | 436 | 165 / 11.30% | 29 / 1.99% |
| D1 ROUND0_H8 | 2048 | 20.902344 | 10 | 84 | 198 | 170 / 8.30% | 26 / 1.27% |

| 动作前age区间 | D0 n / 占比 | D0 H8正例 / H1正例 | D1 n / 占比 | D1 H8正例 / H1正例 |
| --- | --- | --- | --- | --- |
| [0,2) | 571 / 13.94% | 19 / 2 | 215 / 10.50% | 12 / 3 |
| [2,4) | 710 / 17.33% | 29 / 3 | 306 / 14.94% | 19 / 4 |
| [4,8) | 856 / 20.90% | 55 / 8 | 356 / 17.38% | 22 / 1 |
| [8,16) | 898 / 21.92% | 67 / 12 | 396 / 19.34% | 24 / 0 |
| [16,32) | 625 / 15.26% | 59 / 13 | 394 / 19.24% | 32 / 9 |
| [32,∞) | 436 / 10.64% | 71 / 5 | 381 / 18.60% | 61 / 9 |

D1来源checkpoint是只用D0训练的round0 H8（40轮、seed0），SHA256=`f578364db47286c1adaaa29a0988e73cec0d0b68754bf62f43e438b4c374b86d`。D0和D1都覆盖原46个train scenes；`rollin_policy_sha`绑定的是完整参考动作schedule，不能误当checkpoint SHA。

D1选中样本缓存更老：均值增加6.905762帧、≥32帧占比从10.64%到18.60%。但样本由scene/age/policy轮转、每track上限8例、目标数量截断产生；选中分布不是全部query的自然部署分布。按track上限后的完整候选池，D0/D1为15708/5261例，age mean22.220970/37.299183；D1候选池max432，选中集max198。策略组成、query身份、状态历史和采样截断均有差异，不能把age变化单独归因为H8模型或证明H1受害。

H8正例比例D0→D1为7.32%→8.30%，H1为1.05%→1.27%；H1 mean gain却由0.003381655降为0.000595629，H8 mean gain为0.010660508→0.010783175。存在描述性分布变化，直接混合D0+D1随机分行的独立性解释不成立。本轮未另训D1消融，不能隔离D1对正式H1的因果影响。

完整dataset/policy/scene/source/age-bin收益分位数与正例比例见 [distribution.json](../../experiments/budget_free/value_diagnostics_v1/distribution_controls/distribution.json) 和 [distribution_age_bins.csv](../../experiments/budget_free/value_diagnostics_v1/distribution_controls/distribution_age_bins.csv)。

![已存后续帧收益分解与选中样本年龄](../../experiments/budget_free/value_diagnostics_v1/figures/delayed_gains_and_age.png)

**P2：已有受控复现、信息隔离及安慰剂证据边界**

重新核对既有 [correctness_branch_records.json](../../experiments/budget_free/value_v1/evidence/correctness_branch_records.json) 的80个存档分支帧：RR1/RR2、UU1/UU2同动作重复，以及UR/RU分支顺序各8帧的prediction、decoded input、packet、flags、effective IDs及score均精确一致。没有重新运行控制。

既有候选是scene0000、P32、age3、index35，key=`01c01b66de3cb7204b8a99ecdd9bdf5506398867737692f0636021b4f7e1b76b`，确实在D0。控制覆盖仅1/6144候选（0.0163%）、一个scene/一个policy；它的D0标签source为revision1。`correctness.json`自身未显式绑定控制执行时的source identity，不能把该候选行的source当作控制producer身份。不同归档中控制代码AST相同，并不证明每种label source都重新执行过控制。

| 存档控制 | H8平均gain | H1首帧gain | H8 / H1标签 |
| --- | --- | --- | --- |
| 主seed0窗口 | 0.000302948777 | 0 | 0 / 0 |
| 次seed1窗口 | -0.000013091913 | 0 | 0 / 0 |

次seed只是在同一主seed0前缀状态上重新设置窗口RNG，并不是整scene独立seed1 rollout。小幅原始gain改变了符号，冻结标签未翻转；这一个非正例不足以估计正类噪声率。

未找到“把UPDATE当前payload强制等于REUSE缓存payload、再比较不同动作”的独立安慰剂存档。RR/UU是同动作重复，不能替代该控制。unsent/GT poison、tensor independence、INT8实际decode、每帧geometry、冻结参数的原PASS及源码条件保留，但poison原始分支未写入records，不能宣称本次从原始poison输出完整重放验证。

额外利用现存档案的零消费检查：2101条（D0=1393、D1=708）候选八帧都未通过native输入几何mask，全部八帧E差值精确为0，H1/H8正例均0，无异常；4043条至少一帧有效消费承担所有470/69个正例。这支持所记录无消费路径的零收益一致性，不能替代等payload安慰剂，也不能证明有效消费正例均不受随机性影响。

源码确有eval active-track填充的CPU `torch.randperm`；GridMask eval虽不改图像，条件求值仍消耗NumPy抽样。snapshot恢复Python/NumPy/Torch/CUDA起始RNG，但动作可改变后续active tracks、填充子集和执行分支；未保存每帧完整RNG轨迹，也不保证所有CUDA kernel确定性。随机执行路径的存在不能直接证明已观察gain是随机噪声。完整证据、源代码位置、支持范围和未证明范围见 [controls.json](../../experiments/budget_free/value_diagnostics_v1/distribution_controls/controls.json)。

**原成果保护、身份链与研究结论**

诊断前fresh保护基线为12976份文件：沿用以前明确保护的路径，并新增纳入现存完整official-val产物、所有原value文件及现有scripts/configs/reports。基线 [preservation_before.json](../../experiments/budget_free/value_diagnostics_v1/preservation_before.json) SHA256=`48147cb89717ed7dc2c14613236b7d3acb88add885a55ffacabe707117120cd2`；结束结果见 [preservation_after.json](../../experiments/budget_free/value_diagnostics_v1/preservation_after.json)。原160000停止状态、1816帧partial和完整新val6632计数分别保留，未回写账本。

正式H8 checkpoint仍为`111b14ae0b9a038b550e84d16cd665f9264dea9bcc670d40e868adafd5986d38`，正式H1仍为`87044b5142e7cdd9407ad59d617bee46bcebd39606dbd649b22fab45116b39f1`。原训练标签、权重、逐轮训练日志、protocol/state/manifest、预测、正式指标、源码和原报告都按当前fresh SHA核对；未扫描原始sensor/query-cache全树。历史整个工作区`RAW_PRESERVATION_FAIL`继续保留，本次通过也不会覆盖该历史失败。

根审查 [diagnostic_audit.json](../../experiments/budget_free/value_diagnostics_v1/diagnostic_audit.json) 独立核对完整分解数组、原NPZ keys/age、OOF来源/scene划分/全覆盖、混淆矩阵、margin regret、训练折常量选择、10份20轮日志及绑定SHA，结果PASS。P0、分布控制及CV各有输入哈希；最终产物清单见 [diagnostic_manifest.json](../../experiments/budget_free/value_diagnostics_v1/diagnostic_manifest.json)。

本轮证据支持的研究表述是：**学生学习固定参考continuation下，单query当前UPDATE相对REUSE的八帧平均离线代理收益是否超过0.01。** 存档中后续帧收益明确，不能归因于sum尺度放大；大部分H8正例也不是阈值旁的微小LOC变化。它仍不能等同于真实部署策略下无限时域的“长期刷新必要性”。

标签分支固定目标query后七帧REUSE、其他ID沿参考schedule；正式部署可能再次UPDATE该query，并同时改变多个query决策。单query边际收益、固定未来动作、partial observability和实际策略联合决策之间的差异仍未被本轮消除，也没有隔离估计其性能损失。

目前最直接的新证据是有限的跨scene预测能力与明显fit/heldout落差，尤其H1固定阈值决策失败；另有D0/D1描述性shift和控制覆盖不足。不能把它们写成“6144标签都是噪声”，也不能写成“数据集已全面证明长期价值”。正式val仍是既有混合/不利结果，未被本次诊断替换。保留现有数据与模型，将研究结论限定到上述监督目标和证据范围；本轮不生成D2、不调阈值、不续训正式模型。
