# R50 10Hz IDS 专项诊断最终报告

**ROOT_CAUSE: UNRESOLVED（F）；CONFIDENCE: LOW。**
完整的旧 FULL 354→论文201差异尚未闭环。已证实的本地因素是 inference seed 对 reported IDS 的影响（HIGH）；所测大幅跨度主要伴随官方 best-MOTA operating point 变化。共同 score point 下差异显著收窄，固定 query 未明显改善 IDS。当前证据不支持把全部153归给随机 query、reset、checkpoint mismatch 或 CPU inverse。

| Setting | Scope | mAP | AMOTA | IDS |
|---|---|---:|---:|---:|
| Paper R50 10Hz | Table3 vehicle-only | 0.356000 | 0.346000 | 201 |
| Local baseline（原 FULL） | 21 scenes / 3316 frames | 0.362345 | 0.356898 | 354 |
| New controlled seed 0 | 21 scenes / 3316 frames | 0.362281 | 0.357281 | 301 |
| New controlled seed 2026 | 21 scenes / 3316 frames | 0.362063 | 0.351614 | 226 |
| New controlled seed 2025 | 21 scenes / 3316 frames | 0.362777 | 0.357175 | 362 |
| Deterministic query pilot（seed0） | 3 scenes / 611 frames | 0.316663 | 0.317360 | 84 |

三次full运行按预先pilot gate选择0、2026、2025，并非为接近201选择。固定query只覆盖611帧；其可比random seed0为mAP0.316274、AMOTA0.312358、IDS86。所测full最低IDS226仍高于201；136跨度不是对原153差异的因果分解。
论文目标见[官方主论文 Table3](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhong_CoopTrack_Exploring_End-to-End_Learning_for_Efficient_Cooperative_Sequential_Perception_ICCV_2025_paper.pdf)。

## 随机性、重复性与完整扫描

官方eval在保留历史track后，以CPU torch.randperm(900)的前缀选择新的learned detection模板。torch CPU seed控制它；reset清除轨迹、BEV和ID计数，却不重置RNG。模型构建和DataLoader iterator也消耗CPU RNG，所以seed数值必须连同初始化与执行顺序记录。这不是随机生成新Gaussian向量，也不是直接删除历史track。[randomness audit](../artifacts/core/ids_diagnosis/reports/randomness_audit.md)
完整scene0058（249帧）seed0两次的det/track JSON SHA256、ID序列和全部评估指标精确一致：mAP0.321645255、AMOTA0.336560011、IDS23。SEED_REPRODUCIBLE限定于该实测场景和设置；deterministic_algorithms=False不构成所有CUDA算子的全局保证。
新控制实验统一Python/NumPy/torch CPU/CUDA seed，cuDNN deterministic=True、benchmark=False、TF32 matmul/cudnn=True、PYTHONHASHSEED=0。五seed pilot的IDS为86/74/67/88/52，mean73.4、population std13.2、range36，满足预设20次或10% gate。完整val的mAP range=0.000713372、AMOTA range=0.005666659；IDS mean=296.333333、population std=55.619741、range136。
完整逐seed、逐scene、输出track数量、新ID、active数量、哈希与SWITCH帧见[pilot](../artifacts/core/ids_diagnosis/reports/pilot_seed_scan.md)、[full](../artifacts/core/ids_diagnosis/reports/full_seed_scan.md)和[独立trace增补](../artifacts/core/ids_diagnosis/scan_trace_enrichment.json)。

## 决定性补充：IDS取各run最佳MOTA点

AMOTA平均40个recall operating points；报告中的IDS取car最佳MOTA点。相同选择算法仍可能选出不同数值阈值。锁定旧baseline阈值做单次CPU重放的结果如下；正式指标保持原值。

| Full seed | 正式best threshold | 正式recall | 正式IDS | 共同0.5906070506572724点IDS | 共同点recall |
|---|---:|---:|---:|---:|---:|
| 0 | 0.617651013 | 0.491154 | 301 | 357 | 0.514957 |
| 2026 | 0.652858014 | 0.445716 | 226 | 364 | 0.512790 |
| 2025 | 0.590577262 | 0.518829 | 362 | 361 | 0.518509 |

正式best点IDS跨度136，共同点跨度7。支持的链条是seed→query选择/预测分数→最佳点选择→reported IDS；不能将136直接当作更多真实身份断裂。共同阈值仍允许预测分数、入选框和关联变化，不是纯关联反事实。recall直接使用devkit字段；该实现TP不含SWITCH，不能从TP/GT重新定义此列。[共同点分析](../artifacts/core/ids_diagnosis/reports/common_operating_point.md)

固定prefix采用独立方法副本，仅改randperm为arange。其三个reset首帧det/track数组6780个标量精确相同；后续593帧query选择、564帧cooperative association有差异。正式IDS86→84仅下降2；锁定random0点后86→91，排序反转。新active car ID476→466、分配当帧decoded新car ID769→749；这些不是GT SWITCH。[fixed诊断](../artifacts/core/ids_diagnosis/reports/fixed_query_diagnostic.md)及[共同pilot点](../artifacts/core/ids_diagnosis/reports/common_operating_point_pilot.md)。

## 旧354锚定、生命周期与事件证据

旧NATIVE/FULL两份det/track JSON精确相同。新full0两JSON、ID序列和指标不同，且首token000870的det已在首次randperm之前不同。旧cuDNN deterministic=False、启动时未设PYTHONHASHSEED；新为True/0。旧RNG state未保存，不能唯一归因某一后端设置，也不能把新full0内部trace当作旧354历史。首帧原行比较还包含topK排序交换；详见[baseline anchor](../artifacts/core/ids_diagnosis/reports/baseline_anchor.md)。
官方runtime record_threshold=0、active score>0.4、output score>=0.3、max_age=3。历史track低分可休眠，age1/2恢复时保留同ID；连续低分到age3删除。所有新full实测23 reset sessions=21场景起点+2场景内gap，backward0。两gap为003400→003418（1.799898s）和016809→016814（0.500215s），符合公开>0.5s条件。旧infra cache的22sessions使用另一时间轴，不是cooperative的reset计数。
新full0内部候选分配2933449、car active births2235、分配当帧decoded新car ID4351；最终JSON car first appearances4563（按实际reset session隔离）。候选分配、active出生、decoded输出、距离过滤后的JSON身份及GT IDS各有定义；稀疏trace无法恢复所有低分候选的精确car分配数。全11run query replacement实测0，旧run该内部字段仍UNKNOWN。[生命周期审计](../artifacts/core/ids_diagnosis/reports/track_lifecycle_audit.md)、[真实full0实例](../artifacts/core/ids_diagnosis/reports/full_lifecycle_trace_statistics.md)。
原FULL在best index22 / threshold0.5906070506572724精确重放354个SWITCH，全部传统tracking指标一致。观察模式A66、B2、E7、G279；C/D/F历史内部原因未知。E7仅与两gap重合，不能解释完整153。按seed20261004均匀随机抽查30件并逐条核验raw预测、GT、score、中心距离、MISS和reset；样本G22/A7/E1。before14件、after3件框来自evaluator插值/原框缺失，已单列。[事件报告及30条复核](../artifacts/core/ids_diagnosis/reports/ids_event_analysis.md)、[354事件JSONL](../artifacts/core/ids_diagnosis/ids_events.jsonl)。

## 次级数值与公开run来源

CPU审计使用11files、3033真实调用、247unique rotations、62469querypoints。CPU inverse与harness额外CPU observation逐位相同，FP64 NumPy/SciPy参考一致。inverse矩阵max/mean误差为1.194e-07/1.654e-08；inverse-only点L2 max为1.062e-05m，结论为CPU_INVERSE_PATCH_NOT_PRIMARY_SUSPECT。
完整CPU FP32流程点L2 max约0.000174m，实际GPU helper相对FP64 max约0.066932m；后者包括完整坐标/GPU math路径，来源未隔离，不能归给inverse patch，也不能据此断言所有数值因素均无影响。未恢复会崩溃的GPU inverse，未做禁TF32反事实或额外模型forward。[numerical audit](../artifacts/core/ids_diagnosis/reports/numerical_patch_audit.md)。
最后重新读取官方README、TRAIN_EVAL、GitHub issues/作者回复/releases和Hugging Face。PUBLIC_CHECKPOINT_TO_TABLE3_RUN=UNCONFIRMED。训练metadata seed或test.py默认seed不证明论文精确inference seed；epoch metadata也不能单独证明训练总轮数。公开权重身份匹配仍成立；未确认run映射不等于已确认checkpoint mismatch。[fresh provenance](../artifacts/core/ids_diagnosis/reports/checkpoint_run_provenance.md)。

## 最终决策与范围

**ROOT_CAUSE:** UNRESOLVED（F；完整354→201）

**CONFIDENCE:** LOW；已证实本地reported-IDS seed sensitivity为HIGH。

**EVIDENCE:** 官方随机路径、同seed精确重复、五seed pilot、三次完整val、共同点range7、fixed无显著改善、原354+随机30复核、inverse-only小误差、公开run映射仍需确认。

**DOES_THIS_AFFECT_MAIN_COMMUNICATION_CONCLUSION:** NO。未推翻已有通信字节统计、FULL_FP32与native精确等价或已保存主指标观察；微小AMOTA排名与IDS改善仍不能宣称对inference seed稳健。

**NEED_MODEL_RETRAIN:** NO。

**NEXT_MINIMAL_ACTION:** 先取得论文 Table3 原 run 的 tracking metrics_details.json / best-MOTA threshold、checkpoint SHA、inference seed、cuDNN/TF32 与执行顺序 manifest；有条件再取得原 tracking JSON 和 cache-generation manifest。将本地 seed 与后端完整记录；当前无需重训或继续挑 seed。

此前类别对齐、10Hz、split、checkpoint文件身份、cache输入合同及原NATIVE/FULL精确等价结论继承，旧R50报告与产物保持原样。本次不训练、不改GT/物理split、不重新生成cache、不调tracking阈值或关联算法、不开展新通信实验。固定query仅为独立诊断。
独立只读校验 **PASS**：26,027项检查、31个评估scope、3,552个文件哈希；43个登记保护项均未变化。未对全部raw/prepared GT树重新计算哈希，保护范围以manifest与只读挂载为准。详见[校验报告](../artifacts/core/ids_diagnosis/reports/validation.md)。

本次新建的两个诊断容器已停止，原两个容器状态未变化，见[cleanup记录](../artifacts/core/ids_diagnosis/cleanup.json)。机器结果为[final.json](../artifacts/core/ids_diagnosis/final.json)。
