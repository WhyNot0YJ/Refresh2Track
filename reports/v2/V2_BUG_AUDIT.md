# V2_BUG_AUDIT

审计日期：2026-10-09，项目 `D:\CoopTrack_Audit`。

## 1. 结论

**RESULT = 审计未完成。确认影响正式实验有效性的 Bug：0 个。**

当前实现的 REUSE、原生融合输入、P32、INIT_ONLY、缓存隔离、通信计量和原生 task 梯度均有新的受控测试支持；现存预测的样本映射、不同方法内容差异、坐标格式化及指标归档一致性也已检查。

但用户要求的“预测文件与正式 run manifest SHA256 匹配”无法完成：16 个新 V2 run 没有记录生成时的预测内容 SHA256。当前新算 hash 只能标识现存文件，不能证明它就是当时评估的文件。不能用内容检查通过代替这项历史绑定，因此不宣布“无Bug”，也不把证据缺失当成已经复现的实现 Bug。

范围包括 8 个新增周期方法和 MLP/Transformer 各 4 个预算；P1/P2/P4 仅只读核验已有结果。没有训练、optimizer.step、完整 val forward 或 evaluator 重跑。全部审计代码及机器证据仅在 `audits/v2_bug_audit/`；已有源码、配置、权重、缓存、指标及报告未改。

## 2. 强制检查结果

| 项目 | PASS/FAIL/未验证 | 实际证据 | 是否影响正式结果 |
|---|---|---|---|
| REUSE 真正使用旧 latent | PASS | 真实连续 ID152；固定 REUSE 后当前三 latent 强扰动，packet 与最后接收的三 latent bit-exact 不变；§4 / D | 未复现失效 |
| 当前 latent 无旁路泄漏 | PASS | 全 alignment、初始化、reasoner、association、fusion 输入和预测相同；UPDATE 对照实际改变各输入；原生路径无直接 cache 文件读取；§3–4 / D | 未复现失效 |
| P32 周期正确 | PASS | 真实 70 帧 ID869 更新恰为 `[0,32,64]`，逐帧 flags/bytes 与原通信日志相同；§5 / P | 未复现失效 |
| INIT_ONLY 正确 | PASS | 44,329 observations = 1,557 init + 42,772 optional；optional UPDATE=0；§5 / P | 未复现失效 |
| 缓存 scene/ID/session 隔离 | PASS | 实际 session reset、新 ID，合成 scene/ID switch、消失再现、batch lane、alias 测试；§5 / P、T | 未复现失效 |
| 当前几何更新正常 | PASS | REUSE 仍使用当前 `N×3` ref_pts、`N×10` pred_boxes；当前校准与 official pair 对齐；§3–6 / D、P、E | 协议允许，无违规高维传输 |
| Serializer 和 bytes 正确 | PASS | 实际 pack/unpack、780B 差分；16×3316 通信行及 mean/P95/total/nominal B/s 重算；§5 / P | 未复现失效 |
| 数据划分与评估正确 | **未验证：历史预测 SHA 绑定缺失；其余内容检查 PASS** | 官方 46/21 scene 分离，现存 16 份预测各覆盖3316，无重复整体预测，五指标/坐标/配置核验；§6 / E | 不能完全确认历史文件溯源；未证实指标错误 |
| 训练 task loss 梯度正常 | PASS | 两架构分别两个真实 batch4 forward；仅 autograd，三 latent Jacobian 非零，去 asso 后仍到 Scheduler；§7 / T | 未复现失效 |
| train/eval 差异符合预期 | PASS | 原生 reasoner 与冻结 HEAD 一致：train 几何匹配、eval learned affinity；原生检测/回归仍提供梯度；§7 / T | 有代理目标限制，非已确认 Bug |

证据：D = [downstream_evidence.json](../../audits/v2_bug_audit/downstream_evidence.json)，P = [protocol_evidence.json](../../audits/v2_bug_audit/protocol_evidence.json)，T = [training_evidence.json](../../audits/v2_bug_audit/training_evidence.json)，E = [evaluation_evidence.json](../../audits/v2_bug_audit/evaluation_evidence.json)。每份证据记录输入、期望、实际输出、PASS/未验证及源码位置；D 以 shape/dtype/逐 tensor SHA 记录完整输入，而非只比较一组缓存值。

## 3. 真实执行链路

正式入口来自 `logs/v2/train.log:2` 的实际命令，而非推测：

```text
docker exec -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main
python /mnt/d/CoopTrack_Audit/scripts/v2_run.py
--inside --root /mnt/d/CoopTrack_Audit --phase all --resume
```

`scripts/v2_run.py:179` 按冻结协议执行 8 个周期及 8 个自适应评估；`FINAL_PROTOCOL_MANIFEST.json` 的 8 份 V2 源码、main runtime/message、训练配置和最终 Scheduler checkpoint hash 与当前文件严格相等。原生仓库 HEAD 为 `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`；R50 coop 权重实际 MD5 为 `12d4658b70f21931f3b4a793cd1bf033`。

| 环节与实际函数/源码行 | 输入及来源 | 输出及核查 |
|---|---|---|
| `main_runtime.py:157 native_manifest`；`v2_eval.py:170 V2TemporalReceiver.__call__` | 当前 cooperative `sample_idx_inf` →官方 infra token/scene/session/frame/timestamp；`main_runtime.py:139 fields(token.pkl)` | 原始预测字段：三 latent各`N×256` FP32，ref`N×3`，box`N×10`，predicted ID `N` int64；浮点导出ID仅在可精确恢复时转换 |
| `main_message.py:190 scheduler_features`；`v2_message.py:10 TemporalCache.prepare`；`v2_scheduler.py:27 forward` | 当前 INT8 roundtrip、最后接收 latent、差分/elapsed/预测几何；`N×2324`，预算标量来自冻结 train 统计 | `N` logits/probability；sender 可看当前 latent，但它不是下游 receiver 输入 |
| `v2_message.py:45 select_updates` / `:57 periodic_updates` | probability/force，或 last actual UPDATE 后 elapsed | bool `N` flags；可选阈值`.5`，只无 cache 强制，周期按 elapsed；无 V1 Top-K/TTL |
| `main_message.py:48 pack` | 原始当前预测字段、metadata、固定 flags | 每帧ID、flags、ref/box；仅 UPDATE 行携带三 INT8 latent和三个FP32 scale；无额外 raw/history/GT 字段 |
| `main_message.py:117 ReceiverCache.unpack`；`v2_message.py:25 commit` | 有序 packet及预期当前frame；scene/session隔离的ID cache | UPDATE为实际解码，REUSE为最后收到三 latent；缓存detach+clone；当前ref/box从当帧packet取；sender mirror与receiver一致 |
| `v2_eval.py:210`、`:229`、`:271` | decoded五字段 → `results.update(decoded)` → native pipeline；原 LoadInfInformation恰替换一次 | `assert_loaded_queries` 对loader实际五字段逐tensor相等；不会再次使用原LoadInfInformation的当前高维输入 |
| `cooptrack/detectors/cooptrack.py:272 forward_test`、`:1207 _forward_single_frame_inference` | 本车当前image/pose/time、五个通信kwargs | `kwargs[field][0][0]`→`crossview_alignment`→`_init_inf_tracks`，不直接读路端缓存文件；`read_track_query_file_root`只在构造时存为属性 |
| `cooptrack/modules/latent_transformation.py:142 forward`；`detectors/cooptrack.py:450 _init_inf_tracks` | 收到的三 latent、当前预测geom、该官方pair校准 | 同一几何mask保留三 latent；收到值经过冻结投影/坐标变换，初始化clone到inf cache/history；不取当前 raw latent替换 |
| `spatial_temporal_reason.py:222 aggregation`、`:250 _query_fusion` | inf cache_feats/embed/motion及当前geom、本车Instances；通常本车`900×256` | learned/geometry association，再三组MLP融合及unmatched补充；D捕获全部双方字段和匹配indices；原生 query retention Top-K不选择wire UPDATE |
| `v2_eval.py:275`、`:288`、`:35 car_evaluation` | 该方法本轮outputs、同顺序dataset infos；car-only原生NuScenes evaluator | 3316 cooperative tokens、该run独立预测及格式化det/tracking JSON，mAP/AMOTA/AMOTP/MOTA/IDS；归档lossless；E验证现存内容链，历史SHA绑定另列未验证 |

上表原生路径相对于 `code/CoopTrack/projects/mmdet3d_plugin/`。训练 `v2_train.py:137 alignment_input` 在独立样本上下文中将当前loader五字段替换成实际解码的 ST student；几何必须与当前 sender相等。不会把loader原始高维值直接供任务 loss。

## 4. REUSE 与原生下游受控复现

新独立测试：`audits/v2_bug_audit/downstream_audit.py`。读取既有 `coop_val.py`，不调用会写配置的 `make_config`；模型和时序状态为审计独立副本，flags直接固定，构造后重置 seed0，`save_track_query=False`。该两帧从局部空时序状态启动，不等同完整正式P32前史。

有效片段：scene0003/session0，cooperative `000925→000926`，infra `000958→000959`，原生 frame55→56，连续 predicted ID152。两帧各18个sender query（每组三 latent `18×256`）；原生mask各保留1个，正是ID152，第二帧实际融合匹配1对。车端完整 association 输入还包含本车900个query及历史字段。

在5个分支各执行同样的2帧真实R50 forward：baseline REUSE、扰动 REUSE、baseline UPDATE、扰动 UPDATE、GT扰动 REUSE。当前三 latent仅在独立副本中把ID152改为 `linspace(-7,9,256)+{0,3,6}`；prefix相同、几何/校准相同。

| 断言 | 期望 | 实际 | 结果 |
|---|---|---|---|
| REUSE flags固定且三 current latent确实不同 | `.False`，三组sender hash均变 | 第二帧18个flags均False；三组current全改变 | PASS |
| REUSE packet及receiver | packet不变，三 latent等于frame55最后实际解码 | packet同SHA `e4579515…40c395`，2055B；全部三 latent bit-exact旧值 | PASS |
| 完整真实下游输入 | alignment输入/校准/输出、inf初始化全部字段、reasoner、association双方全部字段、fusion输入/indices不变 | 所有shape/dtype/逐tensor SHA相等；最终预测也完全相等 | PASS |
| UPDATE对照 | 只更新ID152，反映当前扰动且误差≤scale/2 | 2835B，恰多780B；三个decoded均等于INT8 roundtrip且不同于baseline；alignment/init/association/fusion输入改变 | PASS |
| UPDATE量化误差 | 每组以自己的量化尺度判断 | feat `.0353875≤.0354331`；embed `.0471826≤.0472441`；motion `.0590401≤.0590551` | PASS |
| 当前低维几何 | REUSE也取frame56当前预测 | 全18行ref/box与当前raw精确相等；它们与prefix值不同 | PASS |
| GT控制 | gt不参与inference输入或输出 | 两帧GT均非空，boxes `31×9`、labels `31`、inds `1×31`；填12345/123后完整hooks和两帧predictions仍bit-exact | PASS |

首次尝试scene前两帧时，所有目标在原生范围mask外，不足以验证融合控制；因此按预测几何选择了确实进入融合的上述片段，没有把empty输入相等当作有效证明。全val纯CPU mask核验：2981/3316帧非空，335帧为空；28059/44329 query保留（63.30%）。这描述实际几何覆盖，不证明性能现象的原因。

## 5. 周期、缓存、通信与时间

P证据使用真实scene0003的70帧raw cache回放：ID869持续存在，P32更新`[0,32,64]`，INIT_ONLY仅`[0]`；所有REUSE三 latent等于最后实际接收值，UPDATE等于真实INT8解码，几何逐帧当前，flags/bytes与既有日志精确一致。新ID946初始化；确定性多目标、重排、交替UPDATE/REUSE、消失后frame40再出现、ID switch、same ID跨scene/session、两个batch lane及引用隔离均按协议通过；所选70帧没有真实消失再现，不能把合成测试称为该真实事件观测。

实际scene0017的infra `003547→003565` 间隔1.843382s，session4→5；native_frame29→30，而通信session frame29→0。共同ID27正确重新初始化。16份日志的全部22次reset首帧所有query都是init；缓存以 `(scene,session)`隔离，再按predicted ID查找，无TTL。预测ID本身的错跟/复用属于身份算法局限，不等同cache键错绑。

应用层wire公式独立拆解为 `16B prefix + JSON header + 61N + 780U`；61N包括ID 8N、flag N、当前FP32几何52N；780U为三组`256×INT8+4B scale`。固定header/输入仅改UPDATE行时，差分恰780B；REUSE行latent强扰动、额外matched_gt/history/future字段修改均不改变packet。16×3316行的mean/P95/total及sample对应关系重算通过。

| 方法 | Mandatory总B | Latent UPDATE总B | Total B | mean B/frame | P95 B/frame |
|---|---:|---:|---:|---:|---:|
| INIT_ONLY | 5,872,146 | 1,214,460 | 7,086,606 | 2137.094692 | 3801 |
| P32 | 5,872,146 | 1,961,700 | 7,833,846 | 2362.438480 | 4238 |
| MLP B1 | 5,872,146 | 2,851,680 | 8,723,826 | 2630.828106 | 4825 |

16个新方法具有相同逐帧mandatory布局，mean mandatory=`1770.852232`B/frame；INIT_ONLY 1557×780即全部latent，P32还有958次optional refresh。统计是应用payload，不包括运输层、重传、网络链路总流量。

同session3294个真实相邻infra dt平均`.100701714s`，范围`.041133–.499930s`，没有一个恰100000µs。正式`bytes_per_second_at_10Hz`明确是名义 `mean×10`，P32为23624.384801B/s；P32“3.2s”也仅名义值，不能当严格墙钟周期。P证据单独定义排除每session首packet的时间戳诊断速率23010.822B/s，不替换原正式指标。

## 6. 评估内容检查与未闭合的溯源

E检查官方46 train scenes/7362 cooperative帧、21 val scenes/3316帧，scene及token不交叉，21个val scene和全部样本都进入每个run。当前基础设施输入来自官方当前配对token，非自行取next token；所有3316 pair的prepared metadata、image路径、timestamp及校准来源逐项核对。校准旋转/平移回算误差仅浮点精度。

原生info使用pointcloud timestamp，prepared NuScenes sample使用同vehicle帧image timestamp；存在继承的异步pair：infra−vehicle pointcloud时间范围−62181到+70789µs，1491对infra略晚。这是官方配对来源，不能宣称严格同步，也不是V2读取未来帧或未来GT。receiver/pack只取当前预测白名单字段；未来annotation关闭；GT在train仅用于原生监督标签，eval虽loader携带当前GT，但静态分支与§4动态干预均证明本例推理不消费GT。

每个新run独立读取锁定的epoch12 Scheduler路径/SHA；16份现存预测均为3316正确cooperative tokens。各自原始预测与归档det/tracking JSON在固定间隔、首非空和末帧使用实际原生formatter函数逐box回算，共32个det/tracking检查、40349个box，坐标/尺寸/旋转/速度最大误差0，score/class/ID一致；五指标与本run原生metrics_summary逐值相等，16套car-only evaluator配置完全相同。120个方法组合的完整tracking内容至少2555/3316帧不同，不仅比较gzip外壳hash，未发现意外整份重复预测。

**未验证项 E-01：16份原始预测的历史SHA绑定。** `scripts/v2_eval.py:122` 构建identity、`:131`读取resume、`:282`保存outputs、`:333`保存result；result及FINAL manifest只有路径和源码/协议/checkpoint identity，没有该次生成预测的内容hash。搜索 `logs/v2`、`reports/v2`、`experiments/v2/metrics`（包括final_summary/run_state）、`configs/main_experiment_v2`及预测目录非gz metadata，排除本次新建的审计报告，没有可恢复的正式生成时SHA。期望为“当前prediction hash==生成时记录hash”；实际右侧证据不存在，所以不能执行该assert。E记录本次实际文件/content SHA，状态仍未验证。缺少绑定不证明文件被替换，也不能据此认定某方法用了错误结果。

另一个历史溯源边界：native HEAD正确，但工作树有继承的空sender初始化及CPU矩阵inverse兼容patch，不能以HEAD冒充工作树全部字节一致。当前代码/diff已检查，reasoner与HEAD一致；native关键源码本次hash未变，但formal manifest未保存其工作树历史hash，无法补造。没有复现这些patch导致val错读/GT泄漏。

历史P1/P2/P4原有prediction SHA记录核验通过，其完整det/tracking配置与16个新方法相等；无需重跑。但历史CTKMSG01与新APTU0001及RNG重置位置不同：P1已有cost12161.645356B/frame，而从新布局计算全UPDATE成本12198.059710B/frame；不回填/重计旧值。实际同输入五字段的legacy decoder与新decoder相等，封包长度不同。历史高低指标对照只能作观察性参考，不能证明刷新频率的因果增益；即便同起始seed，后续原生状态分支也可能不同。

## 7. 训练、梯度、冻结与代理目标

T依据当前正式源码、两份final optimizer state及新受控原生梯度检查。`v2_train.py:605` AdamW只含`scheduler.parameters()`；正式MLP16/16、Transformer36/36 optimizer张量的step均23439，moment非零有限，12epoch完整。final与按相同构造/seed复现的fresh初始化所有参数tensor都改变；没有原始initial checkpoint，不将复现初始化当作历史初始hash。

`main_runtime.py:126`冻结base；`v2_train.py:202`进入任务train模式但所有BN eval，`:216`每epoch验证base参数版本/无grad及BN buffers不变。新模型副本也通过这些断言；BN.training计数0；审计Scheduler参数不变、optimizer steps=0。

实际ST为`v2_scheduler.py:48`：`D.detach() + (surrogate-surrogate.detach())`，surrogate=`p*C+(1-p)*L`，forced概率取1，C/L均detach；这是hard decoder前向的等价ST，而非只在配置里声明`hard+p-p.detach()`。新CPU真实codec混合REUSE/UPDATE/init前向五字段逐tensor等于D，optional logits梯度与解析`p(1-p)(C-L)`相符，forced及C/L/D没有梯度。

新原生控制每个架构仅cold和下一帧两个真实batch4 forward，scenes0002/0004/0018/0022，第二batch tokens `[000411,001084,003493,004263]`，alignment后query `[0,1,2,0]`；非全empty。只运行`autograd.grad`，未调用正式`run_batch`、correctness_check、train、optimizer或checkpoint写入。

| 第二batch独立梯度范数 | MLP | Transformer |
|---|---:|---:|
| 原生task→Scheduler | 6.6209023e-7 | 5.8874217e-4 |
| 去asso后的原生检测/回归→Scheduler | 6.6486481e-7 | 7.3256741e-4 |
| rate→Scheduler | .0237372 | .0119071 |

三组student latent的原生task Jacobian都非零。由此排除“仅rate反传”“整个task被no_grad/float截断”的本次实现假设；MLP在这一短片段task信号很弱，不足以判断全训练的梯度比例或解释所有指标。

`v2_scheduler.py:62` 和 `v2_train.py:376` 的实际公式为：

```text
unavoidable = mandatory + 780 * forced_count
S_soft = unavoidable + 780 * sum(p_optional)
B_feasible = max(B, unavoidable)
L_rate = [ReLU(S_soft - B_feasible) / train_full_int8_mean]^2
L_total = L_native_task + 1.0 * L_rate
S_hard = len(actual_packet)
```

forced和padding无rate梯度；新below/above-budget解析测试通过。原生内部loss/weights与原配置一致，association focal实际alpha=-1、gamma=1、weight10；检测focal另为alpha=.25/gamma2/weight2。V2只把外层stream reducer改为等权mean（`v2_train.py:162`）；不能说整个native batch归约完全未变。没有额外association-preserving teacher loss、预算Top-K、TTL或max_staleness；soft与实际hard发送量分开记录。

`spatial_temporal_reason.py:238`的train先计算可导learned affinity及原生监督loss，再以geometry Hungarian匹配作融合；eval使用sigmoid affinity后detach的离散Hungarian。该文件与冻结官方HEAD一致，不是V2新增差异。新原生分项梯度证明融合后的检测/回归仍参与Scheduler目标；但离散推理匹配并没有精确可导，不能声称优化了实际AMOTA或精确inference reward。

`v2_message.py:35`、`main_message.py:178`及原生Instances/prev_bev跨帧detach：后续帧loss对前一个决策logit的受控gradient为None，当前决策非零。只有当前帧代理信用，没有BPTT或长期回报优化；这是明确的设计限制。

## 8. 影响、解释与剩余范围

没有确认BUG-ID、失败的正式实现assert或需要自动修复的位置。当前执行行为检查通过，但E-01阻止对全部正式结果作完整历史有效性认证；受影响的溯源范围是全部16个新V2方法，历史P1/P2/P4已有hash记录通过。没有证据要求重训；是否需要重新评估尚不能判定，本次均未执行。

后续首先寻找独立可信的运行时hash清单或同期归档/备份，以闭合E-01及native工作树身份；从当前文件补写旧manifest无法证明历史。没有修改源码、重新生成指标或开启新实验，是否进入后续验证由用户决定。

在未复现实现Bug的检查范围内，P32确实可在当前几何及本车时序模型上使用较旧latent；INIT_ONLY也保留初始化时真实收到的协同特征，仍逐帧接收当前预测框及位置。少刷新可能改变噪声、关联和原生状态路径；历史RNG/封包差异、硬门限和soft预算差异、冻结base、弱task信号、train/eval匹配代理及无跨帧信用都可能限制学习效果。这些是可兼容观察结果的解释，不是已证明的性能因果机制，也不能据此宣布全局无Bug。

覆盖边界：原生干预只验证固定真实两帧；周期codec回放覆盖70帧，16套完整通信与现存预测元数据另做全量检查；格式化坐标是固定抽样而非重新评价所有box。未做多seed统计、完整val重跑、网络故障或新模型实验，未覆盖所有可能的身份错误。用户允许的少量同协议P1/P32指标诊断属于可选项，本次没有以片段指标替代正式结果。

## 9. 复现与不变性

在继承容器中、没有其他GPU任务时依次执行；四个脚本只写各自audit evidence：

```powershell
docker exec -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main python /mnt/d/CoopTrack_Audit/audits/v2_bug_audit/protocol_audit.py
docker exec -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main python /mnt/d/CoopTrack_Audit/audits/v2_bug_audit/downstream_audit.py
docker exec -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main python /mnt/d/CoopTrack_Audit/audits/v2_bug_audit/training_audit.py --native-gradient
docker exec -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main python /mnt/d/CoopTrack_Audit/audits/v2_bug_audit/evaluation_audit.py --root /mnt/d/CoopTrack_Audit
```

D的190个既有保护文件（包括configs/weights/results/旧报告/native代码）前后SHA相同，模型全部params/buffers hash相同；实际选例raw cache前后SHA相同。P全3316 val cache combined SHA前后同为`48d7535e8975949db572556b89d8cf4e3f0bf2d5863f05fa8b4076f52c639a23`，另119个保护文件不变。T/E各自的保护hash亦见证据。未删除任何已有文件或保留的pyc。
