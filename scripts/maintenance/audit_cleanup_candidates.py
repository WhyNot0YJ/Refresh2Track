"""Read-only classification of an existing inventory; never deletes project files."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'audits/cleanup_review_20261012'
SAFE, CONDITIONAL, KEEP = '可清理', '条件清理', '保留'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(stream):
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def verify_duplicates():
    rows = []
    for directory in sorted((ROOT / 'experiments/v4_budget_free/validation').glob('epoch*')):
        result_path = directory / 'result.json'
        inputs_path = directory / 'evaluator_inputs.json'
        if not result_path.is_file() or not inputs_path.is_file():
            continue
        result = read(result_path)
        if result.get('status') != 'PASS' or result.get('sample_count') != 3316:
            continue
        inputs = read(inputs_path)
        for name in ('predictions', 'tracking', 'detection'):
            entry = inputs[name]
            source = directory / Path(entry['path']).name
            archive = directory / Path(entry['archive']).name
            before = (source.stat(), archive.stat())
            with source.open('rb') as stream:
                source_sha, source_bytes = sha(stream)
            with archive.open('rb') as stream:
                archive_sha, archive_bytes = sha(stream)
            with gzip.open(archive, 'rb') as stream:
                decoded_sha, decoded_bytes = sha(stream)
            after = (source.stat(), archive.stat())
            unchanged = all(a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns for a, b in zip(before, after))
            match = (unchanged and source_sha == decoded_sha == entry['sha256']
                     and archive_sha == entry['archive_sha256'] and source_bytes == decoded_bytes)
            rows.append({'path': source.relative_to(ROOT).as_posix(),
                         'archive': archive.relative_to(ROOT).as_posix(),
                         'raw_bytes': source_bytes, 'archive_bytes': archive_bytes,
                         'source_sha256': source_sha, 'decoded_sha256': decoded_sha,
                         'archive_sha256': archive_sha, 'status': 'VERIFIED' if match else 'FAIL',
                         'requires_raw_at_resume': True})
        print('verified ' + directory.name, flush=True)
    (OUT / 'verified_duplicates.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    assert rows and all(row['status'] == 'VERIFIED' for row in rows)
    print(json.dumps({'pairs': len(rows), 'duplicate_raw_bytes': sum(row['raw_bytes'] for row in rows)}))


STAGE1_DETAIL = {'cache_inf.json', 'inf_cache_all.json', 'coop_coop.json', 'inf_first_two.json', 'profile_coop.json'}
STAGE2_DETAIL = {'cache_dependencies.json', 'cpu_pipeline_probe.json', 'nonrequired_paths.jsonl'}
STAGE3_DETAIL = {'temporal_pairs.jsonl', 'cache_train_mapping.json', 'execution.json', 'plan.json', 'state.json', 'coop_val.py', 'inf_val.py'}
ONE_SHOT = {'scripts/cleanup_reviewed.ps1', 'scripts/v2_reset_cleanup.ps1',
            'scripts/maintenance/v4_retirement_handoff.py', 'scripts/maintenance/retire_value_v1.ps1',
            'scripts/maintenance/check_h_writers.py', 'scripts/maintenance/check_h_writers.sh',
            'scripts/maintenance/compact_docker_data.ps1', 'scripts/maintenance/compact_docker_data.diskpart.txt'}
STAGE_REASONS = {
    'checkpoint_download_resources.json': '官方 checkpoint 下载 URL/hash 与作者来源的小型清单，保留以证明权重来源。',
    'converter_gate.json': 'Example converter 与 GT 表验收；保留小型正式验收记录，避免 Stage1 报告失去转换证据。',
    'docker_cleanup_verification.json': '早期环境迁移/清理结果，仅 885 B；保留用于解释宿主 prefix 已删除和 Docker 环境继承。',
    'docker_dataset_mount_probe.json': '旧 smoke 的数据挂载诊断，约 10 KiB；体积小，保留环境验收上下文。',
    'host_installation_cleanup_review.json': '宿主安装范围核验，证明未污染其他环境；小型历史记录保留。',
    'local_cleanup_manifest.json': '宿主 prefix/cache 清理账本，仍被 V4 保护清单锁定，保留。',
    'official_example_eval.json': '249 帧 Example 真实 evaluator 指标、命令与内存，替代大 smoke 预测的轻量验收摘要。',
    'official_example_eval_memory.json': 'Example 官方评测进程峰值内存，285 B；保留 4060 Ti 环境验收证据。',
    'resource_download_manifest.json': '依赖、CUDA header 和官方资源来源及哈希清单，独立重建/溯源需要，保留。',
    'stage1_final_gate_checks.json': 'Stage1 最终 PASS 条目和证据索引；保留阶段结论，不作为过时状态文件删除。',
    'archive_hashes.json': '本地原始归档哈希，小型溯源记录，删除无明显空间收益。',
    'archive_integrity.json': '分卷载荷 CRC、Headers Error 与独立 payload 校验结论；正式 raw 数据来源验收证据。',
    'archives_manifest.json': '原始归档、分组、来源和哈希清单；后续备份和原始数据恢复需要。',
    'context_runtime_stdout_diagnostic.json': '数据编排 JSON 解析失败的原始诊断；仅 1.7 KiB，保留以解释当时修复和边界。',
    'conversion_completion.json': '六次官方转换完成结果；保留正式 infos 的生成账本。',
    'conversion_counts.json': '三侧转换计数；保留 train/val 规模和排除范围的核对依据。',
    'excluded_records.jsonl': '官方 test/invalid/coop 配对排除逐条原因；证明未任意删帧，必须保留，建议压缩而不删除内容。',
    'full_trainval_integrity.json': '46 train / 21 val、infos、GT 表与无交集核验；是完整数据验收依据。',
    'image_decode.json': '23527 张 JPEG 的真实解码验收摘要，保留。',
    'image_decode_manifest.jsonl': '全图像解码逐帧账本及 SHA；仍受保护绑定，保留，压缩须另行迁移引用。',
    'label_processing_changes.jsonl': '官方标签修正和 track_id 去重的逐条 ledger，prepared_integrity 记录其 SHA；保留原始修正证据，可规划无损压缩。',
    'official_selection.json': '67 个非 test scene 的官方选集信息；保留以证明没有扩大/缩小正式 split。',
    'prepared_integrity.json': 'prepared 标签修正/元数据身份的权威验收版本，保留；与 prepared_completion 逐字节相同。',
    'raw_integrity.json': '完整 raw 来源与布局验收，仅 692 B；保留。',
    'storage_plan.json': '归档/解压空间的小型历史账本，仅 1.8 KiB；保留数据准备来源。',
    'paths_inherited.json': 'Stage1 到 Stage2 的历史路径继承记录；保留溯源，旧路径不表示目前仍可使用。',
    'prepare_data_execution.json': '官方转换的真实命令、退出状态和数据生成执行 ledger；保留。',
    'stage2_context.json': '继承环境、模型与数据身份的小型阶段上下文；保留。',
    'train_smoke_manifest.json': '从完整 train 派生的 scene0000/32 帧梯度探针映射；仍受保护绑定，保留，避免混入 val/截断监督。',
    'stage2_gate_checks.json': 'FULL_DATA_READY 最终验收与未运行阶段边界；保留数据准备结论。',
    'fp32_regression.json': '完整 val 的 FP32 消息无损回归验收，证明 codec 未改变基线，保留。',
    'stage3_gate_checks.json': 'Stage3 最终诊断 PASS 权威小型汇总；保留，与 state.json 字节一致。',
    'temporal.json': '42772 对 Track Query 时序冗余的聚合统计；保留方法动机，明示不能由此证明跳过传输无损。',
    'train_receiver_audit.json': '32 帧真实 train receiver 与 sender 映射核对，保留。',
    'wire_audit.json': '五个接收字段、字节口径、真实 FP32 roundtrip 与未发送 GT 的协议验收；保留。',
    'wire_audit_manifest.jsonl': '3316 val+32 train 的逐帧无损消息账本，wire_audit 锁定其 SHA，当前 V4 保护记录也锁定；保留。',
}


def base_decision(path, kind='file', is_link=False):
    """Operational decisions; protection bindings are applied separately."""
    name = Path(path).name
    parts = path.split('/')
    if is_link and path.startswith('datasets/prepared/'):
        return KEEP, '活动数据链接', 'prepared 链接指向 /data/spd_raw；图像/标定由当前训练读取，不能删除链接或其目标。'
    if path == 'code/CoopTrack/datasets/V2X-Seq-SPD-Example':
        return SAFE, 'Example 旧诊断链接', '仅为旧 Example 的源码树链接，当前正式配置已显式使用完整数据路径；只可删除该链接本身，不跟随其目标删除。'
    if path.startswith(('data/V2X-Seq-SPD-Example/', 'data_infos/V2X-Seq-SPD-Example/')):
        return SAFE, 'Example 旧诊断', '当前 V4 使用完整 train/val infos；Example smoke 已结束，其视图链接在当前容器失效，仅影响旧 Example 重跑。保留 Stage1 摘要。'
    if path.startswith('data/wheels/torch-cu111-parts/'):
        if name == 'manifest.json':
            return KEEP, '下载来源记录', '完整 Torch wheel 保留；该小清单保存分段范围与下载来源，无需再保留逐分段 headers/log。'
        return SAFE, '下载中间记录', '分块 payload 已在上次清理中删除；该 headers/log 不参与安装、重建或训练，完整 wheel 与 manifest 已保留。'
    if path.startswith(('data/wheels/', 'data/cuda_packages/')):
        return CONDITIONAL, '离线重建资源', '当前训练不重新安装，但 docker/rebuild-image.sh 明确需要这些精确 wheel/archive；仅在备份或放弃离线重建并验证替代来源后可移出。'
    if path.startswith('code/CoopTrack/output/') or (path.startswith('code/CoopTrack/test/stage1_coop/') and name in {'results_nusc.json', 'results_nusc_det.json'}):
        return SAFE, 'Stage1 smoke 原始输出', 'Example 的旧推理/评测大文件；不用于正式结果或当前训练，官方 Example 指标已在 stage1/official_example_eval.json 保留。清理会放弃该 smoke 的原始预测重评。'
    if path.startswith('code/CoopTrack/test/stage1_coop/'):
        return KEEP, 'Stage1 smoke 指标', '小型原始 det/track metrics 作为 smoke 验收依据保留；没有必要为约 0.22 MiB 丢掉指标细节。'
    if path.startswith('datasets/V2X-Seq-SPD-raw/'):
        if path.endswith('.pcd'):
            return CONDITIONAL, '原始点云', '当前 camera-only pipeline 不读 PCD，但它属于完整原始数据，prepared/velodyne 链接及原始完整性记录引用它；先核验外部完整备份和重建要求，不能直接按旧数据删除。'
        return KEEP, '正式原始数据', 'prepared 图像和标定链接依赖 raw；官方标签、映射、地图与身份记录用于数据完整性和转换溯源。'
    if path.startswith('datasets/prepared/'):
        return KEEP, '正式 prepared 数据', '当前 V4 data_root=/data/spd_prepared；修正标签、GT 表和指向 raw 的链接是正式训练/评测输入。'
    if path.startswith('data_infos/V2X-Seq-SPD-CoopTrack-full/'):
        if '/vehicle-side/' in path:
            return CONDITIONAL, '完整 vehicle infos', 'V4 当前使用 cooperative 与 infrastructure infos；vehicle-only infos 暂不进入当前入口，但完整三侧转换核验和未来 vehicle baseline 需要。先更新全量完整性/历史复现范围。'
        return KEEP, '正式 infos', 'cooperative infos 驱动训练/验证，infrastructure infos 定义 cache 顺序和参考统计；缺失会使当前入口失败。'
    if path.startswith('query_cache/'):
        return KEEP, '共享路侧 query cache', '当前训练读取 train 中每帧 pkl；验证及 Periodic 比较读取 val；重新生成需要原生推理并重新锁定身份，不能当临时缓存删除。'
    if path.startswith('checkpoints/'):
        return KEEP, '官方 R50 权重', '原生协同/路侧模型加载和数据缓存身份依赖官方权重；V4 仅训练 scheduler，仍需这些冻结权重。'
    if path.startswith('experiments/v4_budget_free/'):
        return KEEP, '活动 V4 实验', 'V4 正在训练；checkpoint、run.lock、manifest、历史、验证输入及审计迁移记录均服务于恢复、选模或完整性核验。'
    if path.startswith('logs/v4_budget_free/'):
        return KEEP, '活动 V4 日志', '协调器和 worker 仍在写入；当前不得截断、删除或压缩正在写入的日志。'
    if path.startswith('experiments/main/checkpoints/'):
        if name.startswith('epoch_') and name != 'epoch_12.pth':
            return CONDITIONAL, 'V1 中间 epoch 权重', 'V1 已完成，正式结果使用 final/latest；早期 epoch 仅用于回溯训练。保留 epoch12/latest 并迁移保护清单后可删除早期权重。'
        return KEEP, 'V1 最终权重', '正式 V1 结果及历史恢复基准；应保留最终 epoch12/latest，重复副本仅在哈希一致并修订引用后处理。'
    if path.startswith('experiments/main/predictions/') and name in {'predictions.pkl', 'results_nusc.json', 'results_nusc_det.json'}:
        return CONDITIONAL, 'V1 未压缩正式预测', '原始正式证据不应直接删除；可无损 gzip 并核验原字节 SHA，保留 metrics/communication，迁移重评/恢复路径后删原件。'
    if path.startswith(('experiments/main/', 'experiments/v2/')):
        return KEEP, '正式 V1 V2 结果', 'V1/V2 的正式指标、最终权重、预算曲线、三 seed 稳健性及消融依据；V4 references 直接读取部分 V2 结果，已压缩预测无需再次删除。'
    if path.startswith(('logs/main/', 'logs/v2/')):
        return CONDITIONAL, '已结束训练日志', 'V1/V2 已结束；可无损压缩大日志保留全量故障与训练证据。不要仅因历史实验已结束就删除唯一日志。'
    if path.startswith('artifacts/core/stage1/'):
        if name in STAGE1_DETAIL:
            return CONDITIONAL, 'Stage1 逐帧诊断', 'Example 阶段已结束；保留 final gate、eval、converter 与资源来源摘要后，可压缩或移出逐帧 tensor/cache/profile 细节；会降低旧 smoke 的逐帧追溯能力。'
        return KEEP, 'Stage1 验收溯源', '保留环境/资源来源、转换、官方 Example 指标及清理记录，体积小；为 Stage1 最终报告和后续阶段继承提供证据。'
    if path.startswith('artifacts/core/stage2/'):
        if name in STAGE2_DETAIL:
            return CONDITIONAL, 'Stage2 冗长诊断', 'CPU shape/cache 计划或未要求点云 bin 的逐路径清单不驱动现行 V4；保留完整性和排除账本后可压缩/移出细节，历史重核会受限。'
        return KEEP, 'Stage2 数据准备证据', '归档 CRC/hash、raw/prepared、三侧 train/val 完整性、官方修正与排除 ledger 证明正式数据来源；阶段结束不意味着证据已失效。'
    if path.startswith('artifacts/core/stage3/'):
        if name in STAGE3_DETAIL:
            return CONDITIONAL, 'Stage3 中间诊断', '已完成阶段的计划/状态/生成配置或时序逐对记录；保留 fp32 回归、wire、梯度探针、temporal 聚合与 final gate 后可压缩/移出，详细诊断重核会受限。'
        return KEEP, 'Stage3 协议验收', 'FP32 无损回归、wire 字节账本、训练梯度/冻结检查及最终 gate 支撑后续消息和训练协议。'
    if path.startswith('artifacts/core/communication_baselines/'):
        if name in {'results_nusc.json', 'results_nusc_det.json', 'results.pkl', 'predictions.pkl'}:
            return CONDITIONAL, '原始通信基线预测', '完整 FP32/FP16/INT8/CONF/Random 论文基线；可 gzip 保留原始字节，不能用汇总指标替代原始预测。FEAT_INT8 的重评也引用原输入。'
        return KEEP, '通信基线指标', 'V1/V2/V4 基线读取真实 result/communication；必须保留小型指标与完整统计，不能删除整目录。'
    if path.startswith('artifacts/core/ids_diagnosis/'):
        if name in {'results_nusc.json', 'results_nusc_det.json', 'predictions.pkl', 'results.pkl'} or path.endswith('.jsonl'):
            return CONDITIONAL, 'IDS 逐帧证据', '重复 seed、原生/修复路径及 ID 生命周期原始诊断；可无损压缩并维护报告引用，删除会丢失随机性和 IDS 根因重核能力。'
        return KEEP, 'IDS 诊断结论', '保留 seed/生命周期差异、patch 对照、原评估输入身份和最终结论；这些证据解释官方 AMOTA/IDS 数值差异。'
    if path.startswith('artifacts/core/'):
        return KEEP, '核心身份与对齐证据', 'FULL_DATA_READY、full_protocol、checkpoint/cache identity、R50 car-only 对齐及 Docker provenance 为正式输入和历史基线的权威记录。'
    if path in ONE_SHOT:
        return (SAFE if path.endswith('v4_retirement_handoff.py') else CONDITIONAL), '一次性维护脚本', '对应旧清理/交接已完成，不是正式训练导入；保存执行结果后可移出，但必须先核对保护清单与报告引用，避免未来误执行旧清理目标。'
    if path == 'docker/build-image.sh' or path in {'docker/Dockerfile', 'docker/Dockerfile.runtime-fix'}:
        return CONDITIONAL, '旧环境迁移配方', 'build-image.sh 依赖已移除的宿主 prefix，独立重建改用 Dockerfile.rebuild；保留新版配方和历史构建身份后可移出旧配方，但当前保护清单仍可能绑定它。'
    if path.startswith('code/'):
        if '.git' in parts:
            return KEEP, '运行时 Git 身份', '当前 V4 的 merge/resume 检查读取实际 HEAD；删 .git 会使身份核验失败，且丢失已有合并与兼容补丁历史。'
        if path.endswith('.so') or '/mmdet3d/' in path:
            return KEEP, '原生算子与源码', '实际 mmdet3d 导入路径位于 code/mmdetection3d；源码树内 .so 正被原生运行依赖，旧 build 副本已在此前清理。'
        return KEEP, '原生仓库与配置', '官方源码、configs、tools、许可证及 Git 对应文件应保留；多项被冻结 source hashes 绑定，整个 code 约 57 MiB，裁剪收益小且损伤重建。'
    if path.startswith(('configs/', 'scripts/', 'docker/', 'patches/')):
        return KEEP, '运行和重建代码', 'V4 复用 main_runtime/v2_train/v2_message 等旧入口模块；配置、锁、probe 和 inverse 补丁保留，不能仅按旧命名判断未用。'
    if path.startswith('reports/') or path.endswith('.md'):
        return KEEP, '轻量研究文档', '保留阶段最终报告、正式结果、退役原因与历史授权范围；体积小，删除不能显著回收空间且丢失研究解释。'
    if path.startswith('audits/'):
        return KEEP, '审计证据', 'Bug 审计、既有清理清单及保护快照用于解释历史变动和安全核验；保留已压缩清单，不作为一般临时缓存。'
    if name == '.gitignore':
        return KEEP, '版本控制规则', '忽略规则防止数据、权重、编译产物进入版本库；被忽略不等于内容可删除。'
    return KEEP, '用途未完全确认', '未找到足以支持删除的具体证据，默认保留；不得把未发现静态引用等同于无用途。'


def classify():
    record = read(ROOT / 'experiments/v4_budget_free/audit/v3_retirement_transition.json')
    protected = record['protected_shared_files']
    frozen = read(ROOT / 'experiments/v4_budget_free/manifest.json')['source_hashes']
    links = {row['path']: row for row in read(OUT / 'links.json')}
    duplicates = {row['path']: row for row in read(OUT / 'verified_duplicates.json') if row['status'] == 'VERIFIED'}
    stage_duplicates = {row['left']: row for row in read(OUT / 'stage_duplicate_checks.json') if row['equal']}
    with gzip.open(OUT / 'inventory_raw.csv.gz', 'rt', encoding='utf-8') as stream:
        inventory = list(csv.DictReader(stream))
    files, directories = [], []
    for row in inventory:
        if row['kind'] == 'directory':
            directories.append(row['path'])
            continue
        path = row['path']
        decision, category, reason = base_decision(path, is_link=path in links)
        if path.startswith(('artifacts/core/stage1/', 'artifacts/core/stage2/', 'artifacts/core/stage3/')) and Path(path).name in STAGE_REASONS:
            reason = STAGE_REASONS[Path(path).name]
        if path == 'artifacts/core/stage3/train_probe/result.json':
            reason = '真实 train32 原生 task loss 到接口梯度、冻结参数及显存探针结果，是后续训练可行性证据，保留。'
        if path in stage_duplicates:
            decision, category = CONDITIONAL, '阶段重复汇总'
            reason = '与 ' + stage_duplicates[path]['right'] + ' 的完整 SHA256 一致；保留后者，先迁移报告/验收引用后可删本副本。'
        if path in duplicates:
            decision, category = CONDITIONAL, 'V4 已验证重复原件'
            reason = '原文件与 gzip 解压内容 SHA256 一致；evaluate.py 恢复分支仍强制读 raw 和 archive 两份。仅在 V4 全部结束或迁移恢复校验并锁定维护改动后，保留 gzip 删除 raw。'
        guards = []
        if path in protected:
            guards.append('V4 protected_shared_files')
        if path in frozen:
            guards.append('V4 frozen source_hashes')
        if guards:
            reason += ' 当前还被 ' + '、'.join(guards) + ' 锁定，删除/压缩前必须显式迁移保护绑定，否则 V4 核验失败。'
            if decision == SAFE:
                decision = CONDITIONAL
        files.append({'path': path, 'kind': 'link' if path in links else 'file',
                      'bytes': int(row['bytes']), 'mtime_ns': row['mtime_ns'],
                      'decision': decision, 'category': category, 'reason': reason,
                      'protection': '; '.join(guards),
                      'link_target': links.get(path, {}).get('target', ''),
                      'link_target_exists': links.get(path, {}).get('target_exists', ''),
                      'verified_duplicate_archive': duplicates.get(path, {}).get('archive', '')})
    fields = list(files[0])
    with gzip.open(OUT / 'files.csv.gz', 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(files)
    counts = defaultdict(Counter)
    sizes = Counter()
    for row in files:
        for parent in Path(row['path']).parents:
            name = parent.as_posix()
            if name == '.':
                continue
            counts[name][row['decision']] += 1
            sizes[name] += row['bytes']
    directory_rows = []
    for name in directories:
        choices = counts[name]
        if not choices:
            structural = '.git' in Path(name).parts or name.startswith('datasets/')
            decision = KEEP if structural or name.startswith(('experiments/v4_budget_free', 'logs/v4_budget_free', 'configs/v4_budget_free')) else SAFE
            reason = ('Git 内部结构或正式数据布局目录，不能仅因为空删除。' if structural
                      else '活动实验预留目录，训练期间保留。' if decision == KEEP
                      else '目录树没有文件；执行时只在每个目录实际为空后移除，并复核没有 writer。')
        elif len(choices) == 1:
            decision = next(iter(choices))
            reason = '全部子项为' + decision + '；逐文件原因见 files.csv.gz，不能跳过子项保护要求。'
        else:
            decision = '混合，不能整删'
            reason = '同时包含需保留或有条件的子项；只按逐文件清单处理，不按目录名称批量删除。'
        directory_rows.append({'path': name, 'files': sum(choices.values()), 'bytes': sizes[name],
                               'decision': decision, 'reason': reason,
                               'safe_files': choices[SAFE], 'conditional_files': choices[CONDITIONAL], 'keep_files': choices[KEEP]})
    with (OUT / 'directories.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(directory_rows[0]))
        writer.writeheader()
        writer.writerows(directory_rows)
    with (OUT / 'stage_files.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in files if row['path'].startswith(('artifacts/core/stage1/', 'artifacts/core/stage2/', 'artifacts/core/stage3/')))
    with (OUT / 'safe_candidates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in files if row['decision'] == SAFE)
    groups = defaultdict(lambda: {'files': 0, 'bytes': 0, 'example': None})
    for row in files:
        key = row['decision'] + ' / ' + row['category']
        groups[key]['files'] += 1
        groups[key]['bytes'] += row['bytes']
        groups[key]['example'] = groups[key]['example'] or row['path']
    summary = {'snapshot_time_asia_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
               'scope': 'workspace inventory snapshot; this audit output directory excluded',
               'files': len(files), 'directories': len(directories), 'logical_bytes': sum(row['bytes'] for row in files),
               'decisions': {choice: {'files': sum(row['decision'] == choice for row in files),
                                     'bytes': sum(row['bytes'] for row in files if row['decision'] == choice)} for choice in (SAFE, CONDITIONAL, KEEP)},
               'links_checked': len(links), 'broken_links_in_current_container': sum(not row['target_exists'] for row in links.values()),
               'verified_duplicate_pairs': len(duplicates),
               'verified_duplicate_raw_bytes': sum(row['raw_bytes'] for row in duplicates.values()),
               'groups': dict(groups), 'directory_summary': directory_rows,
               'limitations': ['Snapshot inventories metadata and classifies by inspected protocols and paths; it is not a full semantic reread of every dataset frame.',
                               'V4 writes concurrently; current-epoch paths are excluded from duplicate checks.',
                               'Logical sizes do not measure allocated space; duplicate gzip verification does not itself authorize removal.',
                               'An entire shared Docker image must not be counted once per old container.'],
               'deletions_performed': 0}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    assert len(files) == sum(value['files'] for value in summary['decisions'].values())
    assert all(row['reason'] for row in files)
    assert all(row['decision'] != SAFE for row in files if row['protection'])
    assert all(row['decision'] != SAFE for row in files if row['path'].startswith('experiments/v4_budget_free/'))
    print(json.dumps({key: summary[key] for key in ('files', 'directories', 'logical_bytes', 'decisions', 'verified_duplicate_pairs', 'verified_duplicate_raw_bytes', 'deletions_performed')}, ensure_ascii=False))


if __name__ == '__main__':
    if '--verify-duplicates' in sys.argv:
        verify_duplicates()
    else:
        classify()
