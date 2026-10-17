# Docker-only rebuild recipe

状态：`RECIPE_PREPARED_STATIC_CHECKED`。本次实际使用的迁移镜像已完成 GPU gate；这里的独立重建配方按要求只准备、检查，没有再运行一次构建。

当前已验证镜像为 `cooptrack-stage1:29f1c52-official`，ID 为 `sha256:22197870c646dcd6221210f814fe3494218e3e3fb22c6030faa47afc6dc649f8`。宿主临时 Conda prefix 已删除，因此历史 `build-image.sh` 无法再次从该 prefix 构建。以下配方仅使用保留的官方资源和锁文件，全部安装与编译发生在 Docker build 内。

当前 SOP 直接使用已运行并通过 gate 的持久容器：

```bash
docker exec --workdir /mnt/d/CoopTrack_Audit/code/CoopTrack cooptrack-stage1-smoke \
  python /mnt/d/CoopTrack_Audit/docker/container_gate.py
# The wrapper now reuses that same container after verifying its image ID.
bash /mnt/d/CoopTrack_Audit/docker/run.sh python /mnt/d/CoopTrack_Audit/scripts/main_eval.py --help
```

上述是使用现有镜像；以下才是今后需要时独立重建的命令。

```bash
# WSL: only check that retained inputs exist; does not build or install.
bash /mnt/d/CoopTrack_Audit/docker/rebuild-image.sh --check

# Future rebuild, when needed. Uses a separate tag and preserves the verified image.
bash /mnt/d/CoopTrack_Audit/docker/rebuild-image.sh
```

默认新标签为 `cooptrack-stage1:29f1c52-rebuilt`，可用 `COOPTRACK_REBUILD_IMAGE` 改为另一个新标签。脚本流式传入 Dockerfile、锁和明确命名的 6 个资源文件；不创建 tar 落盘，不读取宿主环境，不包含数据集、checkpoint、2 GB Torch 分块缓存或其他项目。可以用 `COOPTRACK_AUDIT_ROOT` 指定这套保留文件的新位置。

配方按以下锁定资源重建：

| 项目 | 锁与来源 |
| --- | --- |
| Ubuntu | 官方 24.04 digest `008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3` |
| micromamba | 官方 mamba-org release `2.9.0-0`；二进制 SHA256 `366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3`；元数据保存在 `locks/micromamba_bootstrap.json` |
| Conda | `locks/conda-explicit-linux-64.lock`，30 个精确 URL/SHA256；来源仅 conda-forge 和 NVIDIA 官方 channel |
| pip | `locks/pip-artifacts-linux-64.lock`，193 个精确 artifact URL/SHA256；不重新解析依赖、不使用第三方镜像 |
| pip build tools | `locks/pip-bootstrap.lock`，先安装 setuptools 59.5.0、wheel 0.38.4、numpy 1.19.5，再构建原来的两项 sdist |
| Torch/Open3D/ninja | 从 `data/wheels` 保留的 3 个已验证 wheel 安装；Torch 1.9.1+cu111、Open3D 0.13.0、ninja 1.11.1.3 |
| CUDA 编译器 | explicit Conda 锁的 NVCC 11.8.89、CCCL、cudart 和 cudart-dev；完全独立于 EMIFF |
| CUDA 附加头文件 | `data/cuda_packages` 的 3 个 NVIDIA dev archive；复核官方 MD5 和本地留存 SHA256后只提取 14 个已记录 header |
| mmdet3d | 官方 commit `f1107977dfd26155fc1f83779ee6535d2468f449`，v0.17.1；未修改源码 |
| CoopTrack | 官方 commit `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`，应用锁定 SHA256 的单行 CPU inverse 补丁 |

原始 `pip freeze`、安装列表、Conda explicit export 和源码 diff 保留在 `locks`，用于来源审计。规范化记录只做三件事：pip 由 Conda 锁管理；mmdet3d 单独从源码构建；把损坏的 host ninja 1.11.1.1 替换为 Docker 实际使用的官方 ninja 1.11.1.3 wheel。`runtime_requirements.txt` 是原始证据，其中旧路径和旧 ninja 不用于重建。

193 个 pip artifact 的来源证据逐项保存在 `locks/pip-artifact-provenance.json`。其余 wheel/两项 sdist 的 URL 和 SHA256 来自本次成功安装的 pip report，域名仅 `files.pythonhosted.org`、`download.pytorch.org`、`download.openmmlab.com`；setuptools/wheel 的缺失早期元数据由官方 PyPI JSON 补齐，没有下载新的 package body。网络不通时可以重试构建；已经保留的大 wheel 不需再次下载。

`Dockerfile.rebuild` 使用多阶段构建，最终镜像复制完整 prefix、官方源码及编译完成的 mmdet3d operator，使用 `/opt/cooptrack/mmdetection3d` 内部 editable 路径。它不再依赖 audit 中原先编译的 mmdet3d `.so`。GCC/G++ 11、CUDA_HOME、`TORCH_CUDA_ARCH_LIST=8.6+PTX`、`MAX_JOBS=2` 与本次通过 gate 的编译选择相同；构建时设 `FORCE_CUDA=1`，因为 Docker build 通常没有 GPU。没有覆盖 `LD_LIBRARY_PATH`，Torch 继续使用其 cu111 runtime。

重建后需要在 GPU 容器中运行同一 gate，再用于 SOP。下面采用另一个容器名，避免与已运行的 `cooptrack-stage1-smoke` 冲突：

```bash
docker run --rm --name cooptrack-stage1-rebuild-gate --gpus all --shm-size=2g \
  --mount type=bind,source=/mnt/d/CoopTrack_Audit,target=/mnt/d/CoopTrack_Audit \
  --mount type=bind,source=/mnt/c/Users/yujie/Desktop/CoopTrack/V2X-Seq-SPD-Example,target=/mnt/c/Users/yujie/Desktop/CoopTrack/V2X-Seq-SPD-Example,readonly \
  cooptrack-stage1:29f1c52-rebuilt \
  python /mnt/d/CoopTrack_Audit/docker/container_gate.py
```

gate 会覆盖 `logs/docker_*gate.json` 等现有 Docker gate 文件，运行前应另存本次证据或为重建 gate 指定不同证据文件名。当前配方尚未执行，所以没有声称重建镜像已经通过 GPU gate。固定 Ubuntu 基础层、Conda 包、pip artifact 和源码能重现这套版本选择；APT 仍从 Ubuntu 官方仓库取可用包，未锁到特定仓库快照，因此不保证未来完整镜像 ID 或系统包字节完全一致。

静态验证：Bash 语法 PASS、Python 源码编译语法 PASS、30 个 Conda SHA256 记录 PASS、193 个 pip artifact/hash 与目标版本齐全、重建输入 `--check` PASS、tar 路径转换的小文件验证 PASS。未执行第二次 build，未在宿主安装软件，未改现有容器。

官方参考：[micromamba 安装文档](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)、[固定 micromamba release](https://github.com/mamba-org/micromamba-releases/releases/tag/2.9.0-0)、[Docker tar context](https://docs.docker.com/build/concepts/context/)、[NVIDIA Docker GPU 参数](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)、[setuptools 元数据](https://pypi.org/pypi/setuptools/59.5.0/json)、[wheel 元数据](https://pypi.org/pypi/wheel/0.38.4/json)。
