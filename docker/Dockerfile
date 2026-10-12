ARG BASE_IMAGE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/zhongjiaru/CoopTrack" \
      org.opencontainers.image.description="Official CoopTrack Stage 1 stack migrated into Docker" \
      org.cooptrack.prefix="/home/yujie/miniconda/envs/cooptrack-smoke"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gcc-11 g++-11 ninja-build \
        libgl1 libglib2.0-0t64 libxext6 libxrender1 libgomp1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Preserve the exact prefix: pip shebangs, RPATHs and editable links stay valid.
COPY cooptrack-smoke/ /home/yujie/miniconda/envs/cooptrack-smoke/
COPY ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl /tmp/

ENV PATH=/home/yujie/miniconda/envs/cooptrack-smoke/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CONDA_PREFIX=/home/yujie/miniconda/envs/cooptrack-smoke \
    CUDA_HOME=/home/yujie/miniconda/envs/cooptrack-smoke \
    CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    TORCH_CUDA_ARCH_LIST=8.6+PTX \
    MAX_JOBS=2 \
    OMP_NUM_THREADS=2 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Keep torch's own cu111 loader behavior; do not prepend the conda CUDA11.8 lib directory.
RUN python -m pip install --no-index --no-deps --force-reinstall \
        /tmp/ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl \
    && rm /tmp/ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl \
    && ninja --version
RUN python -c "import sys,torch; assert sys.version_info[:2] == (3,8); assert torch.__version__ == '1.9.1+cu111'; print(torch.__version__)"

WORKDIR /mnt/d/CoopTrack_Audit/code/CoopTrack
CMD ["/bin/bash"]
