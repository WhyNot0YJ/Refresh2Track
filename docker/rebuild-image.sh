#!/usr/bin/env bash
set -euo pipefail
task_audit_root="${COOPTRACK_AUDIT_ROOT:-/mnt/d/CoopTrack_Audit}"
task_image="${COOPTRACK_REBUILD_IMAGE:-cooptrack-stage1:29f1c52-rebuilt}"
task_docker_dir="${task_audit_root}/docker"
task_wheels=(
  'torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl'
  'open3d-0.13.0-cp38-cp38-manylinux2014_x86_64.whl'
  'ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl'
)
task_cuda=(
  'libcublas-dev-11.11.3.6-0.tar.bz2'
  'libcusparse-dev-11.7.5.86-0.tar.bz2'
  'libcusolver-dev-11.4.1.48-0.tar.bz2'
)
for task_path in "${task_docker_dir}/Dockerfile.rebuild" "${task_docker_dir}/rebuild-helper.py" \
                 "${task_docker_dir}/locks/conda-explicit-linux-64.lock" \
                 "${task_docker_dir}/locks/pip-artifacts-linux-64.lock"; do
  test -f "${task_path}" || { printf 'Missing rebuild input: %s\n' "${task_path}" >&2; exit 1; }
done
for task_file in "${task_wheels[@]}"; do test -f "${task_audit_root}/data/wheels/${task_file}"; done
for task_file in "${task_cuda[@]}"; do test -f "${task_audit_root}/data/cuda_packages/${task_file}"; done
if [[ "${1:-}" == '--check' ]]; then
  printf 'REBUILD_INPUTS_PRESENT: 3 wheels, 3 CUDA archives, captured locks; no host prefix required.\n'
  exit 0
fi
[[ $# == 0 ]] || { printf 'Usage: rebuild-image.sh [--check]\n' >&2; exit 2; }
if [[ "${task_image}" == 'cooptrack-stage1:29f1c52-official' ]]; then
  printf 'Choose a separate rebuilt tag to preserve the verified migration image.\n' >&2
  exit 2
fi
mkdir -p "${task_audit_root}/logs"
# Tar creates only the named resource entries; excludes 2 GB range fragments and data.
# GNU tar transforms each retained data path into the Dockerfile context layout.
tar -C "${task_docker_dir}" -cf - Dockerfile.rebuild rebuild-helper.py locks \
    --transform='s,^data/,,' -C "${task_audit_root}" \
    "${task_wheels[@]/#/data/wheels/}" "${task_cuda[@]/#/data/cuda_packages/}" \
  | docker build --progress=plain --tag "${task_image}" --file Dockerfile.rebuild - \
  2>&1 | tee "${task_audit_root}/logs/docker_rebuild.log"
docker image inspect "${task_image}" --format '{{.Id}} {{.Size}}' \
  | tee "${task_audit_root}/logs/docker_rebuild_identity.txt"
