#!/usr/bin/env bash
set -euo pipefail

audit_root=/mnt/d/CoopTrack_Audit
docker_dir="$audit_root/docker"
env_prefix=/home/yujie/miniconda/envs/cooptrack-smoke
image_tag=cooptrack-stage1:29f1c52-official
ninja_wheel=ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

test "$(realpath -e "$env_prefix")" = "$env_prefix"
test -x "$env_prefix/bin/python"
test -f "$docker_dir/Dockerfile"
test -f "$audit_root/data/wheels/$ninja_wheel"
command -v docker >/dev/null
docker version --format '{{.Server.Version}}'

# Native WSL pipe keeps tar bytes intact and avoids a saved 7 GB archive on either drive.
# Context includes only the named prefix and this Dockerfile, never base conda or EMIFF.
tar --numeric-owner -C /home/yujie/miniconda/envs -cf - cooptrack-smoke \
    -C "$docker_dir" Dockerfile \
    -C "$audit_root/data/wheels" "$ninja_wheel" \
    | docker build --progress=plain --tag "$image_tag" --file Dockerfile - \
    2>&1 | tee "$audit_root/logs/docker_build.log"

docker image inspect "$image_tag" \
    --format '{{.Id}} {{.Size}} {{json .RepoDigests}}' \
    | tee "$audit_root/logs/docker_image_identity.txt"
