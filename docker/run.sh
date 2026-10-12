#!/usr/bin/env bash
set -euo pipefail

audit_root=/mnt/d/CoopTrack_Audit
original_example=/mnt/c/Users/yujie/Desktop/CoopTrack/V2X-Seq-SPD-Example
image_tag="${COOPTRACK_IMAGE_TAG:-cooptrack-stage1:29f1c52-official}"
container_name="${COOPTRACK_CONTAINER_NAME:-cooptrack-stage1-smoke}"

test -d "$audit_root"
test -d "$original_example"
if [ "$#" -eq 0 ]; then
    set -- python "$audit_root/docker/container_gate.py"
fi

expected_image_id=$(docker image inspect "$image_tag" --format '{{.Id}}')
if container_info=$(docker container inspect "$container_name" \
    --format '{{.Image}} {{.State.Running}}' 2>/dev/null); then
    read -r actual_image_id container_running <<< "$container_info"
    if [[ "$actual_image_id" != "$expected_image_id" ]]; then
        printf 'Container %s uses another image; preserve it and select a different COOPTRACK_CONTAINER_NAME.\n' "$container_name" >&2
        exit 2
    fi
    if [[ "$container_running" != true ]]; then
        docker start "$container_name" >/dev/null
    fi
    exec docker exec --workdir "$audit_root/code/CoopTrack" "$container_name" "$@"
fi

docker run --rm --name "$container_name" --gpus all --shm-size=2g \
    --mount "type=bind,source=$audit_root,target=$audit_root" \
    --mount "type=bind,source=$original_example,target=$original_example,readonly" \
    --workdir "$audit_root/code/CoopTrack" \
    "$image_tag" "$@"
