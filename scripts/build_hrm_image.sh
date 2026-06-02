#!/usr/bin/env bash
set -euo pipefail
set -x

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

image_name="${IMAGE_NAME:-registry.h.pjlab.org.cn/ailab-moe-moe_gpu/hrm-text}"
image_tag="${IMAGE_TAG:-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
full_image="${image_name}:${image_tag}"

docker build \
  -f docker/Dockerfile \
  -t "${full_image}" \
  --progress=plain \
  .

if [[ "${PUSH:-0}" == "1" ]]; then
  docker push "${full_image}"
fi

echo "${full_image}"
