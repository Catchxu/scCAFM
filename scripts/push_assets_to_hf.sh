#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${ASSETS_DIR:-${ROOT_DIR}/assets}"
HF_REPO_ID="${HF_REPO_ID:-kaichenxu/scCAFM}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_REVISION="${HF_REVISION:-main}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Sync assets from local package}"
TMP_PARENT="${TMP_PARENT:-/tmp}"

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

have_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1
}

require_command hf
require_command python

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "Assets directory not found: ${ASSETS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ASSETS_DIR}/README.md" ]]; then
  echo "Expected assets/README.md to exist before syncing." >&2
  exit 1
fi

if [[ ! -d "${ASSETS_DIR}/models" ]]; then
  echo "Expected assets/models/ to exist before syncing." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMP_PARENT}/scCAFM-hf-sync.XXXXXX")"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "Ensuring Hugging Face repo exists: ${HF_REPO_ID}"
hf repo create "${HF_REPO_ID}" --repo-type "${HF_REPO_TYPE}" --exist-ok

if have_command git && have_command git-lfs; then
  require_command rsync

  echo "Cloning https://huggingface.co/${HF_REPO_ID} to ${TMP_DIR}"
  git clone --branch "${HF_REVISION}" "https://huggingface.co/${HF_REPO_ID}" "${TMP_DIR}"

  echo "Syncing ${ASSETS_DIR}/ -> ${TMP_DIR}/"
  rsync -av --delete \
    --exclude ".git/" \
    --exclude ".cache/" \
    --exclude "__pycache__/" \
    "${ASSETS_DIR}/" "${TMP_DIR}/"

  cd "${TMP_DIR}"
  git lfs install >/dev/null
  git add -A

  if git diff --cached --quiet; then
    echo "No changes to push. Remote already matches local assets."
    exit 0
  fi

  git commit -m "${COMMIT_MESSAGE}"
  git push origin "${HF_REVISION}"
  echo "Pushed ${ASSETS_DIR} to https://huggingface.co/${HF_REPO_ID} via git mirror sync"
  exit 0
fi

echo "git-lfs is unavailable; falling back to Hugging Face API sync"

mapfile -t REMOTE_EXTRAS < <(
  ASSETS_DIR="${ASSETS_DIR}" HF_REPO_ID="${HF_REPO_ID}" HF_REPO_TYPE="${HF_REPO_TYPE}" HF_REVISION="${HF_REVISION}" \
    python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import HfApi

assets_dir = Path(os.environ["ASSETS_DIR"]).resolve()
repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ["HF_REPO_TYPE"]
revision = os.environ["HF_REVISION"]

ignored_dir_names = {".git", ".cache", "__pycache__"}
local_files: set[str] = set()
for root, dirs, files in os.walk(assets_dir):
    dirs[:] = [name for name in dirs if name not in ignored_dir_names]
    for filename in files:
        file_path = Path(root) / filename
        local_files.add(file_path.relative_to(assets_dir).as_posix())

api = HfApi()
remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision))
for path in sorted(remote_files - local_files):
    print(path)
PY
)

if [[ "${#REMOTE_EXTRAS[@]}" -gt 0 ]]; then
  echo "Deleting ${#REMOTE_EXTRAS[@]} remote file(s) not present in local assets"
  hf repo-files delete \
    --repo-type "${HF_REPO_TYPE}" \
    --revision "${HF_REVISION}" \
    --commit-message "Delete files missing from local assets" \
    "${HF_REPO_ID}" "${REMOTE_EXTRAS[@]}"
fi

hf upload-large-folder \
  "${HF_REPO_ID}" "${ASSETS_DIR}" \
  --repo-type "${HF_REPO_TYPE}" \
  --revision "${HF_REVISION}" \
  --exclude ".cache/**" "__pycache__/**" \
  --no-bars

echo "Pushed ${ASSETS_DIR} to https://huggingface.co/${HF_REPO_ID} via API sync"
