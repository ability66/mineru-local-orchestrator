#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  scripts/sync_to_server.sh --remote user@host:/path/to/repo/ [选项]

选项：
  --local-dir DIR         本地同步源目录，默认当前仓库根目录
  --remote DEST           远端目标目录，必填
  --preserve PATH         白名单目录，可重复传入；相对远端仓库根目录
  --preserve-file FILE    从文件读取白名单目录，每行一个相对路径
  --exclude-from FILE     额外的 rsync exclude 文件
  --dry-run               只打印将要同步的内容，不实际执行
  -h, --help              查看帮助

示例：
  scripts/sync_to_server.sh \
    --remote user@server:/srv/mineru-local-orchestrator/ \
    --preserve data/private \
    --preserve outputs/manual_debug
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_DIR="${REPO_ROOT}"
REMOTE_DEST=""
PRESERVE_FILE=""
EXCLUDE_FILE=""
DRY_RUN=0
PRESERVE_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE_DEST="$2"
      shift 2
      ;;
    --preserve)
      PRESERVE_PATHS+=("$2")
      shift 2
      ;;
    --preserve-file)
      PRESERVE_FILE="$2"
      shift 2
      ;;
    --exclude-from)
      EXCLUDE_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${REMOTE_DEST}" ]]; then
  echo "缺少 --remote 参数" >&2
  usage >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "未找到 rsync，请先安装 rsync" >&2
  exit 1
fi

if [[ -n "${PRESERVE_FILE}" ]]; then
  if [[ ! -f "${PRESERVE_FILE}" ]]; then
    echo "白名单文件不存在: ${PRESERVE_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [[ -n "${line}" ]]; then
      PRESERVE_PATHS+=("${line}")
    fi
  done < "${PRESERVE_FILE}"
fi

RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --itemize-changes
  --exclude=.git/
  --exclude=.venv/
  --exclude=.pytest_cache/
  --exclude=.ruff_cache/
)

if [[ -n "${EXCLUDE_FILE}" ]]; then
  RSYNC_ARGS+=("--exclude-from=${EXCLUDE_FILE}")
fi

for preserve_path in "${PRESERVE_PATHS[@]}"; do
  normalized_path="$(printf '%s' "${preserve_path}" | sed 's#^/*##;s#/*$##')"
  if [[ -z "${normalized_path}" ]]; then
    continue
  fi
  RSYNC_ARGS+=("--filter=P ${normalized_path}/***")
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "同步源目录: ${LOCAL_DIR}"
echo "远端目标目录: ${REMOTE_DEST}"
if [[ ${#PRESERVE_PATHS[@]} -gt 0 ]]; then
  echo "白名单目录:"
  for preserve_path in "${PRESERVE_PATHS[@]}"; do
    echo "  - ${preserve_path}"
  done
fi

rsync "${RSYNC_ARGS[@]}" "${LOCAL_DIR}/" "${REMOTE_DEST}"
