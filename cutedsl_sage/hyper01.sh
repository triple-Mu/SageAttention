#!/usr/bin/env bash
# hyper01 (H200) 远程验证：rsync cutedsl_sage/ 到容器挂载路径 + docker exec 执行。
#   ./hyper01.sh setup        容器内安装 nvidia-cutlass-dsl / apache-tvm-ffi / pytest
#   ./hyper01.sh sync         仅同步代码
#   ./hyper01.sh run '<cmd>'  同步后在容器内执行（venv 已激活、cwd 为 cutedsl_sage）
#   ./hyper01.sh test [args]  同步后跑 pytest（默认 test_sage_sm90.py）
set -euo pipefail

REMOTE=hyper01
CONTAINER=sglang-diffusion-qwenimage
VENV=/data/.torch/bin/activate
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

# 挂载映射已确认（docker inspect）：宿主机 /data02/triplemu ↔ 容器 /data
HOST_DIR=/data02/triplemu/workspace/SageAttention/cutedsl_sage
CTR_DIR=/data/workspace/SageAttention/cutedsl_sage

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

do_sync() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$HOST_DIR'"
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude __pycache__ --exclude .pytest_cache \
    "$LOCAL_DIR/" "$REMOTE:$HOST_DIR/"
}

in_container() {
  local cmd="source $VENV && cd $CTR_DIR && $1"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "docker exec -i $CONTAINER bash -c $(printf '%q' "$cmd")"
}

case "${1:-}" in
  # 用 python -m pip：venv 的 pip 入口脚本 shebang 指向已失效的 python3.10，无法直接执行
  setup) in_container "python -m pip install 'nvidia-cutlass-dsl==4.6.0' apache-tvm-ffi pytest" ;;
  sync)  do_sync ;;
  run)   shift; do_sync; in_container "$*" ;;
  # printf '%q ' 逐参数转义，保留 -m 'not slow' 这类带空格参数的边界
  test)  shift; do_sync; in_container "python -m pytest -v -x -s $(printf '%q ' "${@:-test_sage_sm90.py}")" ;;
  *)     echo "用法: $0 {setup|sync|run '<cmd>'|test [pytest args]}" >&2; exit 1 ;;
esac
