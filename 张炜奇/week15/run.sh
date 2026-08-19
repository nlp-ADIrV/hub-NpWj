#!/bin/bash
# 用法:
#   ./run.sh                 → 跑一次开发规划（CLI）
#   ./run.sh eval 2          → parallel vs serial A/B 对比（只跑前 2 题，约 4~5 分钟）
#   ./run.sh eval            → A/B 对比全部 4 题（约 8~10 分钟）
#   ./run.sh selftest        → 离线自测（不需要任何 API Key）
set -e
cd "$(dirname "$0")"

case "${1:-run}" in
  selftest) python src/self_test.py; exit 0 ;;
esac

# 以下命令需要 API Key（建议写入 ~/.zshrc 永久生效）
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "⚠️  未设置 DEEPSEEK_API_KEY，请先: export DEEPSEEK_API_KEY=\"sk-xxx\"" >&2
  exit 1
fi
if [ -z "$TAVILY_API_KEY" ]; then
  echo "⚠️  未设置 TAVILY_API_KEY，联网搜索会失败（LLM 会收到错误并自行兜底）" >&2
fi

case "${1:-run}" in
  eval) python src/eval_compare.py --limit "${2:-0}" ;;
  *)    python src/agents.py ;;
esac
