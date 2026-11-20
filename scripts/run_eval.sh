#!/usr/bin/env bash
# 两轮评测 + 出报告。可断点续跑：已完成的轮次自动跳过。
# 用法：ADAPTER=/path/to/checkpoint-1900 bash scripts/run_eval.sh
# 后台跑：nohup bash scripts/run_eval.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-/mnt/workspace/output/cls_qwen3/v0-20261119-140807/checkpoint-1900}"
FT_OUT="${FT_OUT:-data/cls/eval_report.json}"
BL_OUT="${BL_OUT:-data/cls/eval_report_baseline.json}"
REPORT="${REPORT:-reports/eval_report.html}"
LIMIT="${LIMIT:-0}"

mkdir -p logs reports data/cls
LOG="logs/eval-$(date +%Y%m%d-%H%M%S).log"
LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then LIMIT_ARG="--limit $LIMIT"; fi

say() { printf '\n[%s] == %s ==\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

if pgrep -f "scripts/eval_cls.py" >/dev/null 2>&1; then
  echo "已有评测进程在跑，不重复启动。查看：ps aux | grep eval_cls"
  exit 1
fi

# JSON 完整才算跑完（文件存在但写了一半的情况很常见）
done_ok() {
  [ -f "$1" ] && "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
sys.exit(0 if isinstance(d.get('n'),int) and d['n']>0 and 'accuracy' in d else 1)
" "$1" 2>/dev/null
}

say "评测开始 $(date '+%F %T')  基座=$MODEL  适配器=$ADAPTER  日志=$LOG"

if done_ok "$FT_OUT"; then
  say "[跳过] 微调轮已完成：$FT_OUT"
else
  say "第 1/3 步 · 微调后评测"
  # shellcheck disable=SC2086
  "$PY" scripts/eval_cls.py --model "$MODEL" --adapter "$ADAPTER" $LIMIT_ARG --out "$FT_OUT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { say "[失败] 微调轮退出码 $rc，中止"; exit "$rc"; }
fi

if done_ok "$BL_OUT"; then
  say "[跳过] 基线已完成：$BL_OUT"
else
  say "第 2/3 步 · zero-shot 基线"
  "$PY" scripts/eval_cls.py --no-adapter --model "$MODEL" $LIMIT_ARG --out "$BL_OUT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || say "[警告] 基线轮退出码 $rc，仍尝试出报告"
fi

say "第 3/3 步 · 生成 HTML 报告"
if "$PY" scripts/make_report.py --baseline "$BL_OUT" --finetuned "$FT_OUT" --out "$REPORT" 2>&1 | tee -a "$LOG"; then
  say "报告已生成：$REPORT"
else
  say "[警告] 报告失败，多半缺 matplotlib：pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/"
  say "评测结果本身没问题，json 都在 data/cls/ 下"
fi

say "汇总"
for f in "$FT_OUT" "$BL_OUT"; do
  [ -f "$f" ] && "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('  %-26s acc=%.4f  macro_f1=%.4f  top3=%.4f  n=%d' % (
  sys.argv[1].split('/')[-1], d['accuracy'], d['macro_f1'], d['top3'], d['n']))
" "$f" | tee -a "$LOG"
done

say "评测结束 $(date '+%F %T')"
echo
echo "日志：$LOG"
echo "断连不影响——用 nohup 启动后进程已脱离终端。重连后 tail -f $LOG 看进度"
