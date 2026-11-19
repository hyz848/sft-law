#!/usr/bin/env bash
# 魔搭 Notebook 一键微调：环境自检 → 装 ms-swift → 数据自检 → 训练 → 挑 checkpoint → 评测
#
# 在魔搭 WebIDE 终端里这样用（仓库公开，数据已在 git 里，不需要手动上传）：
#   cd /mnt/workspace
#   git clone https://github.com/hyz848/sft-law.git
#   cd sft-law
#   bash scripts/run_modelscope.sh
#
# 常用变体：
#   SMOKE=1 bash scripts/run_modelscope.sh   # 只用 500 条、1 轮、每 20 步存一次，约 5 分钟
#   SKIP_EVAL=1 bash scripts/run_modelscope.sh  # 只训不评
#   EPOCHS=3 bash scripts/run_modelscope.sh
#
# 日志落在 logs/ 下，全程 tee，断了也能看。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
EPOCHS="${EPOCHS:-2}"
SMOKE="${SMOKE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
WORK="${WORK:-/mnt/workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/output/cls_qwen3}"
LOG_DIR="$ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/run-$STAMP.log"

mkdir -p "$LOG_DIR"

# 模型缓存挪到持久盘：/root 不持久，重开实例 4GB 基座要重下
export MODELSCOPE_CACHE="$WORK/.cache/modelscope"
export HF_HOME="$WORK/.cache/hf"
export USE_MODELSCOPE_HUB=1
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$OUTPUT_DIR"

say() { echo ""; echo "===== $* ====="; }
die() { echo ""; echo "[失败] $*"; echo "日志：$LOG"; exit 1; }

exec > >(tee -a "$LOG") 2>&1

say "0/6 环境自检"
echo "仓库：$ROOT"
echo "时间：$(date '+%F %T')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || die "torch 不可用，检查镜像是否选了 PyTorch + CUDA"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "CUDA 不可用。CPU 上训 1.7B 是按天算的，回到实例页换成 GPU 规格"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

say "1/6 安装 ms-swift"
# 用阿里源，官方 PyPI 在国内经常卡到怀疑人生
python -m pip install -q -U ms-swift -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
  || die "ms-swift 安装失败"
python -m swift --help >/dev/null 2>&1 || die "ms-swift 装上了但跑不起来"
echo "ms-swift 就绪"

say "2/6 数据自检"
TRAIN="$ROOT/data/cls/export/ms_swift/train.jsonl"
VAL="$ROOT/data/cls/export/ms_swift/val.jsonl"
TEST="$ROOT/data/cls/export/test_eval.jsonl"
for f in "$TRAIN" "$VAL" "$TEST"; do
  [ -f "$f" ] || die "缺文件：$f"
done

# 数据走 Git LFS：没装 lfs 或 clone 时跳过了 smudge，拿到的是三行指针文本
if head -c 60 "$TRAIN" | grep -q "git-lfs"; then
  echo "检测到 Git LFS 指针文件，开始拉取真实数据……"
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "没装 git-lfs，尝试安装"
    (apt-get install -y git-lfs >/dev/null 2>&1 || pip install -q git-lfs >/dev/null 2>&1) || true
  fi
  command -v git-lfs >/dev/null 2>&1 || die "装不上 git-lfs：apt-get install -y git-lfs 后再跑一次"
  git lfs install --skip-repo >/dev/null 2>&1 || true
  git lfs pull --include="data/**/*.jsonl" || die "LFS 拉取失败（GitHub LFS 走公网，慢的话重跑一次）"
fi

echo "train $(wc -l < "$TRAIN") 行"
echo "val   $(wc -l < "$VAL") 行"
echo "test  $(wc -l < "$TEST") 行"
[ "$(wc -l < "$TRAIN")" -gt 1000 ] || die "train 只有 $(wc -l < "$TRAIN") 行，还是指针文件，手动从本机把 data/cls/export/ms_swift/*.jsonl 拖进来"
head -n 1 "$TRAIN" | grep -q '"query"' || die "train 字段不是 query，重新跑 scripts/export_platform_data.py"

if [ "$SMOKE" = "1" ]; then
  TRAIN_USE="$WORK/smoke_train.jsonl"
  VAL_USE="$WORK/smoke_val.jsonl"
  head -n 500 "$ROOT/data/cls/export/ms_swift/train.jsonl" > "$TRAIN_USE"
  head -n 100 "$ROOT/data/cls/export/ms_swift/val.jsonl" > "$VAL_USE"
  EPOCHS=1
  STEPS=(--eval_steps 20 --save_steps 20)
  echo "SMOKE 模式：500 / 100 条，1 轮，每 20 步存一次"
else
  TRAIN_USE="$TRAIN"
  VAL_USE="$VAL"
  STEPS=(--eval_steps 100 --save_steps 100)
fi

say "3/6 起训"
echo "基座 $MODEL · QLoRA 4bit · r=16 α=32 · ${EPOCHS} 轮"
echo "预计耗时：完整数据约 1 小时；SMOKE 约 5 分钟"
python -m swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --quant_method bnb --quant_bits 4 \
  --dataset "$TRAIN_USE" \
  --val_dataset "$VAL_USE" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 512 \
  "${STEPS[@]}" --save_total_limit 2 \
  --logging_steps 20 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  || die "训练失败，往上翻错误信息（OOM 就调小 per_device_train_batch_size）"

say "4/6 挑 checkpoint（按验证准确率，不是按最后一个）"
CKPT="$(python - "$OUTPUT_DIR" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
runs = sorted(glob.glob(os.path.join(out, "v*")), key=os.path.getmtime)
if not runs:
    sys.exit(0)
run = runs[-1]
best = None
for name in ("logging.jsonl", "trainer_state.json"):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        continue
    try:
        if name.endswith(".json"):
            hist = json.load(open(p, encoding="utf-8")).get("log_history", [])
        else:
            hist = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    except Exception:
        continue
    for d in hist:
        if "eval_accuracy" in d and d.get("global_step") is not None:
            if best is None or d["eval_accuracy"] > best[1]:
                best = (d["global_step"], d["eval_accuracy"])
    if best:
        print("验证准确率最高：step %s → %.4f" % best, file=sys.stderr)
        break
if best:
    print(os.path.join(run, "checkpoint-%d" % best[0]))
else:
    cks = sorted(glob.glob(os.path.join(run, "checkpoint-*")), key=lambda x: int(x.rsplit("-", 1)[1]))
    if cks:
        print(cks[-1], file=sys.stderr)
        print(cks[-1])
PY
)"

if [ -z "${CKPT:-}" ] || [ ! -d "$CKPT" ]; then
  echo "没解析出 checkpoint，退回用最新一个"
  CKPT="$(ls -d "$OUTPUT_DIR"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)"
fi
[ -d "${CKPT:-}" ] || die "输出目录里没有 checkpoint，训练可能没真正开始"
echo "选中：$CKPT"
du -sh "$CKPT"

if [ "$SKIP_EVAL" = "1" ]; then
  say "5/6 已跳过评测（SKIP_EVAL=1）"
else
  say "5/6 评测：先基线，再微调后"
  echo "--- 基线（zero-shot）---"
  python scripts/eval_cls.py --no-adapter --model "$MODEL" \
    || die "基线评测失败"

  echo "--- 微调后 ---"
  python scripts/eval_cls.py --model "$MODEL" --adapter "$CKPT" \
    || die "微调后评测失败（检查 --adapter 是不是传了含 adapter_model.safetensors 的目录）"
fi

say "6/6 收尾"
echo "适配器：$CKPT"
echo "评测报告：$ROOT/data/cls/eval_report.json"
echo "训练日志：$LOG"
echo ""
echo "把最后这两段数字带走：基线的 accuracy / macro-F1，和微调后的同一组。"
echo "只报一个数不算实验——没有基线，那个数说明不了任何事。"
