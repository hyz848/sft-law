#!/usr/bin/env bash
# 把 LoRA 适配器从训练输出目录推到魔搭模型仓库。
# 用法：MODELSCOPE_TOKEN=ms-xxx bash scripts/push_adapter.sh
# 可选：RUN_DIR=/path/to/v0-xxx  指定某次训练（默认取最新）
set -uo pipefail

REPO="${REPO:-hyz848/lawcls-qwen3-lora}"
TOKEN="${MODELSCOPE_TOKEN:-}"
OUT_ROOT="${OUT_ROOT:-/mnt/workspace/output}"
RUN_DIR="${RUN_DIR:-}"
PY="${PY:-python}"

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[注意]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[错误]\033[0m %s\n' "$*" >&2; exit 1; }

say "0/5 检查 token"
if [ -z "$TOKEN" ]; then
  die "没给 token。用法：MODELSCOPE_TOKEN=ms-xxx bash scripts/push_adapter.sh"
fi
case "$TOKEN" in
  ms-*) ok "token 格式正常（${TOKEN:0:6}…）" ;;
  *)    warn "token 不以 ms- 开头，照跑，错了会在上传时 401" ;;
esac

say "1/5 定位训练输出目录"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR=$(ls -1dt "$OUT_ROOT"/*/v* "$OUT_ROOT"/v* 2>/dev/null | head -1)
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  die "找不到训练输出目录。手动指定后重跑：RUN_DIR=/mnt/workspace/output/cls_qwen3/v0-xxx bash scripts/push_adapter.sh"
fi
ok "$RUN_DIR"
echo "  含 checkpoint：$(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null | wc -l) 个"

say "2/5 挑验证准确率最高的 checkpoint"
FORCE_STEP="${CKPT_STEP:-}"
if [ -n "$FORCE_STEP" ]; then
  BEST_STEP="$FORCE_STEP"; BEST_ACC="manual"
  ok "按 CKPT_STEP 指定：checkpoint-$BEST_STEP"
else
PICKED=$("$PY" - "$RUN_DIR" <<'PY'
import glob, json, os, sys
run = sys.argv[1]
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
        acc, step = d.get("eval_accuracy"), d.get("global_step")
        if acc is not None and step is not None:
            if best is None or float(acc) > best[1]:
                best = (int(step), float(acc))
    if best:
        break
if best:
    print("%d %.4f" % best)
else:
    cks = sorted(glob.glob(os.path.join(run, "checkpoint-*")),
                 key=lambda x: int(x.rsplit("-", 1)[1]))
    if cks:
        print("%s 0" % cks[-1].rsplit("-", 1)[1])
PY
)
if [ -z "$PICKED" ]; then
  die "$RUN_DIR 下没有任何 checkpoint"
fi
BEST_STEP=$(echo "$PICKED" | awk '{print $1}')
BEST_ACC=$(echo "$PICKED" | awk '{print $2}')
if [ "$BEST_ACC" = "0" ]; then
  warn "日志里没有 eval_accuracy，退回用最后一个 checkpoint-$BEST_STEP"
  echo "  日志中出现的评估相关键（用于定位真实指标名）："
  grep -ho '"eval[a-zA-Z_]*"' "$RUN_DIR"/logging.jsonl "$RUN_DIR"/trainer_state.json 2>/dev/null | sort -u | head -8
  echo "  想强制指定就用：CKPT_STEP=<步数> MODELSCOPE_TOKEN=... bash scripts/push_adapter.sh"
fi
fi
if [ -z "${BEST_STEP:-}" ]; then
  die "没能确定 checkpoint，用 CKPT_STEP=<步数> 手动指定"
fi
CKPT="$RUN_DIR/checkpoint-$BEST_STEP"
[ -d "$CKPT" ] || die "checkpoint 目录不存在：$CKPT（可用 ls $RUN_DIR 看实际有哪些）"
if [ "$BEST_ACC" = "manual" ]; then
  ok "使用指定 checkpoint-$BEST_STEP"
elif [ "$BEST_ACC" != "0" ]; then
  ok "best = checkpoint-$BEST_STEP（eval_accuracy=$BEST_ACC）"
fi

say "3/5 只挑适配器文件（不带 optimizer.pt）"
STAGE="${STAGE:-/mnt/workspace/upload_adapter}"
rm -rf "$STAGE"; mkdir -p "$STAGE"
found=0
for f in adapter_config.json adapter_model.safetensors adapter_model.bin additional_config.json; do
  if [ -f "$CKPT/$f" ]; then
    cp "$CKPT/$f" "$STAGE/" && found=$((found+1))
  fi
done
[ "$found" -ge 2 ] || die "$CKPT 里没有 adapter_config.json + 权重文件，确认这是 MS-Swift 的 LoRA 输出目录"

# 容器内的绝对路径会跟着 adapter 一起被交出去，别的机器加载直接报路径不存在
if [ -f "$STAGE/adapter_config.json" ]; then
  "$PY" - "$STAGE/adapter_config.json" "${BASE_MODEL_ID:-Qwen/Qwen3-1.7B}" <<'PYCFG'
import json, sys
p, pub = sys.argv[1], sys.argv[2]
d = json.load(open(p, encoding="utf-8"))
old = d.get("base_model_name_or_path", "")
if old.startswith("/") or "\\" in old:
    d["base_model_name_or_path"] = pub
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  base_model_name_or_path：%s -> %s" % (old, pub))
else:
    print("  base_model_name_or_path =", old)
PYCFG
fi

ls -lh "$STAGE"
SIZE_MB=$(du -sm "$STAGE" | awk '{print $1}')
if [ "$SIZE_MB" -gt 500 ]; then
  warn "暂存目录 ${SIZE_MB}MB，偏大，检查是否混入完整权重"
else
  ok "暂存目录 ${SIZE_MB}MB（LoRA r=16 正常在 20~80MB）"
fi

say "4/5 确保 modelscope CLI 可用"
if ! command -v modelscope >/dev/null 2>&1; then
  warn "找不到 modelscope 命令，安装中"
  "$PY" -m pip install -q -U modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
fi
command -v modelscope >/dev/null 2>&1 || die "modelscope 安装后仍不可用，手动跑：$PY -m pip install -U modelscope"
if ! modelscope upload --help >/dev/null 2>&1; then
  warn "当前 modelscope 不支持 upload 子命令，升级中"
  "$PY" -m pip install -q -U modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
fi
modelscope upload --help >/dev/null 2>&1 || die "modelscope CLI 不含 upload，版本过旧"
ok "$(modelscope --version 2>/dev/null | tail -1)"

say "5/5 上传到 $REPO"
modelscope --token "$TOKEN" upload "$REPO" "$STAGE" \
  --repo-type model \
  --commit-message "50类部法分类 QLoRA 适配器（best step $BEST_STEP，eval_acc $BEST_ACC）"
RC=$?
if [ $RC -ne 0 ]; then
  die "上传失败（退出码 $RC）。常见原因：token 失效 / 仓库无写权限 / 网络抖动，重跑本脚本即可续传"
fi

cat <<EOF

$(printf '\033[1;32m上传完成\033[0m')
  模型页：https://www.modelscope.cn/models/$REPO
  提交记录：best = checkpoint-$BEST_STEP (eval_accuracy $BEST_ACC)
  暂存目录：$STAGE

把这个链接发给绵绵，剩下的下载、打包、校验由本地完成。
EOF
