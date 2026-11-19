#!/usr/bin/env python3
"""从 ms-swift 的 logging.jsonl 画训练曲线，不需要 TensorBoard。

用法：
    python scripts/plot_training.py /mnt/workspace/output/cls_qwen3/v0-20261119-140807
    # 产出 <run目录>/training_curves.png

为什么不用 TensorBoard：它在容器里跑在 6006 端口，要额外做端口映射才能从浏览器打开；
这个脚本直接出 PNG，WebIDE 文件树里双击就能看，也方便塞进作业报告。

依赖：matplotlib（容器里没有就 pip install matplotlib）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError:
    sys.exit("缺 matplotlib：pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/")


def pick_cjk_font() -> None:
    """容器里通常没有中文字体，找不到就退回英文标签，别让图上全是方块。"""
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return


pick_cjk_font()

METRICS = {
    "loss": ("训练 loss", "Train loss"),
    "eval_loss": ("验证 loss", "Val loss"),
    "eval_accuracy": ("验证准确率", "Val accuracy"),
    "token_acc": ("token 准确率", "Token accuracy"),
    "learning_rate": ("学习率", "Learning rate"),
}


def load_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def x_of(d: dict, idx: int) -> float:
    if "epoch" in d and isinstance(d["epoch"], (int, float)):
        return float(d["epoch"])
    for key in ("global_step", "step"):
        if key in d and isinstance(d[key], (int, float)):
            return float(d[key])
    return float(idx)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    run = Path(sys.argv[1]).expanduser().resolve()
    if not run.exists():
        sys.exit(f"目录不存在：{run}")

    log_file = run / "logging.jsonl"
    if not log_file.exists():
        # swift 有时把日志放在更深一层
        cands = list(run.glob("**/logging.jsonl"))
        if not cands:
            sys.exit(f"{run} 下没有 logging.jsonl")
        log_file = cands[0]

    rows = load_log(log_file)
    if not rows:
        sys.exit("logging.jsonl 是空的")

    series: dict[str, list[tuple[float, float]]] = {k: [] for k in METRICS}
    for i, d in enumerate(rows):
        x = x_of(d, i)
        for key in METRICS:
            v = d.get(key)
            if isinstance(v, (int, float)):
                series[key].append((x, float(v)))

    has_epoch = any("epoch" in d for d in rows)
    xlabel = "epoch" if has_epoch else "step"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training curves · {run.name}", fontsize=13)

    ax = axes[0][0]
    for key, color in (("loss", "#378ADD"), ("eval_loss", "#D85A30")):
        pts = series[key]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=METRICS[key][0], color=color, linewidth=1.2)
    ax.set_title(METRICS["loss"][0])
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    pts = series["eval_accuracy"]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#1D9E75", linewidth=1.2, marker="o", markersize=3)
        best = max(pts, key=lambda p: p[1])
        ax.axhline(y=0.02, color="#B4B2A9", linestyle="--", linewidth=1)
        ax.annotate(f"best {best[1]:.3f} @ {best[0]:.2f}",
                    xy=best, xytext=(-8, 10), textcoords="offset points",
                    fontsize=9, color="#0F6E56", ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(METRICS["eval_accuracy"][0])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.06, "50 类随机猜 = 0.02", transform=ax.transAxes, fontsize=8, color="#5F5E5A")
    else:
        ax.text(0.5, 0.5, "没有 eval 记录\n训练时是不是忘了 --val_dataset？",
                ha="center", va="center", fontsize=11, color="#A32D2D", transform=ax.transAxes)
        ax.set_title(METRICS["eval_accuracy"][0])
        ax.axis("off")

    ax = axes[1][0]
    pts = series["token_acc"]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#534AB7", linewidth=1.2)
        ax.set_ylim(0, 1.02)
    ax.set_title(METRICS["token_acc"][0])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    pts = series["learning_rate"]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#BA7517", linewidth=1.2)
    ax.set_title(METRICS["learning_rate"][0])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)

    out = run / "training_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已生成 {out}")

    # 控制台顺手把关键数字打出来，省得去翻 jsonl
    print("\n--- 摘要 ---")
    print(f"总记录 {len(rows)} 行 · 日志 {log_file}")
    if series["loss"]:
        print(f"loss      : {series['loss'][0][1]:.4f} → {series['loss'][-1][1]:.4f}")
    if series["eval_accuracy"]:
        best = max(series["eval_accuracy"], key=lambda p: p[1])
        print(f"验证准确率: 最高 {best[1]:.4f} @ {xlabel} {best[0]:.2f}，最后一次 {series['eval_accuracy'][-1][1]:.4f}")
        if series["token_acc"] and series["token_acc"][-1][1] > 0.99 and best[1] < 0.9:
            print("注意：token 准确率接近 1 但验证准确率不高，说明模型很会背答案格式，判定能力没跟上。")
    if series.get("eval_loss"):
        print(f"验证 loss : {series['eval_loss'][0][1]:.4f} → {series['eval_loss'][-1][1]:.4f}")
        if series["eval_loss"][-1][1] > series["eval_loss"][0][1]:
            print("注意：验证 loss 在上升，过拟合已经发生，该用 early stopping 或减轮数。")


if __name__ == "__main__":
    main()
