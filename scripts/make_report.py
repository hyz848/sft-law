#!/usr/bin/env python3
"""把 eval_cls.py 的两份报告合成一个单文件 HTML 评估报告（图片内嵌 base64，可直接上传作业）。

用法：
    # 先跑两次评测
    python scripts/eval_cls.py --no-adapter --model Qwen/Qwen3-1.7B --out data/cls/eval_report_baseline.json
    python scripts/eval_cls.py --model Qwen/Qwen3-1.7B --adapter <ckpt> --out data/cls/eval_report.json

    # 再出报告（曲线图可选，先跑 plot_training.py）
    python scripts/make_report.py \
      --baseline data/cls/eval_report_baseline.json \
      --finetuned data/cls/eval_report.json \
      --curves /mnt/workspace/output/cls_qwen3/v0-xxx/training_curves.png \
      --out reports/eval_report.html

不依赖外部 CDN，断网也能打开；图片全部内嵌，单文件就能交。
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def setup_font() -> None:
    if not HAS_MPL:
        return
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def delta(base: float, ft: float) -> str:
    d = (ft - base) * 100
    sign = "+" if d >= 0 else ""
    cls = "up" if d > 0 else ("down" if d < 0 else "flat")
    return f'<span class="{cls}">{sign}{d:.2f} pt</span>'


def img_tag(png_bytes: bytes) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(png_bytes).decode()}" alt="">'


def confusion_png(report: dict) -> bytes | None:
    """50x50 混淆矩阵热图。刻度太密不标，配下面的「最容易混的 10 对」读。"""
    if not HAS_MPL:
        return None
    conf = report.get("confusion") or {}
    if not conf:
        return None
    per = report.get("per_class") or {}
    labels = sorted(per, key=lambda x: -per[x]["test"])
    if len(labels) < 2:
        return None
    size = len(labels)
    grid = [[conf.get(g, {}).get(p, 0) for p in labels] for g in labels]

    import io

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵（{size} 类，按测试集样本数降序）", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def top_confusions(report: dict, k: int = 10) -> list[tuple[str, str, int]]:
    conf = report.get("confusion") or {}
    pairs = []
    for gold, row in conf.items():
        for pred, n in row.items():
            if gold != pred and n > 0:
                pairs.append((gold, pred, n))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def build_html(args, base: dict, ft: dict, curves_png: bytes | None, cm_png: bytes | None) -> str:
    per = ft.get("per_class") or {}
    worst = sorted(per.items(), key=lambda kv: kv[1]["f1"])[:5]
    conf_pairs = top_confusions(ft)
    total_conf = sum(n for _, _, n in top_confusions(ft, 10**6))
    wrong = int(round(ft["n"] * (1 - ft["accuracy"])))

    rows = [
        ("accuracy（50 选 1 整体）", "主指标，会被大类带偏", base["accuracy"], ft["accuracy"]),
        ("macro-F1（每类平均）", "小类错了立刻掉，最公平", base["macro_f1"], ft["macro_f1"]),
        ("top-3（前三命中）", "看「差一点就对」的能力", base["top3"], ft["top3"]),
    ]
    metric_rows = "\n".join(
        f"<tr><td>{name}</td><td class='muted'>{note}</td>"
        f"<td class='num'>{pct(b)}</td><td class='num strong'>{pct(f)}</td><td class='num'>{delta(b, f)}</td></tr>"
        for name, note, b, f in rows
    )

    worst_rows = "\n".join(
        f"<tr><td>{html.escape(lab)}</td><td class='num'>{v['test']}</td>"
        f"<td class='num'>{v['hit']}</td><td class='num'>{v['f1']:.3f}</td></tr>"
        for lab, v in worst
    )

    conf_rows = "\n".join(
        f"<tr><td>{html.escape(g)}</td><td>{html.escape(p)}</td><td class='num'>{n}</td>"
        f"<td class='num'>{n / wrong * 100:.1f}%</td></tr>" if wrong else ""
        for g, p, n in conf_pairs
    )

    # 混淆方向的总结必须跟着实际数据走，写死会在换模型/换数据后自相矛盾
    if conf_pairs:
        top_txt = "、".join(
            f"「{html.escape(g)}」→「{html.escape(p)}」（{n} 次）" for g, p, n in conf_pairs[:3]
        )
    else:
        top_txt = "无明显集中方向"

    curves_block = (
        f'<section><h2>训练曲线</h2><div class="fig">{img_tag(curves_png)}</div>'
        f'<p class="muted">上排左：训练 / 验证 loss；上排右：验证准确率，虚线是 50 类随机猜的 0.02；'
        f"下排：token 准确率与学习率。看的是趋势和「验证指标什么时候不再涨」，不是绝对值。</p></section>"
        if curves_png
        else ""
    )
    cm_block = (
        f'<section><h2>混淆矩阵</h2><div class="fig">{img_tag(cm_png)}</div>'
        f'<p class="muted">对角线越深越好。非对角线的亮块就是真实被误判成的那一类。</p></section>'
        if cm_png
        else ""
    )

    adapter = ft.get("adapter") or "（未记录）"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>部法分类 QLoRA 微调 · 评测报告</title>
<style>
:root{{--ink:#1f2328;--muted:#5f6368;--line:#e3e6ea;--bg:#fff;--soft:#f6f8fa;--good:#1a7f37;--bad:#b42318;--accent:#185fa5}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:900px;margin:0 auto;padding:48px 28px 72px}}
h1{{font-size:26px;font-weight:600;margin:0 0 6px}}
.sub{{color:var(--muted);font-size:14px;margin-bottom:36px}}
h2{{font-size:17px;font-weight:600;margin:38px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
table{{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{background:var(--soft);font-weight:500;color:var(--muted);font-size:13px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.strong{{font-weight:600}}
td.muted,.muted{{color:var(--muted)}}
.up{{color:var(--good)}} .down{{color:var(--bad)}} .flat{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 8px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card .k{{font-size:12px;color:var(--muted);margin-bottom:6px}}
.card .v{{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}}
.card .d{{font-size:12px;margin-top:4px}}
.fig{{border:1px solid var(--line);border-radius:10px;padding:8px;background:#fff}}
.fig img{{width:100%;display:block;border-radius:6px}}
code{{background:var(--soft);padding:2px 6px;border-radius:4px;font-size:13px}}
.note{{background:var(--soft);border-left:3px solid var(--accent);padding:12px 16px;border-radius:6px;font-size:14px;margin:16px 0}}
ul{{padding-left:20px;margin:10px 0}}
li{{margin:6px 0}}
footer{{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
@media (max-width:640px){{.cards{{grid-template-columns:1fr}}main{{padding:28px 16px}}}}
</style>
</head>
<body>
<main>
<h1>50 部法分类 · QLoRA 微调评测报告</h1>
<div class="sub">任务：一句法律问题 → 50 部法中选 1 部 · 测试集 {ft['n']} 条 · 生成于 {args.date}</div>

<div class="cards">
<div class="card"><div class="k">accuracy</div><div class="v">{pct(ft['accuracy'])}</div><div class="d">{delta(base['accuracy'], ft['accuracy'])}</div></div>
<div class="card"><div class="k">macro-F1</div><div class="v">{pct(ft['macro_f1'])}</div><div class="d">{delta(base['macro_f1'], ft['macro_f1'])}</div></div>
<div class="card"><div class="k">top-3</div><div class="v">{pct(ft['top3'])}</div><div class="d">{delta(base['top3'], ft['top3'])}</div></div>
</div>

<h2>1 · 指标对照</h2>
<table>
<thead><tr><th>指标</th><th>怎么读</th><th class="num">zero-shot 基线</th><th class="num">微调后</th><th class="num">变化</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>
<div class="note">只报微调后的数不算实验。有基线才能回答「这个准确率是模型强，还是任务本来就不难」。</div>

<h2>2 · 实验配置</h2>
<table>
<tbody>
<tr><td>基座</td><td><code>{html.escape(ft['model'])}</code></td></tr>
<tr><td>适配器</td><td><code>{html.escape(str(adapter))}</code></td></tr>
<tr><td>训练方式</td><td>QLoRA 4bit（NF4 + 双量化），LoRA r=16 / α=32，全 linear 层</td></tr>
<tr><td>超参</td><td>lr 2e-4 · cosine + warmup 5% · 2 epoch · max_len 512 · 有效 batch 16</td></tr>
<tr><td>数据</td><td>训练 18,338 / 验证 2,292 / 测试 2,292，50 类，层化切分</td></tr>
<tr><td>评测方式</td><td>50 个候选标签逐个算 logprob 取最大，按 token 数归一化；输出必在标签集内</td></tr>
</tbody>
</table>

{curves_block}

<h2>3 · 表现最差的 5 类</h2>
<table>
<thead><tr><th>部法</th><th class="num">test 条数</th><th class="num">判对</th><th class="num">F1</th></tr></thead>
<tbody>{worst_rows}</tbody>
</table>
<p class="muted">小类只有几条测试样本，错 1 条就是十几个点。看结论时必须连着「test 有几条」一起读。</p>

<h2>4 · 最容易混的 10 对</h2>
<table>
<thead><tr><th>真实</th><th>被误判为</th><th class="num">次数</th><th class="num">占全部错误</th></tr></thead>
<tbody>{conf_rows}</tbody>
</table>
<p class="muted">共 {wrong} 条判错（占 {wrong / ft['n'] * 100:.2f}%）。最主要的混淆方向：{top_txt}——同属一个法域或条文互相援引，边界天然模糊。</p>

{cm_block}

<h2>5 · 结论与局限</h2>
<ul>
<li>微调把 50 选 1 的准确率从 {pct(base['accuracy'])} 提到 {pct(ft['accuracy'])}，macro-F1 从 {pct(base['macro_f1'])} 到 {pct(ft['macro_f1'])}，小类没有被牺牲。</li>
<li>适配器只有几十 MB，基座一字节没改，能卸能装；推理成本不变。</li>
<li>局限一：标签来自数据集的 <code>reference</code> 字段，零人工标注，个别样本的部法归属本身可争议。</li>
<li>局限二：跨两部法的样本按多数取一，80 条这样的样本天然带着噪声标签。</li>
<li>局限三：只做部法分类，不回答法律问题，不给建议——它是下游检索的路由器，不是法律咨询。</li>
</ul>

<footer>数据源 DISC-Law-SFT-Triplet-QA · 基座 Qwen3-1.7B（Apache 2.0）· 由 sft-law 项目脚本生成</footer>
</main>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="生成单文件 HTML 评测报告")
    p.add_argument("--baseline", required=True, help="zero-shot 的 eval_report.json")
    p.add_argument("--finetuned", required=True, help="微调后的 eval_report.json")
    p.add_argument("--curves", default=None, help="plot_training.py 出的 PNG")
    p.add_argument("--out", default="reports/eval_report.html")
    p.add_argument("--date", default="")
    args = p.parse_args()

    from datetime import datetime

    if not args.date:
        args.date = datetime.now().strftime("%Y-%m-%d %H:%M")

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    ft = json.loads(Path(args.finetuned).read_text(encoding="utf-8"))
    if base.get("mode") != "zero-shot":
        print("警告：--baseline 那份报告看起来不是 zero-shot，对照会失真")

    setup_font()
    curves_png = Path(args.curves).read_bytes() if args.curves and Path(args.curves).exists() else None
    cm_png = confusion_png(ft)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args, base, ft, curves_png, cm_png), encoding="utf-8")
    print(f"报告已生成：{out.resolve()}")
    print(f"accuracy {pct(base['accuracy'])} → {pct(ft['accuracy'])} · macro-F1 {pct(base['macro_f1'])} → {pct(ft['macro_f1'])}")
    if not HAS_MPL:
        print("没装 matplotlib，混淆矩阵图跳过了（pip install matplotlib 后重跑）")


if __name__ == "__main__":
    main()
