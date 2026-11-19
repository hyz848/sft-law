#!/usr/bin/env python3
"""对已部署的 OpenAI 兼容接口（vLLM）跑测试集，出准确率——作业第 2 题要的那个数。

用法：
    python scripts/eval_api.py \
      --base-url http://127.0.0.1:6006/v1 \
      --model lawcls \
      --limit 200

前提：服务端已用 --enable-lora 加载适配器，模型名就是你起的 --lora-modules 名字。
本机也能跑：AutoDL 的自定义服务给了公网地址，把 --base-url 换成它即可。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("缺 requests：pip install requests")

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "请判断以下法律问题应适用哪部法律。只输出法律名称，不要解释。"


def load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels") if isinstance(data, dict) else None
    if not labels:
        sys.exit(f"标签文件里没有 labels 字段：{path}")
    return sorted(labels, key=lambda x: -labels[x])


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def ask(base_url: str, model: str, question: str, timeout: int = 60) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{PROMPT}\n\n问题：{question}"}],
        "temperature": 0,
        "max_tokens": 24,
        # Qwen3 默认会先输出一段思考，不关掉准确率会莫名其妙地崩
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def extract(raw: str, labels: list[str]) -> str | None:
    """模型自由生成，宽松匹配：先精确，再包含。都匹配不到算判错。"""
    text = raw.strip().strip("《》「」\"'。.,， \n\t")
    if text in labels:
        return text
    for lab in labels:
        if lab in text:
            return lab
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="调部署好的接口，跑测试集准确率")
    p.add_argument("--base-url", default="http://127.0.0.1:6006/v1")
    p.add_argument("--model", default="lawcls", help="vLLM --lora-modules 起的名字")
    p.add_argument("--data", default=str(ROOT / "data/cls/export/test_eval.jsonl"))
    p.add_argument("--labels-from", default=str(ROOT / "data/cls/label_stats.json"))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条，试水用")
    p.add_argument("--out", default=str(ROOT / "data/cls/eval_report_api.json"))
    p.add_argument("--sleep", type=float, default=0.0, help="每条之间的间隔秒数")
    args = p.parse_args()

    labels = load_labels(Path(args.labels_from))
    rows = load_rows(Path(args.data), args.limit)
    print(f"接口 {args.base_url} · 模型 {args.model} · 测试 {len(rows)} 条 · 标签 {len(labels)} 类")

    # 先探一条，接口或模型名不对就立刻失败，别跑一半才发现
    try:
        probe = ask(args.base_url, args.model, rows[0]["question"])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"接口不通：{exc}\n检查 --model 是不是 vLLM 里 --lora-modules 的名字、端口映射是否生效")
    print(f"探针输出：{probe!r}")

    correct = 0
    parsed_out = 0
    errors = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        try:
            raw = ask(args.base_url, args.model, row["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {i} 请求失败：{exc}")
            raw = ""
        pred = extract(raw, labels)
        if pred is None:
            parsed_out += 1
        if pred == row["label"]:
            correct += 1
        elif len(errors) < 10:
            errors.append({"question": row["question"][:80], "gold": row["label"], "pred": pred, "raw": raw[:40]})
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  accuracy={correct / i:.4f}")
        if args.sleep:
            time.sleep(args.sleep)

    n = len(rows)
    acc = correct / n if n else 0.0
    report = {
        "mode": "api",
        "base_url": args.base_url,
        "model": args.model,
        "n": n,
        "accuracy": round(acc, 4),
        "unparsed": parsed_out,
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 46)
    print(f"accuracy : {acc:.4f}  （{correct}/{n}）")
    print(f"无法匹配到标签的输出：{parsed_out} 条")
    print(f"耗时 {report['elapsed_s']}s · 报告 {args.out}")
    print("=" * 46)
    if parsed_out > n * 0.05:
        print("提示：超过 5% 的输出匹配不到任何标签，多半是 thinking 没关，或 prompt 和训练时不一致。")
    if errors:
        print("\n前几条错例：")
        for e in errors[:5]:
            print(f"  真值 {e['gold']} · 预测 {e['pred']} · 原始输出 {e['raw']!r}")


if __name__ == "__main__":
    main()
