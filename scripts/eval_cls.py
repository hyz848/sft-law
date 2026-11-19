"""50 类部法分类 · 候选打分评测。

不给模型自由生成，而是把 50 个部法名逐个喂进去算 logprob，取最大——输出必在标签集内，
还能顺手算 top-3。logprob 按 token 数取平均，否则短标签会被系统性偏爱。

用法：
    # 微调之后（带 adapter）
    python scripts/eval_cls.py --adapter output/checkpoint-600

    # zero-shot 基线（不带 adapter，先跑这个）
    python scripts/eval_cls.py --no-adapter

输出：accuracy / macro-F1 / top-3 / 每类准确率 / 混淆矩阵（写 eval_report.json）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DATA = str(ROOT / "data" / "cls" / "export" / "test_eval.jsonl")
DEFAULT_LABELS = str(ROOT / "data" / "cls" / "label_stats.json")
DEFAULT_PROMPT = "请判断以下法律问题应适用哪部法律。只输出法律名称，不要解释。"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="50 类部法分类 · 候选打分评测")
    p.add_argument("--model", default=DEFAULT_MODEL, help="基座模型路径或 HF id")
    p.add_argument("--adapter", default=None, help="LoRA 适配器目录；不传则用 --no-adapter")
    p.add_argument("--no-adapter", action="store_true", help="跑 zero-shot 基线")
    p.add_argument("--data", default=DEFAULT_DATA, help="评测集 jsonl（question / label）")
    p.add_argument("--labels-from", default=DEFAULT_LABELS, help="标签来源（label_stats.json）")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="与训练时一致的指令前缀")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试水用）")
    p.add_argument("--out", default=str(ROOT / "data" / "cls" / "eval_report.json"))
    return p.parse_args()


def load_rows(path: str, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def load_labels(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = data.get("labels") if isinstance(data, dict) else None
    if not labels:
        sys.exit(f"标签文件里没有 labels 字段：{path}")
    return sorted(labels, key=lambda x: -labels[x])


def build_prompt_ids(tokenizer, instruction: str, question: str) -> list[int]:
    """按训练时的 qwen3 模板构造 prompt（关闭 thinking，与训练一致）。"""
    messages = [{"role": "user", "content": f"{instruction}\n\n问题：{question}"}]
    kwargs = dict(add_generation_prompt=True, tokenize=True, return_tensors=None)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # 老版本 tokenizer 不认 enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    args = parse_args()

    if args.adapter is None and not args.no_adapter:
        sys.exit("要么给 --adapter <目录>，要么显式加 --no-adapter 跑基线")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"缺依赖（{exc}）：pip install torch transformers peft accelerate")

    labels = load_labels(args.labels_from)
    rows = load_rows(args.data, args.limit)
    print(f"评测集 {len(rows)} 条 · 标签 {len(labels)} 类 · 基座 {args.model}")
    print(f"模式：{'zero-shot 基线' if args.no_adapter else '微调后 ' + str(args.adapter)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 候选标签的 token 序列先编码好，避免每条样本重复 tokenize
    label_ids = {lab: tokenizer(lab, add_special_tokens=False).input_ids for lab in labels}
    eos = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    correct = 0
    top3_hits = 0
    per_class_total: Counter[str] = Counter()
    per_class_hit: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            prefix = build_prompt_ids(tokenizer, args.prompt, row["question"])

            # 50 个候选拼成一个 batch，一次前向算完，比逐个候选快约 50 倍
            seqs = [prefix + label_ids[lab] + ([eos] if eos is not None else []) for lab in labels]
            maxlen = max(len(s) for s in seqs)
            padded = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
            input_ids = torch.tensor(padded, device=model.device)
            attention_mask = (input_ids != pad_id).long()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)

            scores: dict[str, float] = {}
            offset = len(prefix) - 1
            for row_i, lab in enumerate(labels):
                ids = label_ids[lab] + ([eos] if eos is not None else [])
                total = 0.0
                for j, tid in enumerate(ids):
                    total += logprobs[row_i, offset + j, tid].item()
                # 长度归一化：按 token 数取平均
                scores[lab] = total / len(ids)

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            pred = ranked[0][0]
            gold = row["label"]
            per_class_total[gold] += 1
            confusion[gold][pred] += 1
            if pred == gold:
                correct += 1
                per_class_hit[gold] += 1
            if gold in [x[0] for x in ranked[:3]]:
                top3_hits += 1

            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  accuracy={correct / i:.4f}")

    n = len(rows)
    accuracy = correct / n if n else 0.0
    top3 = top3_hits / n if n else 0.0

    # macro-F1：每类 F1 的算术平均，小类与大类同权
    f1s: dict[str, float] = {}
    for lab in per_class_total:
        tp = per_class_hit[lab]
        fp = sum(confusion[g][lab] for g in confusion if g != lab)
        fn = per_class_total[lab] - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0

    worst = sorted(f1s.items(), key=lambda kv: kv[1])[:5]

    print("\n" + "=" * 46)
    print(f"accuracy : {accuracy:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")
    print(f"top-3    : {top3:.4f}")
    print("-" * 46)
    print("F1 最差的 5 类（含 test 条数）：")
    for lab, f1 in worst:
        print(f"  {lab}: F1={f1:.3f}  test={per_class_total[lab]} 条")
    print("=" * 46)

    report = {
        "model": args.model,
        "adapter": None if args.no_adapter else args.adapter,
        "mode": "zero-shot" if args.no_adapter else "finetuned",
        "n": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "top3": round(top3, 4),
        "per_class": {
            lab: {
                "test": per_class_total[lab],
                "hit": per_class_hit[lab],
                "f1": round(f1s.get(lab, 0.0), 4),
            }
            for lab in per_class_total
        },
        "confusion": {g: dict(c) for g, c in confusion.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()
