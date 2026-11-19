# sft-law

法律场景的 **50 部法分类**微调工程。独立项目，不含检索系统、不含业务系统、不依赖任何外部仓库。

---

## 文档地图

| 文档 | 讲什么 | 什么时候看 |
|---|---|---|
| `README.md`（本文） | 项目是什么、目录结构、边界与已量出的数 | 第一次进项目 |
| [`docs/使用说明.md`](docs/使用说明.md) | 从零到出数的手把手：造数据 → 训练 → 评测 → 岔子速查 | 要真的动手 |
| [`docs/finetune-qwen-law-sft.md`](docs/finetune-qwen-law-sft.md) | 脚本参数表、超参、评测算法、常见错法 | 要改参数或查实现 |

---

## 一句话

给一个法律问题，模型直接说出它该走哪部法——50 部里选 1 部。

```
用户提问 → [分类器 · 本项目产出] → 部法名 → 下游才知道去哪本法里找条文
```

它只输出一个词：`劳动法`。不回答用户、不给建议、不写「建议您」、不推理结论。

## 任务契约

| 项 | 内容 |
|---|---|
| 输入 | 一句生活语言的法律问题，实测平均 68.6 字、最长 333 字 |
| 输出 | 一个部法名，必须落在 50 类标签集内 |
| 标签 | 50 部法，来自 Triplet 的 `reference` 字段（零人工标注） |
| 基座 | Qwen3-1.7B（Apache 2.0）+ QLoRA 4bit |
| 框架 | 主线 LLaMA-Factory（AutoDL）+ Unsloth；魔搭上换 ms-swift |
| 指标 | accuracy、macro-F1、top-3、混淆矩阵 |
| 数据 | 22,922 条「问题 → 部法」，train / val / test = 18,338 / 2,292 / 2,292 |

## 为什么是分类，不做问答

同一份数据能训四种东西，差别不在难度，在「做完怎么知道你训成了」：

| 候选任务 | 数据够不够 | 怎么验证 | 结论 |
|---|---|---|---|
| 法律问答 | Pair 7.9 万条够，但答案好坏没标准 | 基本没法量化 | 否 |
| 问题改写 | 缺「口语」那一头 | 要看检索命中，得搭知识库 | 否（降为支线，见下） |
| 意图识别 | 没有现成标签，得另造 | 看分类准确率 | 否 |
| **部法分类** | **reference 里的法条名就是标签** | **准确率一算就有** | **是** |

代价也要认：只做分类、只追推理便宜的话，BERT + 分类头更省。这里选生成式，是为了整门课的主线能贯通——训完的适配器能卸、能合、能继续拿去做生成。

> **两条线的关系**：`data/sft/` 那套 HyDE 改写数据是项目前身（把口语翻成法条语体，靠检索命中率验证）。它没被删，作为支线保留，脚本和数据都在。**当前主线是分类**，下文不带标记的部分一律指分类。

---

## 项目结构

```
sft-law/
├── README.md                         本文件
├── requirements-finetune.txt         手写 transformers+peft 链路的依赖（支线用，非主线）
│
├── course/                           课程物料：课时页与课纲，单文件离线 HTML
│   ├── 法律大模型微调实战-15课时课纲.html   全课规划，权威排课依据
│   ├── L00_项目整体介绍_课时页.html        项目线第 1 课：任务 / 数据 / 尺子全景
│   ├── L00_微调全景与前导_课时页.html      理论线前导：概念铺垫（与上一课编号重名，待重排）
│   ├── L01_微调选型_课时页.html           八环决策链：需求→任务、基座→平台
│   ├── L02_需求与数据_课时页.html         需求定位、验收前置、数据决定任务上限
│   ├── L03_环境框架与脚本_课时页.html     平台配方法、三个入口、配置与脚本
│   ├── L04_第一次微调跑通_课时页.html     魔搭 ms-swift：从注册到跑通
│   ├── L05_AutoDL手把手微调_课时页.html   AutoDL + LLaMA-Factory：从开机到出适配器
│   ├── L06_魔搭实战_部法分类_课时页.html   分类任务在魔搭上的完整一遍
│   ├── L07_AutoDL实战_部法分类_课时页.html 分类任务在 AutoDL 上的完整一遍
│   ├── _backup/  .bak/                历史版本堆，不当正式文件读
│
├── data/
│   ├── raw/                          原始数据，只读。清洗脚本一条都不改它
│   │   ├── DISC-Law-SFT-Triplet-QA-released.jsonl   82M / 23,331 条，本项目唯一数据源
│   │   └── DISC-Law-SFT-Pair-QA-released.jsonl      90M / 79,692 条，本项目不用
│   │
│   ├── cls/                          【主线】50 部法分类数据集
│   │   ├── train.jsonl               18,338 条，query / response 两字段
│   │   ├── val.jsonl                  2,292 条，同上
│   │   ├── test.jsonl                 2,292 条，同上
│   │   ├── label_stats.json           50 类清单 + 每类条数 + 切分统计（候选打分读这个，别手抄标签）
│   │   └── export/                    两个平台各自吃的格式，由 export_platform_data.py 生成
│   │       ├── ms_swift/train.jsonl           query/response，魔搭原样吃
│   │       ├── ms_swift/val.jsonl             同上
│   │       ├── llamafactory/cls_train.jsonl   instruction/output，AutoDL 吃
│   │       ├── llamafactory/cls_val.jsonl     同上
│   │       ├── llamafactory/dataset_info.json 贴进 LLaMA-Factory/data/dataset_info.json 的片段
│   │       └── test_eval.jsonl                2,292 条，question / label 两字段，eval_cls.py 的输入
│   │
│   ├── sft/                          【支线】HyDE 改写数据集
│   │   ├── train.jsonl               21,785 条，问题 → 法条语体
│   │   ├── val.jsonl                  1,146 条
│   │   ├── demo_hyde_500.jsonl        L04 课堂演示用 500 条（query/response）
│   │   ├── demo_hyde_500_alpaca.jsonl L05 课堂演示用 500 条（instruction/output）
│   │   └── demo_hyde_raw.jsonl        600 条原始抽取，造上面两份的中间产物
│   │
│   ├── corpus/articles.jsonl         6,432 条法条语料（50 部法，支线跑检索基线用）
│   ├── eval/                         支线的评测集与基线结果
│   │   ├── evalset.jsonl              400 条书面语评测集
│   │   ├── evalset_book40.jsonl        40 条对照组
│   │   ├── evalset_oral.jsonl          40 条人工改写的口语版
│   │   └── baseline*.json             三次检索基线的结果
│   └── external/DISC-Law-Eval/       复旦官方评测集（2,563 + 300），遗忘检测用
│
├── scripts/                          全部脚本，每个文件头部有中文 docstring
│   ├── build_cls_dataset.py          【主线】造 50 类数据集：剥法条 → 取多数 → 剔考卷 → 切分
│   ├── export_platform_data.py       【主线】导出 ms-swift / LLaMA-Factory 两种格式 + test_eval
│   ├── eval_cls.py                   【主线】候选打分评测：accuracy / macro-F1 / top-3 / 混淆矩阵
│   │
│   ├── build_baseline.py             【支线】抽法条语料 + 抽评测集 + 跑检索基线（corpus/evalset/run）
│   ├── build_trainset.py             【支线】造 HyDE 训练集：问题 → 法条语体
│   ├── prepare_hyde_sft_data.py      【支线·旧】早期 HyDE 数据构造，口径已被 build_trainset 取代
│   ├── train_qwen_hyde_lora.py       【支线·参照】手写 transformers+peft 训 HyDE，非主线
│   ├── infer_qwen_hyde_lora.py       【支线·参照】HyDE 推理，肉眼验训练效果
│   ├── train_qwen_law_sft_lora.py    【支线·废弃】法律问答 SFT，已验不可量化，留档
│   ├── infer_qwen_law_lora.py        【支线·废弃】法律问答推理，留档
│   ├── merge_qwen_lora.py            【通用】把 adapter 合进基座，出完整 HF 目录
│   │
│   ├── lint_lesson.py                【课程工具】课时页机检 12 项，改完必跑
│   ├── check_bilingual.py            【课程工具】专有名词首现是否中英并列
│   ├── first_occur.py                【课程工具】定位术语首现行号（标 svg 内/外）
│   ├── l02_tiny_lm.py                【课程工具】L02 配套 bigram 小实验，CPU 几秒出 loss
│   ├── probe_overflow.py             【课程工具】窄屏溢出探针
│   └── probe_overflow2.py            【课程工具】窄屏溢出探针 v2，iframe 强出真实窄视口
│
└── docs/
    └── finetune-qwen-law-sft.md      HyDE / 问答两条旧链路的说明，口径已过时
```

---

## 数据：从 Triplet 到 50 类标签

原始 Triplet 一条样本里塞着几条法条原文、一个真问题、一段答案。我们只取两样：

```
一条 Triplet  →  剥出问题（只留「问题：」后的）  →  抽部法名（《X》第Y条 → X）  →  问题 → 部法
```

清洗只剩三条规矩，每条都对应一个坑：

| 规矩 | 为什么 | 不做会怎样 |
|---|---|---|
| 必须剥掉 input 里的法条 | 法条原文是答案，留着等于把答案喂给模型 | 模型学会抄法条，不学会判部法 |
| 多部法的样本取多数 | 80 条样本 reference 跨两部法，单标签只留一个 | 标签打架，训练信号乱 |
| 评测集问题必须剔除 | evalset 与训练集同源，不剔就是偷看考卷 | 准确率虚高，上真实数据就掉 |

加工完的一条长这样：

```json
{"query": "请判断以下法律问题应适用哪部法律。只输出法律名称，不要解释。\n\n问题：小王是一家个人独资企业的投资人，他的企业已经解散，但仍有未偿还的债务，他需要承担什么样的偿还责任？收回的财产如何分配？", "response": "个人独资企业法"}
```

两个刻意的选择：**指令只有两句**（指令越长模型越爱补解释，候选打分不好比）；**答案只有一个词**（放整句会让各候选 token 数不一致，长短没法比）。

原始 `reference` 里的法条名本身很干净，50 部法没有一个带「中华人民共和国」前缀，不用额外归一化。这一点省了不少事，但换数据时要先查一遍。

### 50 类有多不均衡

| 位置 | 部法 | 样本数 | 说明 |
|---|---|---|---|
| 1 | 民法典 | 3,727 | 占全部约 16% |
| 2 | 刑法 | 1,800 | 与刑事诉讼法只差两个字 |
| 3 | 刑事诉讼法 | 1,131 | 极易与刑法混 |
| … | 中间 44 类 | 94 ~ 889 | — |
| 50 | 公益事业捐赠法 | 93 | 进训练集只剩 70 条，进 test 只剩 8 条 |

两个后果：**accuracy 会被大类带偏**（必须同时看 macro-F1）；**小类结论不稳**（8 条里错 1 条就是 12.5 个点，报数必须带「这一类 test 有几条」）。

---

## 脚本怎么用

整条线六步，每一步都留一个「别人能复算」的东西：

| 步 | 做什么 | 产出 | 怎么算过 |
|---|---|---|---|
| 1 | 造数据 | `data/cls/` 三份 jsonl + 标签统计 | 50 类无空类、三份零重叠 |
| 2 | 量基线 | zero-shot 的 accuracy / macro-F1 | 有具体数字落在报告里 |
| 3 | 训模型 | LoRA 适配器 + 训练日志 | val 准确率不再涨 |
| 4 | 评效果 | test 三指标 + 混淆矩阵 | 能指到最差的两类 |
| 5 | 做对比 | 基线 vs 微调 一张对照表 | 提升幅度说得清 |
| 6 | 交付 | 模型 + 数据 + 评测报告 | 别人照着能复现这组数 |

最容易塌方的是跳过第 2 步：训完 85% 挺高兴，但没人知道不训是多少——这个 85% 是模型强还是任务简单，说不清。

### 主线：造数据 → 出平台格式 → 评测

```bash
# 1. 造 50 类数据集（本机即可，不需要 GPU）
python scripts/build_cls_dataset.py
#   → data/cls/{train,val,test}.jsonl + label_stats.json
#   默认剔掉 data/eval/ 下的评测集问题，80/10/10 分层切分

# 2. 导出两个平台吃的格式
python scripts/export_platform_data.py
#   → data/cls/export/ms_swift/、llamafactory/、test_eval.jsonl

# 3. 评测（要 GPU）
python scripts/eval_cls.py --no-adapter              # 先量 zero-shot 基线
python scripts/eval_cls.py --adapter <适配器目录>     # 再量微调后
#   → accuracy / macro-F1 / top-3 / 每类准确率 / 混淆矩阵，写 eval_report.json
```

### 平台一：魔搭（ms-swift，免费 A10 24G）

```bash
pip install ms-swift -U
# 数据放 /mnt/workspace/cls/，产物落 /mnt/workspace/output/cls_qwen3
swift sft \
  --model Qwen/Qwen3-1.7B \
  --tuner_type lora \
  --quant_method bnb --quant_bits 4 \
  --dataset /mnt/workspace/cls/train.jsonl \
  --val_dataset /mnt/workspace/cls/val.jsonl \
  --num_train_epochs 2 --learning_rate 2e-4 --max_length 512 \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --output_dir /mnt/workspace/output/cls_qwen3
```

注意：ms-swift 没有 `qlora` 这个 tuner，QLoRA = `--tuner_type lora` + `--quant_method bnb --quant_bits 4`。

### 平台二：AutoDL（LLaMA-Factory + Unsloth，4090 24G）

```bash
# 数据放 /root/autodl-tmp/LLaMA-Factory/data/，把 dataset_info.json 片段并进去
llamafactory-cli train cls_qlora.yaml
# 产物落 /root/autodl-tmp/saves/qwen3-1.7b/cls/checkpoint-*
llamafactory-cli chat cls_qlora.yaml        # 试聊
```

yaml 里 dataset 填**注册名**（`cls_law`），不是文件路径——这是最常见的报错来源。完整参数、自检命令、混淆矩阵读法见 [`docs/finetune-qwen-law-sft.md`](docs/finetune-qwen-law-sft.md)。

训练配置（两个平台同一套）：基座 Qwen3-1.7B、QLoRA 4bit（NF4 + 双量化）、r=16 / α=32 挂 attention + FFN、lr 2e-4 cosine + warmup、epoch 2、max_len 512。

显存账（24G 卡真实占用不到三分之一）：基座 ≈0.9G + LoRA 权重/梯度/优化器 ≈0.3G + 激活 ≈2~4G + CUDA 上下文 ≈1~2G = **约 5~7G**。

> 别在现场跑「全参 vs QLoRA」对照：全参 1.7B 要 20G 以上，24G 卡卡在 OOM 边缘，一崩就掉课堂节奏。全参只讲数字。

---

## 评测：不让模型自由生成

自由生成最大的坑是「说漏嘴」——模型可能吐「民法典第一编」，不在 50 类里，字符串一匹配就判错，其实它知道答案。所以改成 **50 个候选逐个打分取最大**，输出必在标签集内，还能顺手算 top-3。

**必须按 token 数取平均**：

```python
score = sum(logprob(t) for t in label_tokens) / len(label_tokens)
```

「民法典」3 个 token、「国际刑事司法协助法」9 个，直接加总比，长标签天然吃亏——混淆矩阵里那批「本该判长名、却判给短名」的错案，原因不在模型，在打分代码。

四把尺子缺一把都会误判：

| 指标 | 回答什么问题 | 怎么读 |
|---|---|---|
| accuracy | 50 选 1 整体对了多少 | 主指标，但会被大类带偏 |
| macro-F1 | 每一类平均好不好 | 小类错了立刻掉，最公平 |
| top-3 | 差一点就对的有多少 | 看「接近对」的能力 |
| 混淆矩阵 | 具体混在哪两对 | 重点看民法典/民诉、刑法/刑诉 |

---

## 已经量出来的数

### 分类线

| 项 | 数 | 来源 |
|---|---|---|
| 清洗后样本 | 22,922 条 / 50 类 | `label_stats.json` |
| 剔除评测集 | 400 条 | 同上 |
| 跨两部法样本 | 80 条（取多数） | 同上 |
| 重复问题 | 9 条 | 同上 |
| zero-shot 基线准确率 | **待补**（需 GPU） | — |
| 微调后 test 指标 | **待补**（需 GPU） | — |

### 支线（HyDE 改写器，已跑完）

法条库 6,432 条、模型 `BAAI/bge-small-zh-v1.5`、top-5：

| 评测集 | hit_rate@5 | 首位命中 | MRR |
|---|---|---|---|
| 书面语 400 条 | 0.7975 | — | 0.6538 |
| 书面语 40 条（对照） | 0.775 | 87% | 0.7063 |
| 口语 40 条（同批改写） | 0.725 | 66% | 0.5792 |

两个结论：口语化之后命中率只掉 5 个点，但首位命中从 87% 掉到 66%——语义鸿沟主要伤排序，不只伤召回；40 条里 7 条两种问法都失败，翻标注发现是标错了，**未命中不等于模型差**。

---

## 三条边界，越界就要重做

1. **输入里不许有答案**——问题里夹着法条原文，模型就只在抄，不在判。
2. **评测集不许进训练**——evalset 哪怕只漏几条进 train，准确率就不可信。
3. **没有基线不许报提升**——只报「微调后 88%」不报「基线多少」，等于没做实验。

## 环境

```bash
pip install -r requirements-finetune.txt    # 支线手写链路用
pip install sentence-transformers           # 只有跑检索基线才需要

# 国内下载模型先设镜像
export HF_ENDPOINT=https://hf-mirror.com    # Windows: set HF_ENDPOINT=...
```

主线训练在平台上跑，本机只需要 Python 造数据。
