# 部署上线：vLLM + LoRA 适配器（AutoDL）

作业第 2 题要的是「模型能对外提供服务」+「测试集上的准确率」。前者靠 vLLM 起 OpenAI 兼容接口，后者靠 `scripts/eval_api.py` 打这个接口。

## 为什么选 AutoDL 不选阿里云 PAI-EAS

| | AutoDL | 阿里云 PAI-EAS |
|---|---|---|
| 计费 | 4090 按量约 1.6~2.6 元/时 | 最低配 GPU 实例按小时算，通常几十元起 |
| 起服务 | 一条命令，6006 端口自带公网映射 | 要建服务、配镜像、配网络，控制台点十几个页面 |
| 适合 | 课程作业、一次性验收 | 长期在线业务 |

做完这一套，1 小时以内，花费 5 元以内。

## 1. 租实例

AutoDL 控制台 → 租用实例 → 选 **RTX 4090 / 24G** → 镜像选 **PyTorch 2.5.0 / Python 3.12 / CUDA 12.4** 基础镜像（不要选带一堆预装包的，容易和 vLLM 冲突）。

开机后进 JupyterLab，打开终端。

## 2. 装 vLLM

```bash
# Qwen3 需要 vLLM >= 0.8.5，直接装最新
pip install -U vllm -i https://mirrors.aliyun.com/pypi/simple/
```

装完验证：

```bash
python -c "import vllm; print(vllm.__version__)"
```

## 3. 把适配器传上去

从魔搭那边把 `checkpoint-1900` 里的两个文件打包下载，再传到 AutoDL：

```
/root/autodl-tmp/adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

**只传这两个文件**。整个 checkpoint 里还有优化器状态（`optimizer.pt`），几百 MB，部署用不上。

JupyterLab 左侧文件树可以直接拖拽上传；命令行用 `scp` 也行。

## 4. 起服务

```bash
export VLLM_USE_MODELSCOPE=true   # 从魔搭拉基座，国内快得多

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --enable-lora \
  --lora-modules lawcls=/root/autodl-tmp/adapter \
  --max-lora-rank 16 \
  --host 0.0.0.0 --port 6006 \
  --gpu-memory-utilization 0.85
```

- `lawcls` 是服务里的模型名，后面调接口就用它
- `--max-lora-rank 16` 必须 ≥ 训练时的 `--lora_rank`，写小了加载直接报错
- 首次启动要下基座（约 4GB），等 2~5 分钟看到 `Uvicorn running on http://0.0.0.0:6006` 才算起来

验证：

```bash
curl http://127.0.0.1:6006/v1/models
```

## 5. 映射公网端口

AutoDL 实例详情页 → **自定义服务** → 端口填 `6006` → 得到形如
`https://u123456-xxxx.westb.seetacloud.com:8443` 的地址。

这就是「上线」的证据。接口根路径是 `<该地址>/v1`。

## 6. 跑测试集出准确率

在 AutoDL 终端里（项目代码用 git clone 下来）：

```bash
git clone -b workbuddy/modelscope-run https://github.com/hyz848/sft-law.git sft-law
cd sft-law
git lfs pull

python scripts/eval_api.py \
  --base-url http://127.0.0.1:6006/v1 \
  --model lawcls \
  --limit 200
```

跑 200 条约 1~2 分钟。确认没问题再去掉 `--limit` 跑全量 2,292 条。

结果写进 `data/cls/eval_report_api.json`，和 `eval_cls.py` 那份（本地加载适配器的评测）互为印证——两份差得不多，说明部署没出岔子。

## 一定会踩的坑

**thinking 没关，准确率断崖式下跌。**
Qwen3 默认先输出一段思考再给答案，`eval_api.py` 里已经通过 `chat_template_kwargs.enable_thinking=False` 关掉了。你如果用 `curl` 手测，记得也带上，否则会看到模型洋洋洒洒分析半天然后输出「民法典」。

**模型名填成基座名。**
`--model lawcls` 是 LoRA 的名字。填 `Qwen/Qwen3-1.7B` 请求的是没微调的基座，出来的准确率会退回基线水平。

**端口不是 6006。**
AutoDL 的自定义服务只认 6006，vLLM 起在别的端口上映射不过去。

**bfloat16 不支持的卡。**
4090 支持 bf16，没问题。如果换成 V100/T4 要加 `--dtype float16`。

## 作业提交清单

| 题目 | 交什么 |
|---|---|
| 第 1 题：Adapter 文件包 | `adapter_config.json` + `adapter_model.safetensors` 打成的 zip/tar.gz |
| 第 2 题：部署上线 | 自定义服务的公网地址 + `curl /v1/models` 的返回截图 |
| 第 2 题：评估报告 | `reports/eval_report.html`（含基线对照、混淆矩阵、最差类别）+ `eval_report_api.json` |
