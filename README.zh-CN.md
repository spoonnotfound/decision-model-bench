# Decision Model Bench

统一评测**离散决策模型**：给出状态、问题和候选项，比较判断准确率，并提供可复现的逐题结果。

[English README](README.md) · [评测方法](docs/methodology.md) · [扩展指南](docs/extending.md)

第一版支持候选项分类、二分类及明确的标签合并。连续评分、序数评分、多轮智能体和跨任务总榜尚未实现。

## 直接复算，不需要模型和 API Key

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
dmb validate --task tasks/turtlebench-en.json
dmb score --task tasks/turtlebench-en.json \
  --records results/turtlebench-en-20260920/jev.jsonl
pytest -q
```

仓库提供固定版本的英文 TurtleBench 数据和经过整理的历史逐题结果。这是离线重新计分，不是假装重新调用模型。

## 首个参考实验

2026-09-20，三款模型使用同样的 1,532 道英文海龟汤裁判题，来自 [TurtleBench 开源项目](https://github.com/mazzzystar/TurtleBench) 的 32 个故事。本次对比聚焦判断准确率：

|模型|答对|准确率|
|---|---:|---:|
|Jev（Vercel AI Gateway）|1,299|84.79%|
|Laya 英文基础版|1,009|65.86%|
|Laya typed-decisions|956|62.40%|
|始终选否定类的基线|886|57.83%|

这是框架建立前完成的探索性实验，现已导入并通过测试重新计分。**结果只说明这套任务上的表现，不是通用能力排名，也不是对 Laya 原 typed-decisions 基准宣传的复现或证伪。**

- 模型拿到汤面、汤底和玩家猜测，承担裁判工作。先选三类，再把 Incorrect / Unknown 合并计分；高分不能证明它能分清“否”和“信息不足”。
- 两款 Laya 使用官方完整、固定版本权重，保留官方温度配置，没有量化或针对海龟汤微调。
- Jev 使用模型标识 `typesafe-ai/jev`，底层精确版本号未知。
- 数据集经过困难样本筛选，不能把错误率等同于正常游戏体验；公开数据是否进入训练未知。

详细证据见 [实验说明](results/turtlebench-en-20260920/study.json)、[完整指标](results/turtlebench-en-20260920/summary.json) 和 [方法说明](docs/methodology.md)。

## 重新跑模型

```bash
# Laya：先少量检查，再新建目录跑全量。
pip install -e '.[laya]'
dmb run --task tasks/turtlebench-en.json --config configs/laya-typed.toml \
  --out runs/laya-smoke --limit 2

# Jev 网关：需要 Node.js 24+，并在环境变量中设置 AI_GATEWAY_API_KEY。
npm ci --prefix bridge
dmb run --task tasks/turtlebench-en.json --config configs/jev-gateway.toml \
  --out runs/jev-smoke --limit 2 --live
```

模型配置中可明确选择 CPU / CUDA / MPS。接口不支持或文本超长会留下失败记录，不会悄悄截断故事。远程调用必须传 `--live`，没有隐藏重试；`--retry-failed` 才会重新尝试失败题，原失败记录保留。

断点恢复会检查输入、配置、代码和依赖版本的指纹；修改条件后需使用新目录。新增任务只需提供 JSON 配置和 JSONL 数据；新增模型实现统一的适配器接口，详见 [扩展指南](docs/extending.md)。

第一版借鉴 lm-evaluation-harness 的任务/模型分离、Inspect 的日志与执行结构、jev-benchmarks 的概率评估思路。完整引用及许可见 [英文 README](README.md#design-references-and-attribution) 与 [NOTICE](NOTICE)。
