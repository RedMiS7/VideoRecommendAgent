# 推荐 Agent 的四项 AI Eval 指标

日期：2026-10-08。

## What was added

默认 `video-agent eval` 使用用户画像运行现有 `Recommender.run`，保存实际前 5 名，
报告仅包含以下四组指标：Precision@5、NDCG@5、Diversity@5、推荐成本与延迟。
每个画像使用独立数据库与空缓存，不修改真实用户记忆，也不在画像之间共享推荐历史。
冻结语料替代联网搜索、元数据和字幕下载；查询生成、初筛、字幕索引、画像检索、
最终排序仍运行应用代码。此次不改变推荐算法，不调用 LLM 评判模型。

JSON / Markdown 报告包含逐画像结果、dev/test 分组、人工标签状态、调用原始用量、
模型配置、数据及单价文件哈希、提示词指纹和用于多样性计算的摘要向量。
不设置自动通过阈值，结果用于后续版本之间的比较。

## Why it exists

推荐场景的输入是用户兴趣、学习目标、水平和偏好。评估同样使用这些输入，
并检查最终推荐列表是否适合用户、顺序是否合理、内容是否重复、生成一轮的资源消耗。
人工标签是“这个视频对这个画像有多适合”，允许多个正确视频；不需要编写问答检索题。

## 指标口径

### 1. Precision@5：推荐准确率

先人工给每个“画像 × 候选视频”打 0–3 分：

| 分数 | 含义 | Precision 中算相关吗？ |
| --- | --- | --- |
| 0 | 无关或违反明确约束 | 否 |
| 1 | 弱相关，同主题但不满足主要学习需求 | 否 |
| 2 | 满足主要学习需求 | 是 |
| 3 | 高度匹配目标、水平和偏好 | 是 |

`Precision@5 = 前 5 名中人工分数 ≥2 的视频数量 / 5`。
例如 5 个推荐中 4 个适合，结果为 0.8。仅返回 2 个且都适合时结果为 0.4，
空位计为未命中。候选库里合适视频不足 5 个时，最高值也会受到限制。

### 2. NDCG@5：排序质量

使用同一份人工分数，计算：

```text
DCG@5 = Σ [(2^grade_i - 1) / log2(i + 1)]，i 为 1–5 的排名
IDCG@5 = 全部候选按人工分数从高到低排列时的理想 DCG@5
NDCG@5 = DCG@5 / IDCG@5
```

范围为 0–1。更适合的视频排在前面时分数更高。IDCG 使用全部候选，
因此必须给每个候选打分，不能只标推荐出来的 5 个，否则容易高估。
若全部候选均为 0 分，则 NDCG 无定义，显示 `null`，不进入该项平均值。
推荐执行失败按空列表计算：有非零金标准时 NDCG=0、Precision=0，保留在分母中。

上述指数增益与对数折扣可参见
[Recommenders 的 NDCG 实现](https://github.com/recommenders-team/recommenders/blob/main/recommenders/evaluation/python_evaluation.py)。

### 3. Diversity@5：内容语义差异

复用推荐过程中生成的字幕摘要 Embedding，计算前 5 名视频之间的平均两两余弦距离：

```text
Diversity@5 = 所有视频对的 (1 - cosine(summary_vector_a, summary_vector_b)) 的平均值
```

5 个视频共有 10 对；不足 5 个时按实际视频对计算，并展示返回条数。
相同方向的向量距离为 0，正交向量为 1，理论范围为 0–2，不是百分比。
少于 2 个推荐，或任一推荐缺少字幕摘要向量时，显示 `null`，不填成 0。
计算过程不追加模型请求。不同主题的无关视频也可能获得高多样性分数，
所以应结合 Precision 与 NDCG 阅读；多样性没有独立的“越高越好”验收目标。

### 4. 推荐成本与延迟

每轮记录模型请求数、输入/输出/总 token、估算模型费用与推荐耗时；
汇总平均 token、平均费用、总费用、平均耗时、P50、P95。
记录包括重试请求和 Embedding 请求；任一请求缺少用量或单价时，
对应整轮费用显示未知，整体费用也不按已知子集冒充完整总额。

费用使用你实际服务的单价：

`估算模型费用 = Σ (输入 token × 输入每百万单价 + 输出 token × 输出每百万单价) / 1,000,000`。

每个画像从空缓存开始，耗时与模型成本包含首次分块、摘要和索引。
不包含冻结语料之前的采集、联网搜索费用、下载延迟、服务初始化和报告写入。
因此它是固定候选库下的首次推荐测量，不能直接作为线上完整请求或已有缓存的日常耗时。
单价按固定输入/输出费率估算，不自行推断供应商的缓存折扣、套餐或阶梯计费。

## How to test or use it

### 1. 无凭据验证流程

在项目根目录执行：

```powershell
uv run video-agent eval --offline
```

默认使用 `evals/recommendation-cases.jsonl`、同目录 `videos.json` 和 `manifest.json`。
这是一份 6 视频 / 4 画像的合成示例，数字标签是待人工复核的草稿。
离线得分仅验证计算与流程；token 和金额为未知，离线耗时不代表真实服务延迟。
报告路径由命令返回，默认保存在 `reports/`。

### 2. 准备真实视频库与画像标签

单画像可使用新增的自动采集脚本搜索并采集 30 个有字幕的视频，
步骤见 [按画像采集视频库](video-corpus-crawler.md)。

可以沿用采集命令：

```powershell
uv run video-agent eval-collect --bvid-file data/eval-bvids-20261005.txt --output data/recommendation-corpus-v1
```

`videos.json` 和 `manifest.json` 同目录放置 `cases.jsonl`。每条为一个 JSON 对象，例如：

```json
{
  "id": "rag-practical-test",
  "type": "recommendation",
  "split": "test",
  "label_status": "needs_human_review",
  "profile": {
    "topics": ["RAG"],
    "goal": "实现一个 RAG 项目",
    "level": "有 Python 基础",
    "daily_count": 5,
    "preferences": [
      {"dimension": "style", "value": "practical", "direction": "prefer", "scope": "global"}
    ]
  },
  "relevance": {"BV0000000001": null, "BV0000000002": null}
}
```

示例中的 BV 号需替换为真实候选，`relevance` 必须覆盖 `videos.json` 的全部视频。
`null` 表示尚未打分；存在 `null` 时该画像的 Precision 和 NDCG 均为待完成。
候选库最多 30 个，与应用的候选上限一致；建议先使用 20 个，包含适合与容易混淆的视频。

人工根据画像阅读视频与字幕，填入 0–3 分；时长超限或时长无法验证的候选需标 0。
全量检查后才将 `label_status` 改为 `human_reviewed`。草稿分数可以调试，但报告会标为待复核。
同一意图及其改写通过 `intent_id` 关联，不应跨 dev/test。固定好标签后再看模型输出。
更改金标准之后重新运行评估，旧报告不随外部文件变化。

现有两视频真实试验不足以代表前 5 名推荐质量，需要扩充候选后再形成基准。

### 3. 填写实际单价并运行真实模型

将 `evals/pricing.example.json` 复制到本地数据目录，按实际账单填写
`input_per_million`、`output_per_million` 和 `currency`；单位为每百万 token。
模型名需与本机配置一致，Embedding 输出单价填 0。模板中的 `null` 不代表免费。

```powershell
uv run video-agent eval --dataset data/recommendation-corpus-v1/cases.jsonl --pricing data/model-pricing.json --output reports/recommendation-v1
uv run video-agent eval --dataset data/recommendation-corpus-v1/cases.jsonl --split test --pricing data/model-pricing.json --output reports/recommendation-test-v1
```

真实模型需本机模型凭据；评估使用冻结语料，不联网采集视频。
缺少 `--pricing` 时仍记录 token 与耗时，费用显示未知。
比较版本时使用相同候选库、画像、人工标签、Embedding 模型和费用口径。

### 验证与历史诊断

```powershell
uv run pytest -q tests/test_recommendation_evaluation.py
uv run ruff check src tests
```

2026-10-08：全部 109 个测试通过，Ruff 通过，默认离线命令成功生成四项报告。
本次未运行收费的真实模型评估，不提供真实推荐质量或费用结论。

历史组件用例保存在 `evals/cases.jsonl`，显式传入该路径时运行原有诊断，
使用说明见 [历史组件评估](evaluation.md)。`eval-review` 的理由支持标签适用于历史
version 2 报告；当前 version 3 推荐评估在数据集内维护人工相关性分数。
