# 历史组件 AI Eval 与人工复核

2026-10-08 起，默认评估已收敛为用户画像驱动的四项推荐指标。
当前使用说明见 [推荐四项 AI Eval](recommendation-evaluation.md)。
本文保留原有反馈、问答检索、排序对照及理由复核的诊断步骤。

## What was added

保留原有 30 个合成回归用例，并支持固定的真实 B 站视频语料、dev/test 划分、
无匹配排序场景和逐条人工理由复核。模型通过现有统一接口运行，不新增依赖。

三组对照为标题无记忆、字幕 RAG 无记忆、字幕 RAG 有记忆。
无记忆组在检索之前移除偏好；各组共用候选、基础目标、时长限制、0.5 分门槛和排序规则。
两个 RAG 组均保存理由、实际引用与模型评判。空 gold_bvids 表示预期没有合适视频。

JSON/Markdown 报告包括运行 ID、数据哈希、实际系统消息及 Schema 指纹、模型、Embedding 空间、
CHUNK_POLICY、检索参数、各阶段调用用量与耗时、失败记录、指标分子/分母及分组结果。
报告中的 prompt_hashes 是调用契约中系统消息与 Schema 的哈希，不保存用户提示词。
默认生成与评判可以使用同一个模型；不同实例不代表独立的评判模型。

## Why it exists

把检索、排序和理由支持拆开评估，用公平对照定位字幕和记忆的作用。
人工标签检查模型评判的可靠性；请求失败保留在指标分母中，避免遗漏失败样本后高估效果。
评估范围为固定候选库中的检索和排序，不包含在线搜索召回、每日调度或整体 B 站推荐质量。

## How to test or use it

```powershell
uv run video-agent eval --dataset evals/cases.jsonl --offline
# 真实模型运行原有合成数据（仍不是真实视频基准）：
uv run video-agent eval --dataset evals/cases.jsonl
```

报告保存在 `reports/`，评估数据库保存在 `data/eval/` 下的独立运行目录，不修改真实用户记忆。
离线模式使用确定性测试替身，仅验证流程、结果结构和指标计算，不评估 AI 能力。

### 1. 采集并冻结真实语料

准备每行一个 BV 号的文本文件，再执行：

```powershell
uv run video-agent eval-collect --bvid-file data/eval-bvids-20261005.txt --output data/eval-real-v2
```

只调用元数据与字幕接口，不调用模型、不修改推荐数据库。
输出 videos.json 和 manifest.json，记录成功、无字幕、失败以及正文哈希。
逐个采集，视频之间间隔一秒；HTTP 412 或登录错误时停止并保留部分结果。
发生错误或未采完时 complete=false，命令返回退出码 1，部分快照仍会保存。
已有快照不能覆盖，重试使用新输出目录。部分采集成功不代表 20 视频目标已达成。

### 2. 编写与人工确认用例

在同目录放置 cases.jsonl，沿用原有反馈、检索和排序结构。每条新增字段：

- split：dev 或 test；旧用例默认 unspecified。
- intent_id：同一个问题及其改写共用 ID，不能跨 dev/test。
- label_status：起草时 needs_human_review，人确认正确答案后改为 human_reviewed。

检索用例提供 query、gold_bvid、gold_start、gold_end；时间区间需在原字幕范围内。
排序用例提供 profile 和 gold_bvids（允许空数组）。先固定标签，再看模型结果。
manifest.json 的 source 使用 bilibili，videos_sha256 与冻结文件一致。
source 未注明的自定义数据不会被自动认定为真实语料；原有 evals/manifest.json 明确标为 synthetic。

```powershell
uv run video-agent eval --dataset data/eval-pilot-20261005/cases.jsonl --output reports/eval-pilot-20261005
# 开发调试可只运行 dev；最终留出集运行 test：
uv run video-agent eval --dataset data/eval-pilot-20261005/cases.jsonl --split test --output reports/eval-test
```

单个索引、检索、排序或评判错误会记录为失败，仍输出其余用例的结果。
索引失败时 RAG 组标为错误，避免将缺失正文当作正常 RAG 结果。

### 3. 导出和导入人工理由标签

把下面的 eval-xxx.json 替换为上一条命令返回的真实报告路径：

```powershell
uv run video-agent eval-review --report reports/eval-pilot-20261005/eval-xxx.json
```

生成 .review.jsonl，包含运行 ID、用例 ID、对照组、画像、理由和引用原文。
不展示模型判定，便于先独立检查。逐条填写 supported 为 true/false，note 填写判断依据。
运行 ID 和 output_hash 保持不变，它们防止把其他运行或已变化的输出标签导入。
导出不会覆盖已有人工模板；重新导出使用 --output 指定新文件。

```powershell
uv run video-agent eval-review --report reports/eval-pilot-20261005/eval-xxx.json --labels reports/eval-pilot-20261005/eval-xxx.review.jsonl
```

生成新的 .reviewed.json/.md，保留原始报告。支持导入部分已填写条目，
但不能含 supported=null、重复用例、未知用例或来自其他运行的标签。
部分复核显示覆盖率和待复核数量，不能通过质量验收；理由标签不会替代金标准的人审。

### 指标与验收口径

| 指标 | 含义 |
| --- | --- |
| video_hit_at_1 / video_hit_at_5 | 单个正确视频是否在前 1 / 5 名 |
| video_mrr_at_5 | 正确视频在前 5 名内的倒数排名平均值，未命中为 0 |
| chunk_hit_at_5 | 全局按相似度选取的前 5 个片段中，是否有片段完整覆盖正确时间区间 |
| 各组 top1_accuracy | Top-1 是否属于可接受集合；预期无匹配时检查返回空结果 |
| human_support_rate | rag_memory 已人工复核理由的支持率，同时给出样本数 |
| 各 RAG 组 human_review_coverage | 需复核的生成理由中已经复核的比例 |
| judge_human_agreement | 有双方标签时的判定一致率 |
| judge_false_support_rate | 模型认为支持的已复核样本中，人认为不支持的比例 |
| execution_failure_rate | 有执行错误的场景占比，与内容判断错误分开 |

原来误称 Recall@5 的单目标指标改名为 Hit@5，报告 JSON 不再提供旧名称。
空输出不会进入理由支持率分母；它单独体现在空结果率和排序准确率中。
MRR@5 无法反映第 5 名以后的位置。时间区间完全覆盖容易偏向长片段，需结合原文审阅。

quality_verified 只代表当前固定语料的 test 集通过项目目标，需要：

- 真实模型、明确标注为 B 站来源的真实快照。
- test 集同时包含检索和排序用例，全部运行用例的金标准已由人确认。
- 两个 RAG 组所有需复核的理由已由人标注，且没有索引或场景执行错误。
- test 集片段 Hit@5 ≥80%、rag_memory 人工支持率 ≥90%；若含反馈用例，其准确率 ≥90%。

离线运行、未分留出集、缺少标签、指标未达标时均为 false，并列出原因。
两视频试验集不具有 20 视频/60 用例基准的代表性，不能据此宣传总体质量。

### 本次真实接入状态（2026-10-05）

已尝试采集 20 个 BV 号：首个视频获得 524 条字幕，第二个元数据请求返回 HTTP 412，
因此停止采集。实际新快照只有 1 个视频，保存于 data/eval-real-20261005。
结合此前已下载的另一部视频，准备了 data/eval-pilot-20261005：2 个视频、12 个用例，
其中 8 个 dev、4 个 test。金标准由代理起草，尚待人工复核。
reports/eval-pilot-20261005/gold-review.md 提供问题、区间附近原文和复核步骤。
20 视频/60 用例基准仍待接口恢复后采集并补充，不能用这份试验集替代其验收。

本次真实模型结果与失败分析见 [闭环实施与验证](evaluation-validation.md)。
