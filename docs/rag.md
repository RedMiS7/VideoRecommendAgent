# 字幕 RAG

## What was added

第 1 P 字幕采用 Embedding 语义切分。每约 4 秒选择一个字幕条目边界，
比较左右各约 20 秒的字幕上下文，使用余弦距离和视频内第 85 百分位识别明显变化。
首尾不足 10 秒上下文的区域不评分；没有可比较的边界时，仅按长度划分。

每个输出块严格为 **200–600 字符**，包含空格、标点和换行，不是 token 数或时长。
动态规划在可完整覆盖字幕的划分中优先保留显著语义切点，再减少额外块数量，
并让长度所需切点优先靠近较大的语义距离。短尾块通过重新分配边界解决。
重叠最多约 100 字符，仅加入完整字幕条目，且加入后仍不超过 600 字符。
超长字幕条目拆分时保留原字幕时间范围，不声称精确定位到拆分后的词语。
空字幕返回空列表；全文不足 200 字符或保持普通条目完整时无法合法划分，明确报错，
不填充、不丢弃原文、不生成长度违规的块。

先按字幕批次生成带引用的摘要，再合并长视频摘要。SQLite 保存原文、摘要、引用和向量。
检索先根据视频摘要选候选，再在相应视频中检索最多 5 个片段。
摘要和推荐引用都会检查所属视频和已存在的片段。

## Why it exists

用真实字幕支持学习内容判断，减少标题党和无依据的推荐理由。
视频级摘要避免仅因某一句话匹配就选中，片段级检索为理由提供可核查证据。
AI 字幕常无标点且按短语断开。比较切点两侧上下文可识别语义变化，
长度可行性检查避免选择切点后留下过短尾块。
字幕不能用于可靠判断声音、画面质量；检索未找到也不等于整段视频没有该内容。

## How to test or use it

```powershell
uv run pytest tests/test_rag.py -q
uv run pytest tests/test_chunking.py -q
uv run video-agent recommend
uv run video-agent reindex
```

推荐结果的 `evidence` 包含字幕原文、起止秒数和片段 ID。
正文、CID、分块策略版本及参数、Embedding 空间未变时复用索引。
旧固定分块索引在该视频下一次 `index_video()` 时重新生成；Embedding 空间改变
仍要求显式重新索引。
重新索引只重建派生索引，不删除反馈或用户偏好。

无字幕视频只在有字幕候选不足时补位，标注“依据标题和摘要推荐”，不产生字幕证据。

单独验证真实分块，不运行推荐或改写推荐库：

```powershell
uv run python scripts/validate_subtitle_chunks.py --input-dir data/semantic-chunk-validation --output reports/chunk-validation.json
```

输入目录保存以 BV 号命名的 `Video` JSON（包括已下载的 `segments`），
对应 `tests/fixtures/chunk-boundaries.json` 的预先标注样本。
脚本只调用配置的 Embedding，输出切点、距离、长度、标注对照和调用成本；
命中率不足 80% 或出现违规长度时返回退出码 1。
真实验证结果与限制见 [语义切分验证](embedding-chunking-validation.md)。
现有演示及评估中的合成字幕补充了同主题文字以满足新下限；它们仍是测试数据。
