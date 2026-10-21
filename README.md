# Video Agent MVP

本地单用户 B 站学习视频推荐 Agent：每日推荐、字幕 RAG、反馈记忆及 AI eval。

## 快速开始

在项目根目录使用 PowerShell：

```powershell
uv sync --locked
Copy-Item .env.example .env
```

在本机 `.env` 填写 `DASHSCOPE_API_KEY`，用于默认模型和 EnhancedSearch MCP 搜索。
搜索可通过 `SEARCH_API_KEY` 独立配置；B 站字幕访问可能需要登录凭据。
不要将凭据提交到 Git。文本生成和 eval 默认使用 `qwen3.8-flash`，
向量化使用 `qwen3.7-text-embedding-flash`；业务通过统一模型接口调用。

```powershell
uv run video-agent init --file docs/examples/profile.json
uv run video-agent live-check --bvid BV1uv411q7Mv
uv run video-agent recommend
uv run video-agent serve
```

API 文档：<http://127.0.0.1:8000/docs>。推荐包含本地点击地址；记录点击时 API 需保持运行。

自然语言初始化先生成待确认草稿：

```powershell
uv run video-agent init --text "我懂 Python 基础，想学习 RAG 工程实践，每个视频不超过 30 分钟"
# 检查 data/profile-draft.json 后再执行：
uv run video-agent init --file data/profile-draft.json
```

## 无凭据演示与测试

```powershell
uv run video-agent demo --days 8
uv run video-agent eval --offline
uv run pytest -q
uv run ruff check src tests
```

`demo` 使用合成视频和确定性模型替身，数据与真实用户数据隔离。
离线 eval 分数只验证评估流程；不代表真实模型或 B 站推荐质量。

## 每日任务

初始化偏好并验证真实服务后：

```powershell
.\scripts\Install-DailyTask.ps1 -WhatIf
.\scripts\Install-DailyTask.ps1
```

默认每天北京时间 08:00；脚本读取偏好的 `daily_time`，要求 Windows 使用中国标准时间。
机器错过执行时间后补跑当天；同日成功结果复用。日志保存在数据目录的 `daily-task.log`。
修改时间后需要手动更新任务计划中的触发器。安装脚本不会覆盖同名任务。

## 功能文档

| 功能 | 文档 |
| --- | --- |
| 模型适配与更换 | [模型层](docs/models.md) |
| 搜索、元数据与字幕 | [视频接入](docs/video-sources.md) |
| 切片、向量与证据 | [字幕 RAG](docs/rag.md) |
| 初始化与反馈沉淀 | [用户记忆](docs/memory.md) |
| 推荐、API 与 CLI | [每日推荐](docs/recommendations.md) |
| 定时任务 | [Windows 调度](docs/scheduling.md) |
| 推荐四项指标与人工标注 | [AI eval](docs/recommendation-evaluation.md) |
| 单画像自动采集 30 个字幕视频 | [视频库采集脚本](docs/video-corpus-crawler.md) |

## 当前验证边界

真实模型和搜索请求需要 API 凭据。`bilibili-api-python` 上游已经归档，当前网络和登录条件可能影响接口访问；
用 `live-check` 检查实际可用性，结果保存到 `data/live-check.json`。
统计数据缺失时保留未知；字幕缺失与获取失败分别标注，不伪造内容结论。

默认 AI eval 使用用户画像运行推荐流程，仅报告 Precision@5、NDCG@5、Diversity@5
及推荐成本与延迟。默认示例包含 6 个合成视频和 4 个画像，相关性评分尚待人工复核。
离线得分与耗时仅用于验证流程，不代表真实模型质量或服务性能。

使用步骤、0–3 分人工标注规则与单价文件见 [AI eval](docs/recommendation-evaluation.md)。
原有 30 个组件场景继续用于回归；历史两视频试验见
[闭环验证](docs/evaluation-validation.md)，其指标不属于当前四项主评估。
