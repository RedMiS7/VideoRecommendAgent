# 模型适配层

## What was added

供应商无关的 `LanguageModel.generate_json(messages, schema)` 与
`EmbeddingModel.embed(texts)` 接口，以及千问文本、Embedding 适配器。
文本生成、向量化和 eval 评判分别注入；业务模块不导入千问实现。
默认文本和 eval 模型已切换为 `qwen3.8-flash`，向量模型为
`qwen3.7-text-embedding-flash`，共享地址采用千问 AI 平台的 OpenAI 兼容接口。

适配器负责鉴权、HTTP 请求、JSON Schema 校验、一次格式修复及错误转换。
调用日志保存模型、操作、耗时和供应商返回的用量，不保存凭据与提示词。

## Why it exists

使推荐、反馈提取和 RAG 与供应商协议解耦。当前只实现千问；更换模型名称与新增供应商是不同操作。
新增供应商需实现统一接口，并在 `build_models` 增加装配分支；上层业务无需修改。

## How to test or use it

```powershell
uv run pytest tests/test_models.py -q
```

默认配置（已有 `.env` 需同步更新这些非凭据字段）：

```dotenv
TEXT_MODEL=qwen3.8-flash
EMBEDDING_MODEL=qwen3.7-text-embedding-flash
EVAL_MODEL=qwen3.8-flash
MODEL_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
```

共享地址与凭据由 `MODEL_BASE_URL`、`DASHSCOPE_API_KEY` 设置。
各能力的 `TEXT_*`、`EMBEDDING_*`、`EVAL_*` 地址/凭据覆盖项为空时使用共享配置。

新适配器必须通过契约测试：结构校验、输入/向量顺序、错误分类和索引空间身份。
Embedding 默认 1024 维；空间身份包含供应商、地址、模型和维度。
更换 Embedding 后执行 `uv run video-agent reindex`，不能混用旧向量。
文本继续使用 JSON 对象输出并显式关闭思考，返回值由本地 JSON Schema 校验。
向量继续采用 1024 维、每批最多 10 条文本，处于新模型的 20 条批次上限内。

真实接口检查：`uv run video-agent live-check --bvid BV1uv411q7Mv`。
检查记录保存在 `data/live-check.json`；该命令也检查搜索和 B 站，需分别查看各服务结果。

依据：[平台入门](https://platform.qianwenai.com/docs/developer-guides/getting-started/introduction)、
[向量模型规格](https://platform.qianwenai.com/docs/developer-guides/embeddings/embedding)、
[Flash 思考模式](https://platform.qianwenai.com/docs/developer-guides/getting-started/latest-model)。

2026-10-02 真实适配器探测通过：`qwen3.8-flash` 返回符合 Schema 的 JSON，
`qwen3.7-text-embedding-flash` 为两条文本分别返回 1024 维向量。
这是接口兼容性验证，推荐质量仍需单独运行 AI eval 并复核证据。

已支持的统一错误：`authentication`、`rate_limit`、`unavailable`、`invalid_output`。
不支持的供应商配置会在装配时明确报错；不会静默退回千问。
