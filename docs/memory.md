# 用户偏好记忆

## What was added

支持结构化配置和自然语言初始化草稿。保存显式偏好、反馈原文、维度提取、点击与推断偏好。
维度包括主题、深度、讲解方式、时长和节奏；适用范围分为视频、主题与全局。

同一维度、值与范围，最近 30 天内最近 3 个不同视频的反馈一致后才更新长期偏好。
重复反馈同一个视频只算一个来源。单次证据只影响对应视频；模糊反馈不推广。
有冲突时保留已生效偏好并继续观察。用户直接设置的偏好优先。

反馈提取在校验前将 `video` 范围绑定到当前视频 BV 号，修复模型返回
`scope_value=null`、空值或其他视频编号时的校验失败。
`topic` 仍须包含主题名，`global` 的 `scope_value` 仍须为 `null`。

## Why it exists

例如“主题有用，但太基础”不能简单变成“不喜欢这个主题”。
累计多个视频的证据可以降低个别视频对长期偏好的影响，原文保留便于复核。
点击是弱信号，不代表观看完成，未点击也不代表不喜欢。
当前反馈视频由请求和入库记录确定，无需模型重新识别。
先绑定视频再校验可避免丢失有效的视频反馈，同时保持主题和全局偏好的范围约束。

## How to test or use it

```powershell
uv run video-agent init --file docs/examples/profile.json
uv run video-agent init --text "我懂 Python 基础，希望学习 RAG 的工程案例"
# 草稿确认后，用 init --file 激活。
uv run video-agent feedback BVxxxxxxxxxx "主题有用，但这个太基础"
uv run video-agent memory
uv run video-agent profile --file docs/examples/profile.json
uv run pytest tests/test_memory.py -q
```

`BVxxxxxxxxxx` 替换为已入库视频。`profile --file` 替换整份配置并将其中偏好视为显式设置。
首次初始化无需模型调用的路径是 `init --file`。
模型提取失败仍保存反馈原文，`extracted=null` 表示尚未成功提取；重新提交同一视频不会多算一个来源。
无法通过校验的模型结果转换为 `ModelError(kind="invalid_output")`；
API 返回 503，CLI 输出简短错误，不将无效观察写入偏好。
回归测试覆盖视频范围缺失/空值/错误 BV 号，以及非法主题和全局范围的拒绝。

2026-10-02 使用 `qwen3.8-flash` 在独立测试数据库中验证三种反馈：
“这个视频太基础了”绑定当前 BV 号；“以后不要太基础”提取全局范围；
“以后 RAG 主题不要太基础，其他主题不限制”提取主题范围 `RAG`。
三次提取均通过校验；这项检查不等同于完整 AI eval 质量验收。

明确说“看完了”或“不再推荐这个视频”会将该视频排除在后续推荐之外。
