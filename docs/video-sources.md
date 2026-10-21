# 视频数据接入

## What was added

搜索已替换为千问平台的 EnhancedSearch MCP，通过官方 Python MCP SDK、Streamable HTTP
调用 `search_pro`，参数仅包含 `query`。每次搜索独立创建并关闭连接，整体超时 45 秒。
查询统一添加 `site:bilibili.com`。解析实际返回的 `status`、`pages[].url/title/snippet`，
只接收 `bilibili.com`、`www.bilibili.com`、`m.bilibili.com` 的 HTTPS BV 视频链接，
去重并统一为第 1 P；不把搜索摘要中的链接或 BV 字样当作候选。
`bilibili-api-python==17.4.2` 获取视频元数据和第 1 P 字幕。
存储播放、点赞、收藏、评论数、采集时间、字幕状态与正文，元数据缓存 24 小时。
字幕下载只接受 B 站和其 CDN 域名。数据接入不会自动点赞、收藏或上报观看行为。

## Why it exists

搜索摘要用于发现候选，统计数据用于辅助初筛，完整字幕为推荐内容结论提供证据。
字幕列表可能为空，也可能因登录或网络条件获取失败，必须分别记录。
字幕调用前检查 `SESSDATA`，缺失时返回明确配置提示并保留 `error` 状态。
成功获取字幕缓存 24 小时；空列表和失败结果下次调用重新检查，方便补充凭据后恢复，
也避免复用旧版本误判的“无字幕”。这会增加真正无字幕视频的接口调用次数。
读取完整 `get_player_info()` 响应，检查顶层 `need_login_subtitle` 和 `login_mid`，
再解析 `subtitle.subtitles`。未登录或响应结构不完整不再归类为“无字幕”。
轨道选择同时识别 `zh…` 与 `ai-zh…`，优先下载中文字幕，包括 AI 中文字幕。
搜索适配保留 `search(query)` 接口，上层推荐逻辑不依赖 MCP 或千问模型。
鉴权、传输、工具调用和格式错误转换为 `SourceError`，保持现有缓存候选降级行为。

## How to test or use it

```powershell
uv sync --locked
uv run pytest tests/test_sources.py -q
```

本机 `.env` 配置：

```dotenv
DASHSCOPE_API_KEY=你的千问平台APIKey
SEARCH_MCP_URL=https://maas.qianwenaiapi.com/api/v1/mcps/EnhancedSearch/mcp
SEARCH_API_KEY=
```

`SEARCH_API_KEY` 留空时使用 `DASHSCOPE_API_KEY`；需要独立搜索凭据时填写它。
更换文本、向量或评估模型不影响搜索服务。旧 `BOCHA_*` 配置已不再使用。
已有 `.env` 请手动补充需要的字段，避免用模板覆盖凭据。

只验证搜索（会消耗真实调用额度，不写入推荐库）：

```powershell
uv run python -c "from video_agent.config import Settings; from video_agent.sources import EnhancedSearch; print([v.url for v in EnhancedSearch(Settings.from_env()).search('RAG 学习视频')])"
```

输出为有效视频链接列表；`[]` 表示没有符合过滤条件的候选，不代表鉴权失败。
查询形如 `site:bilibili.com RAG`，总长度限制为 500 字符。
最新真实对比中，域名查询返回 10 个 B 站页面，适配器识别出 9 个 BV 视频。
移动端视频链接统一转为 `www.bilibili.com/video/BV…/?p=1`。
列表、专栏、付费课程和第三方页面均被过滤。
这是经样本验证的查询限制；MCP 未提供独立域名参数，不能保证服务内部只访问 B 站。
详细对比见 [域名限制验证](search-domain-scope.md)。

综合检查模型、搜索和视频数据：

```powershell
uv run video-agent live-check --bvid BV1uv411q7Mv
```

最多同时指定 3 个 `--bvid`。命令检查真实服务，失败返回退出码 1，并写入 `data/live-check.json`。
不会将探测视频写入真实推荐候选库。

当前安装的 SDK 将字幕依赖的播放器接口标记为需要登录，`BILI_SESSDATA` 为必填项。
`CredentialNoSessdataException` 表示凭据未提供或为空；不是“该视频没有字幕”。
排查步骤：

1. 在浏览器登录 B 站，用开发者工具查看 B 站域名下的 Cookies。
2. 在本机 `.env` 将 `SESSDATA` 的值填入 `BILI_SESSDATA`；
   可将 `bili_jct`、`buvid3` 分别填入 `BILI_JCT`、`BILI_BUVID3`。
3. 重启已运行的 `serve` 进程，让新建的 SDK Credential 读取新配置。
4. 运行 `uv run video-agent live-check --bvid BV1AodfByE7D`。

期望 `subtitle_status=available` 且 `segments>0`；若请求成功但没有字幕则为 `absent`。
只有接口确认登录且返回合法的空字幕列表时才判为 `absent`。
已配置的 Cookie 仍可能失效，以真实探测结果为准。
这些是登录凭据，不要在聊天或 Git 中提供。

来源资料：[EnhancedSearch](https://www.qianwenai.com/hub/mcp/EnhancedSearch)、
[官方 MCP 外部调用](https://platform.qianwenai.com/docs/agent-infra/mcp/external-invocation)、
[Python 库发布包](https://pypi.org/project/bilibili-api-python/17.4.2/)、
[上游仓库现状](https://github.com/Nemo2011/bilibili-api)。

2026-10-02 已使用本机凭据完成 EnhancedSearch 握手、工具列表读取及真实搜索，
确认工具参数和返回格式；搜索能返回网页，但适配器查询“RAG 学习视频”过滤后为 0 个 BV 候选。
连接验证已通过，有效视频召回仍需按目标主题验证。
此前公开视频元数据请求返回 SDK `NetworkException`，此次仅替换搜索，未重新验证视频与字幕接入。

2026-10-02 排查最近一次探测：`BV1AodfByE7D` 的元数据成功，字幕因本机缺少
`BILI_SESSDATA` 失败；三个 B 站 Cookie 配置均为空。已修复提示及错误缓存重试行为，
字幕成功获取仍需在本机补充有效登录 Cookie 后验证。

### AI 字幕误判排查（2026-10-02）

真实复测 `BV1gpc4zjEtj` 和 `BV1zBv4exEo6`：配置中存在 SESSDATA，
但播放器响应的 `login_mid=0`、顶层 `need_login_subtitle=true`，字幕列表为空。
SDK 的 `get_subtitle()` 仅返回字幕子对象，旧代码因此丢失登录提示并误记为 `absent`。
修复后前者从缓存的 `absent` 重新检查，正确返回 `error` 和更新登录 Cookie 的提示。
浏览器已登录不代表本机 `.env` 中的 Cookie 仍有效或填写正确。
请更新 `BILI_SESSDATA` 为浏览器当前 SESSDATA 的值，重启服务后运行：

```powershell
uv run video-agent live-check --bvid BV1gpc4zjEtj
```

当前凭据下尚未验证真实字幕正文下载成功；离线测试验证了 AI 中文字幕选择、正文解析、
登录失败识别、旧空列表重试和已有字幕缓存。

### 推荐清单与实时字幕状态不同（2026-10-05）

**What was added:** 排查并记录推荐流程与 `live-check` 的差异；本次未修改下载代码。
两者调用同一个 `BilibiliSource.subtitles()`，`live-check` 的 `segments` 为实际下载并解析的
正文段数，不只是字幕轨道数。

**Why it exists:** 当天非空推荐清单会直接复用，清单中的字幕状态是生成时的快照。
2026-10-05 的清单保存了登录失败状态；当前凭据复测清单中的 `BV1pSQEBpEAL`，
播放器确认登录，返回 1 条轨道，实际成功下载 114 段字幕。此前 `data/live-check.json`
也记录 `BV1AodfByE7D` 成功下载 618 段。因此旧清单的失败状态不代表当前下载接口仍失败。

**How to test or use it:** 对推荐清单中相同 BV 号运行
`uv run video-agent live-check --bvid BV1pSQEBpEAL`，核对 `subtitle_status=available`、
`segments>0`。该探测不会更新推荐数据库或重写当天清单；新一天生成推荐才会再次执行
字幕获取。单独元数据探测仍可能遇到 HTTP 412，需与字幕下载失败区分。
