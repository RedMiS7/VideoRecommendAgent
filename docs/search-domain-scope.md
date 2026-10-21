# EnhancedSearch 域名限制验证

## What was added

搜索词统一加上 `site:bilibili.com` 前缀，长度限制保留为 500 字符。
结果过滤允许三个明确域名：`bilibili.com`、`www.bilibili.com`、`m.bilibili.com`。
只接受 HTTPS 的 `/video/BV…` 路径，移动端链接转为标准第 1 P 链接并按 BV 号去重。

## Why it exists

“B站 BV视频”只是普通关键词，会召回提到 B 站的文章。
只写“不要搜索其他网站”的自然语言要求，在本次探测中也没有限制域名。
此前的过滤器还排除了合法的移动端视频，导致有召回却没有候选。

## How to test or use it

```powershell
uv run pytest tests/test_sources.py -q
uv run python -c "from video_agent.config import Settings; from video_agent.sources import EnhancedSearch; print([v.url for v in EnhancedSearch(Settings.from_env()).search('RAG')])"
```

第二条命令读取本机凭据并消耗一次真实搜索调用，输出有效 BV 链接列表，不写入推荐库。
测试覆盖查询前缀、500 字符上限、移动端链接转换以及恶意域名和非视频页面过滤。

2026-10-02 对比结果：

| 查询 | 返回页面 | B 站域名页面 | 其他域名页面 |
| --- | ---: | ---: | ---: |
| `RAG B站 BV视频` | 8 | 3 | 5 |
| `RAG site:bilibili.com` | 10 | 10 | 0 |
| `site:www.bilibili.com RAG` | 10 | 10 | 0 |
| `仅搜索bilibili.com域名内的RAG视频，不搜索其他网站` | 6 | 0 | 6 |

适配器改用 `site:bilibili.com RAG` 后，真实调用返回 9 个有效 BV 候选。
探测明细保存在本机 `data/search-domain-probe/`，含查询、链接、标题和统计。

## Verification limits

`search_pro` 的工具 Schema 只接受 `query`，没有独立的域名白名单参数。
以上结果验证了本次查询效果，不能保证其他查询或服务内部网页抓取始终受此限制。
本地域名和视频路径校验继续强制执行，不从第三方摘要提取或猜测 BV 号。

推荐命令现允许重试当天已保存的成功空清单，详见 [每日推荐说明](recommendations.md)。
