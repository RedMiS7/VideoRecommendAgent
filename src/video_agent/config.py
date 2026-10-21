import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    text_provider: str = "qwen"
    text_model: str = "qwen3.8-flash"
    embedding_provider: str = "qwen"
    embedding_model: str = "qwen3.7-text-embedding-flash"
    eval_provider: str = "qwen"
    eval_model: str = "qwen3.8-flash"
    model_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    model_api_key: str = ""
    text_base_url: str = ""
    text_api_key: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    eval_base_url: str = ""
    eval_api_key: str = ""
    search_api_key: str = ""
    search_mcp_url: str = "https://maas.qianwenaiapi.com/api/v1/mcps/EnhancedSearch/mcp"
    bili_sessdata: str = ""
    bili_jct: str = ""
    bili_buvid3: str = ""

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")).resolve(),
            text_provider=os.getenv("TEXT_PROVIDER", "qwen"),
            text_model=os.getenv("TEXT_MODEL", cls.text_model),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "qwen"),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            eval_provider=os.getenv("EVAL_PROVIDER", "qwen"),
            eval_model=os.getenv("EVAL_MODEL", os.getenv("TEXT_MODEL", cls.eval_model)),
            model_base_url=os.getenv("MODEL_BASE_URL", cls.model_base_url),
            model_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            text_base_url=os.getenv("TEXT_BASE_URL", ""),
            text_api_key=os.getenv("TEXT_API_KEY", ""),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            eval_base_url=os.getenv("EVAL_BASE_URL", ""),
            eval_api_key=os.getenv("EVAL_API_KEY", ""),
            search_api_key=os.getenv("SEARCH_API_KEY", ""),
            search_mcp_url=os.getenv("SEARCH_MCP_URL", "") or cls.search_mcp_url,
            bili_sessdata=os.getenv("BILI_SESSDATA", ""),
            bili_jct=os.getenv("BILI_JCT", ""),
            bili_buvid3=os.getenv("BILI_BUVID3", ""),
        )
