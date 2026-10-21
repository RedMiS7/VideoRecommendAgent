import ast
from pathlib import Path

from video_agent.config import Settings
from video_agent.runtime import build_models


def test_business_modules_do_not_import_vendor_adapters_or_sdk():
    for name in ("memory", "rag", "recommender", "evaluation", "api"):
        tree = ast.parse(Path(f"src/video_agent/{name}.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any("qwen" in name or "dashscope" in name or name == "openai" for name in imports)


def test_model_capabilities_can_use_independent_configuration():
    settings = Settings(text_model="text", embedding_model="embed", eval_model="judge",
        text_base_url="https://text.test/v1", embedding_base_url="https://embed.test/v1",
        eval_base_url="https://judge.test/v1", text_api_key="text-test",
        embedding_api_key="embed-test", eval_api_key="judge-test")
    models = build_models(settings)
    assert models.language.base_url == "https://text.test/v1"
    assert models.embedding.base_url == "https://embed.test/v1"
    assert models.judge.base_url == "https://judge.test/v1"
    assert models.language.model == "text" and models.embedding.model == "embed"
    assert models.judge.model == "judge"
    for close in models.closers:
        close()
