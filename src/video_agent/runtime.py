from dataclasses import dataclass, field
from typing import Callable

from .config import Settings
from .memory import Memory
from .models.contracts import CallLog, EmbeddingModel, LanguageModel
from .models.qwen import QwenEmbeddingModel, QwenLanguageModel
from .rag import Rag
from .recommender import Recommender
from .sources import BilibiliSource, EnhancedSearch
from .store import Store


@dataclass
class Models:
    language: LanguageModel
    embedding: EmbeddingModel
    judge: LanguageModel
    calls: CallLog
    closers: list[Callable[[], None]] = field(default_factory=list)


def build_models(settings: Settings) -> Models:
    # Provider selection belongs only to application assembly.
    for name in (settings.text_provider, settings.embedding_provider, settings.eval_provider):
        if name != "qwen":
            raise ValueError(f"Provider '{name}' has no adapter installed; implement its model contract")
    calls = CallLog()
    language = QwenLanguageModel(settings.text_api_key or settings.model_api_key,
                                 settings.text_base_url or settings.model_base_url,
                                 settings.text_model, calls=calls)
    embedding = QwenEmbeddingModel(settings.embedding_api_key or settings.model_api_key,
                                   settings.embedding_base_url or settings.model_base_url,
                                   settings.embedding_model, calls=calls)
    judge = QwenLanguageModel(settings.eval_api_key or settings.model_api_key,
                              settings.eval_base_url or settings.model_base_url,
                              settings.eval_model, calls=calls)
    return Models(language, embedding, judge, calls, [language.close, embedding.close, judge.close])


class Runtime:
    def __init__(self, settings: Settings, offline: bool = False):
        self.settings, self.offline = settings, offline
        if offline:
            from .demo import DemoEmbedding, DemoLanguageModel, DemoSource

            self.models = Models(DemoLanguageModel(), DemoEmbedding(), DemoLanguageModel(), CallLog())
            self.search = self.source = DemoSource()
        else:
            self.models = build_models(settings)
            self.search, self.source = EnhancedSearch(settings), BilibiliSource(settings)
        self.store = Store(settings.data_dir / "agent.sqlite")
        self.memory = Memory(self.store, self.models.language)
        self.rag = Rag(self.store, self.models.language, self.models.embedding)
        self.recommender = Recommender(self.store, self.models.language, self.rag,
                                       self.memory, self.search, self.source, self.models.calls)

    def close(self):
        if not self.offline:
            for close in self.models.closers:
                close()
            self.search.close()
        self.source.close()
        self.store.close()
