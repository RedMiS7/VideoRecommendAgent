import json
import time

import httpx
import numpy as np
from jsonschema import ValidationError, validate

from .contracts import CallLog, Message, ModelError


class _QwenHTTP:
    def __init__(self, api_key: str, base_url: str, model: str,
                 client: httpx.Client | None = None, calls: CallLog | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=60)
        self.calls = calls or CallLog()

    def _request(self, path: str, payload: dict) -> dict:
        if not self.api_key:
            raise ModelError("authentication", "Model API key is not configured")
        start = time.monotonic()
        status = "failed"
        usage = None
        try:
            response = self.client.post(self.base_url + path, json=payload,
                                        headers={"Authorization": "Bearer " + self.api_key})
            if response.status_code in (401, 403):
                raise ModelError("authentication", "Model authentication failed")
            if response.status_code == 429:
                raise ModelError("rate_limit", "Model request was rate limited")
            if response.is_error:
                raise ModelError("unavailable", f"Model HTTP error {response.status_code}")
            result = response.json()
            usage = result.get("usage")
            status = "success"
            return result
        except httpx.RequestError:
            raise ModelError("unavailable", "Model network request failed") from None
        except (ValueError, AttributeError):
            raise ModelError("invalid_output", "Model response was not a JSON object") from None
        finally:
            self.calls.events.append({"provider": "qwen", "model": self.model,
                                      "operation": path, "status": status, "usage": usage,
                                      "seconds": round(time.monotonic() - start, 3)})

    def close(self):
        self.client.close()


class QwenLanguageModel(_QwenHTTP):
    def generate_json(self, messages: list[Message], schema: dict) -> dict:
        wire = [{"role": m.role, "content": m.content} for m in messages]
        wire.insert(0, {"role": "system", "content":
                       "Return only a JSON object matching this JSON Schema: " +
                       json.dumps(schema, ensure_ascii=False)})
        for attempt in range(2):
            response = self._request("/chat/completions", {
                "model": self.model, "messages": wire, "temperature": 0,
                "response_format": {"type": "json_object"}, "enable_thinking": False,
            })
            try:
                result = json.loads(response["choices"][0]["message"]["content"])
                validate(result, schema)
                return result
            except (KeyError, IndexError, TypeError, ValueError, ValidationError):
                if attempt == 1:
                    raise ModelError("invalid_output", "Model JSON failed schema validation") from None
                wire.append({"role": "user", "content":
                             "Your output was invalid. Return valid JSON matching the schema."})
        raise AssertionError("unreachable")


class QwenEmbeddingModel(_QwenHTTP):
    @property
    def space_id(self) -> str:
        return f"qwen:{self.base_url}:{self.model}:1024"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            result = self._request("/embeddings", {"model": self.model, "input": batch,
                                                  "dimensions": 1024})
            try:
                data = sorted(result["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ValueError
                current = [item["embedding"] for item in data]
                matrix = np.asarray(current, dtype=float)
                if matrix.ndim != 2 or matrix.shape[1] != 1024 or not np.isfinite(matrix).all():
                    raise ValueError
                if vectors and len(vectors[0]) != matrix.shape[1]:
                    raise ValueError
                vectors.extend(current)
            except (KeyError, ValueError, TypeError):
                raise ModelError("invalid_output", "Invalid embedding response") from None
        return vectors
