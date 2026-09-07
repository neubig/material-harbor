from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import urllib.error
import urllib.request

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class BioReasonClassifier(BaseAgent):
    @staticmethod
    def name() -> str:
        return "bioreason-classifier"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    def _classify(self, case: dict[str, str]) -> tuple[str, dict]:
        prompt = (
            "Classify this genetic variant as benign or pathogenic. "
            "Return exactly one word: benign or pathogenic.\n"
            f"Context: {case['question']}\n"
            f"Reference sequence: {case['reference_sequence']}\n"
            f"Variant sequence: {case['variant_sequence']}"
        )
        model = (self.model_name or "deepseek-v4-flash").split("/", 1)[-1]
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 20,
                "reasoning_effort": "none",
            }
        ).encode()
        request = urllib.request.Request(
            self._get_env("LLM_BASE_URL").rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": "Bearer " + self._get_env("LLM_API_KEY"),
                "Content-Type": "application/json",
            },
        )
        error = None
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    data = json.load(response)
                content = data["choices"][0]["message"].get("content") or ""
                labels = re.findall(r"\b(?:benign|pathogenic)\b", content.lower())
                if not labels:
                    raise ValueError(f"invalid model response: {content!r}")
                return labels[-1], data.get("usage") or {}
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if attempt < 7:
                    time.sleep(min(30, 2**attempt))
        raise RuntimeError("model request failed") from error

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        result = await environment.exec(command="cat /app/data/case.json")
        if result.return_code != 0 or not result.stdout:
            raise RuntimeError(result.stderr or "failed to read case")
        label, usage = await asyncio.to_thread(self._classify, json.loads(result.stdout))
        encoded = base64.b64encode((label + "\n").encode()).decode()
        result = await environment.exec(
            command=f"printf '%s' '{encoded}' | base64 -d > /app/answer.txt"
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "failed to write answer")
        context.n_input_tokens = usage.get("prompt_tokens")
        context.n_output_tokens = usage.get("completion_tokens")
        context.n_cache_tokens = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        context.cost_usd = usage.get("cost")
        context.metadata = {"response": label}
