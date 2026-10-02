"""Harbor agent adapter that keeps image content alive for DeepSeek vision models.

Why this exists
---------------
Harbor 0.22.0's ``openhands-sdk`` adapter builds the SDK ``LLM`` from a fixed set
of environment variables (``LLM_MODEL``/``LLM_API_KEY``/``LLM_BASE_URL``/
``LLM_TEMPERATURE``/``MAX_ITERATIONS``). It exposes no way to pass
``force_string_serializer`` or ``capability_overrides``, and its bundled
``openhands_sdk_runner.py`` constructs ``LLM(model, api_key, base_url,
temperature)`` only.

That matters because the SDK resolves two independent model properties:

* ``force_string_serializer`` -- a case-insensitive *substring* match against
  ``FORCE_STRING_SERIALIZER_MODELS``, which contains the bare token ``deepseek``.
  Every DeepSeek model therefore takes the string serializer branch.
* ``supports_vision`` -- resolved from LiteLLM metadata plus
  ``capability_overrides``.

The string serializer emits ``TextContent`` only, so a message carrying
``ImageContent`` was silently reduced to text and the pixels never reached the
model. Setting ``supports_vision`` alone does not help: ``vision_enabled`` only
selects the list serializer when ``force_string_serializer`` is false.

The upstream fix (OpenHands/software-agent-sdk#5460) keeps the list format for
messages that actually carry an image whenever vision is active, even when
string serialization is forced. It is required together with a
``capability_overrides={'supports_vision': True}`` override, because LiteLLM
reports ``supports_vision=False`` for the DeepSeek proxy model ids.

This adapter supplies both without editing Harbor's shared runner. It replaces
the runner Harbor uploads with a task-local one that applies the overrides, and
it reads configuration from a JSON file because ``OpenHandsSDK.run`` forwards
only a fixed set of environment keys.

Prebaked images
---------------
Harbor's ``install`` skips its pip step when ``/opt/openhands-sdk-venv`` already
imports ``openhands.sdk``. Harbor's ``version`` kwarg cannot request the fix,
because it renders as ``openhands-sdk==<value>`` -- an exact PyPI release, never
a commit. A prebaked image carrying the pinned commit is therefore the supported
path; see ``harbor_vision/README.md``.

Scope
-----
This is a *transport* adapter. It changes how messages are serialized, not how
tasks are scored. Accuracy numbers produced with it are only valid if the
pixels are independently shown to reach the model; see the controlled probe
procedure in ``harbor_vision/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import override

from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.environments.base import BaseEnvironment


class VisionOpenHandsSDK(OpenHandsSDK):
    """OpenHands SDK agent with vision capability overrides applied.

    Extra ``--ak`` kwargs:

    ``vision_supports_vision`` (bool, default True)
        Value for ``capability_overrides['supports_vision']``. Required because
        LiteLLM reports ``supports_vision=False`` for the DeepSeek proxy ids.
    ``vision_force_string_serializer`` (str, default "" = leave unset)
        ``"true"``/``"false"`` to pin ``force_string_serializer`` explicitly.
        Left unset, the SDK's substring default still applies; the upstream fix
        then preserves images because vision is active.
    """

    def __init__(
        self,
        vision_supports_vision: bool = True,
        vision_force_string_serializer: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._vision_config = {
            "supports_vision": bool(vision_supports_vision),
        }
        if vision_force_string_serializer:
            self._vision_config["force_string_serializer"] = (
                str(vision_force_string_serializer).lower() == "true"
            )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Install the SDK, then overlay the task-local vision runner.

        The base adapter uploads its own ``openhands_sdk_runner.py`` at the end
        of ``install``; we call it first and then replace that file so the
        prebaked-venv path and the custom-runner path cannot drift apart.
        """
        await super().install(environment)

        runner = Path(__file__).parent / "vision_openhands_sdk_runner.py"
        local_copy = self.logs_dir / "run_agent.py"
        local_copy.write_text(runner.read_text())

        config_copy = self.logs_dir / "vision_config.json"
        config_copy.write_text(json.dumps(self._vision_config, indent=2) + "\n")

        await environment.upload_file(
            source_path=local_copy,
            target_path="/installed-agent/run_agent.py",
        )
        await environment.upload_file(
            source_path=config_copy,
            target_path="/installed-agent/vision_config.json",
        )
        await environment.exec(
            command="chmod +x /installed-agent/run_agent.py",
            user="root",
        )
