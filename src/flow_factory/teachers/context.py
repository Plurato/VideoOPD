# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/teachers/context.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch


MEDIA_CONTEXT_KEYS = {
    "first_frame",
    "last_frame",
    "depth",
    "pose",
    "optical_flow",
    "object_tracks",
}


class OPDContextBuilder:
    """Build optional teacher-only prompt context from sample metadata."""

    def __init__(
        self,
        context_keys: List[str],
        prompt_template: str,
        context_dropout: float = 0.0,
    ):
        self.context_keys = context_keys
        self.prompt_template = prompt_template
        self.context_dropout = context_dropout
        if self.context_keys and "{context}" not in self.prompt_template:
            raise ValueError(
                "`teacher.prompt_template` must contain `{context}` when "
                "`train.teacher_context_keys` is non-empty. Set "
                "`teacher_context_keys: []` for same-prompt OPD."
            )

    @staticmethod
    def normalize_context(context: Any) -> Dict[str, Any]:
        """Normalize a serialized or dictionary OPD context."""
        if context is None:
            return {}
        if isinstance(context, str):
            if not context:
                return {}
            try:
                loaded = json.loads(context)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "`opd_context` must be a JSON object string when serialized; "
                    f"failed to parse value beginning with {context[:80]!r}."
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"`opd_context` JSON must decode to a dict, got {type(loaded)}.")
            return loaded
        if isinstance(context, dict):
            return context
        raise TypeError(f"`opd_context` must be a dict or JSON string, got {type(context)}.")

    @staticmethod
    def serialize_context(context: Any) -> str:
        """Serialize OPD context for safe storage inside BaseSample.extra_kwargs."""
        context = OPDContextBuilder.normalize_context(context)
        return json.dumps(context, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format one context value as compact teacher prompt text."""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _should_drop(self, generator: Optional[torch.Generator]) -> bool:
        """Return whether to drop a context key for teacher context dropout."""
        if self.context_dropout <= 0:
            return False
        return torch.rand((), generator=generator).item() < self.context_dropout

    def build_prompt(
        self,
        prompt: str,
        context: Any,
        generator: Optional[torch.Generator] = None,
    ) -> str:
        """Build one teacher prompt from the student prompt and OPD context."""
        context_dict = self.normalize_context(context)
        lines = []
        unsupported_media_keys = []

        for key in self.context_keys:
            if key not in context_dict or context_dict[key] in (None, ""):
                continue
            if self._should_drop(generator):
                continue
            if key in MEDIA_CONTEXT_KEYS:
                unsupported_media_keys.append(key)
                continue
            label = key.replace("_", " ")
            lines.append(f"{label}: {self._format_value(context_dict[key])}")

        if unsupported_media_keys:
            raise ValueError(
                "This OPD MVP supports text-serializable teacher context only. "
                f"Unsupported media context keys requested: {unsupported_media_keys}. "
                "Use dense_caption/scene_graph for Wan T2V teachers, or add an image/video "
                "context encoder before enabling first_frame/last_frame/depth/pose/flow/tracks."
            )

        context_text = "\n".join(lines)
        if not context_text:
            return prompt
        return self.prompt_template.format(prompt=prompt, context=context_text)

    def build_prompts(
        self,
        prompts: List[str],
        contexts: List[Any],
        generator: Optional[torch.Generator] = None,
    ) -> List[str]:
        """Build teacher prompts for a batch."""
        if len(prompts) != len(contexts):
            raise ValueError(
                "Prompt/context batch mismatch: "
                f"{len(prompts)} prompts vs {len(contexts)} contexts."
            )
        return [
            self.build_prompt(prompt=prompt, context=context, generator=generator)
            for prompt, context in zip(prompts, contexts)
        ]
