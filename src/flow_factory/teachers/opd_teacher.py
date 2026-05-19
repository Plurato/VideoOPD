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

# src/flow_factory/teachers/opd_teacher.py
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator

from .context import OPDContextBuilder
from ..hparams import OPDTrainingArguments, TeacherArguments
from ..models.abc import BaseAdapter
from ..utils.base import filter_kwargs


class OPDTeacher:
    """Frozen teacher wrapper for OPD step-level velocity targets."""

    _SUPPORTED_REQUIRED_FORWARD_ARGS = {
        "self",
        "t",
        "latents",
        "prompt_embeds",
    }

    def __init__(
        self,
        adapter: BaseAdapter,
        teacher_args: TeacherArguments,
        training_args: OPDTrainingArguments,
        accelerator: Accelerator,
    ):
        self.adapter = adapter
        self.teacher_args = teacher_args
        self.training_args = training_args
        self.accelerator = accelerator
        self.context_builder = OPDContextBuilder(
            context_keys=training_args.teacher_context_keys,
            prompt_template=teacher_args.prompt_template,
            context_dropout=training_args.teacher_context_dropout,
        )

    def prepare(self) -> None:
        """Freeze teacher parameters, move runtime components to device, and validate forward."""
        for component_name in self.adapter._resolve_component_names(None):
            component = self.adapter.get_component(component_name)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
                component.eval()

        self.adapter.on_load_components(
            components=self.teacher_args.runtime_components,
            device=self.teacher_args.device,
        )
        self.adapter.eval()
        self._validate_forward_signature()

    def _validate_forward_signature(self) -> None:
        """Fail fast when a teacher forward requires unsupported non-text inputs."""
        signature = inspect.signature(self.adapter.forward)
        required = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unsupported = sorted(required - self._SUPPORTED_REQUIRED_FORWARD_ARGS)
        if unsupported:
            raise ValueError(
                "This OPD MVP supports text-conditioned teacher forwards only. "
                f"Teacher adapter {type(self.adapter).__name__}.forward requires unsupported "
                f"arguments: {unsupported}. Use a text-to-video teacher such as `wan2_t2v`, "
                "or add a teacher context encoder for those required inputs."
            )

    def encode_prompt(
        self,
        prompts: List[str],
        contexts: List[Any],
        negative_prompts: Optional[List[Optional[str]]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode teacher prompts after appending teacher-only context."""
        teacher_prompts = self.context_builder.build_prompts(
            prompts=prompts,
            contexts=contexts,
            generator=generator,
        )
        if self.teacher_args.negative_prompt is not None:
            negative_prompt_input = [self.teacher_args.negative_prompt] * len(teacher_prompts)
        else:
            negative_prompt_input = negative_prompts

        encode_kwargs = {
            "prompt": teacher_prompts,
            "negative_prompt": negative_prompt_input,
            "guidance_scale": self.training_args.teacher_guidance_scale,
            "device": self.teacher_args.device,
        }
        encode_kwargs = filter_kwargs(self.adapter.encode_prompt, **encode_kwargs)
        return self.adapter.encode_prompt(**encode_kwargs)

    def forward_velocity(
        self,
        batch: Dict[str, Any],
        contexts: List[Any],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        encoded_prompt: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return teacher velocity/noise prediction for the provided noised latents."""
        prompts = batch["prompt"]
        negative_prompts = batch.get("negative_prompt")
        if negative_prompts is not None and not isinstance(negative_prompts, list):
            negative_prompts = [negative_prompts] * len(prompts)

        encoded = encoded_prompt
        if encoded is None:
            encoded = self.encode_prompt(
                prompts=prompts,
                contexts=contexts,
                negative_prompts=negative_prompts,
                generator=generator,
            )
        prompt_embeds = encoded["prompt_embeds"].to(device=latents.device)
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=latents.device)

        t_flat = timestep.view(-1)
        forward_kwargs = {
            "t": t_flat,
            "t_next": torch.zeros_like(t_flat),
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "guidance_scale": self.training_args.teacher_guidance_scale,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"],
            "noise_level": 0.0,
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        output = self.adapter.forward(**forward_kwargs)
        return output.noise_pred
