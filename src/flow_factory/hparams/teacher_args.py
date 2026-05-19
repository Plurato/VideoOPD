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

# src/flow_factory/hparams/teacher_args.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from .model_args import ModelArguments, dtype_map


@dataclass
class TeacherArguments(ModelArguments):
    r"""Arguments for a frozen teacher model used by distillation trainers."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or Hugging Face identifier for the frozen teacher model."},
    )
    model_type: str = field(
        default="wan2_t2v",
        metadata={"help": "Registered model adapter type for the teacher model."},
    )
    finetune_type: str = field(
        default="full",
        metadata={"help": "Teacher is always loaded as a frozen full model."},
    )
    target_components: Union[str, List[str]] = field(
        default_factory=list,
        metadata={"help": "Teacher trainable components. Must stay empty for frozen teachers."},
    )
    target_modules: Union[str, List[str]] = field(
        default="default",
        metadata={"help": "Unused for frozen teachers; kept for ModelArguments compatibility."},
    )
    device: Union[str, torch.device] = field(
        default="cuda",
        metadata={"help": "Device where teacher forward modules are loaded."},
    )
    runtime_components: List[str] = field(
        default_factory=lambda: ["text_encoders", "transformers"],
        metadata={
            "help": (
                "Teacher components to keep on the teacher device. Text-only OPD needs "
                "text encoders and transformers; VAE is intentionally omitted."
            )
        },
    )
    prompt_template: str = field(
        default="{prompt}\n\nTeacher-only context:\n{context}",
        metadata={
            "help": "Template used to build the teacher prompt from prompt and context text."
        },
    )
    negative_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional teacher-only negative prompt override."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is not None:
            self.model_name_or_path = str(self.model_name_or_path)
        self.finetune_type = "full"
        if self.target_components:
            raise ValueError(
                "`teacher.target_components` must be empty because OPD teacher models are frozen "
                "and must not enter optimizer or accelerator.prepare()."
            )
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if isinstance(self.master_weight_dtype, str):
            self.master_weight_dtype = dtype_map[self.master_weight_dtype]

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["device"] = str(self.device)
        return d
