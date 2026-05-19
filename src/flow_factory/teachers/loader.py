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

# src/flow_factory/teachers/loader.py
from __future__ import annotations

from copy import deepcopy

from accelerate import Accelerator

from .opd_teacher import OPDTeacher
from ..hparams import Arguments, OPDTrainingArguments
from ..models.registry import get_model_adapter_class, list_registered_models
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def load_opd_teacher(config: Arguments, accelerator: Accelerator) -> OPDTeacher:
    """Load a frozen teacher adapter for OPD training."""
    teacher_args = config.teacher_args
    if teacher_args is None:
        raise ValueError(
            "OPD requires a top-level `teacher:` config block with at least "
            "`model_type` and `model_name_or_path`."
        )
    if teacher_args.model_name_or_path is None:
        raise ValueError("OPD requires `teacher.model_name_or_path` to be set.")
    if teacher_args.device.type == "cuda" and teacher_args.device.index is None:
        teacher_args.device = accelerator.device
    if not isinstance(config.training_args, OPDTrainingArguments):
        raise TypeError(
            "load_opd_teacher expects OPDTrainingArguments, "
            f"got {type(config.training_args).__name__}."
        )

    teacher_config = deepcopy(config)
    teacher_config.model_args = teacher_args
    teacher_config.model_args.finetune_type = "full"
    teacher_config.model_args.target_components = []

    try:
        adapter_cls = get_model_adapter_class(teacher_args.model_type)
    except ImportError as exc:
        registered_models = list(list_registered_models().keys())
        raise ImportError(
            f"Failed to load OPD teacher adapter '{teacher_args.model_type}'. "
            f"Available models: {registered_models}"
        ) from exc

    logger.info(
        "Loading OPD teacher model: %s (%s)",
        teacher_args.model_type,
        teacher_args.model_name_or_path,
    )
    adapter = adapter_cls(config=teacher_config, accelerator=accelerator)
    teacher = OPDTeacher(
        adapter=adapter,
        teacher_args=teacher_args,
        training_args=config.training_args,
        accelerator=accelerator,
    )
    teacher.prepare()
    return teacher
