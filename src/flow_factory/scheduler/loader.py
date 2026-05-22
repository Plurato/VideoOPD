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

# src/flow_factory/scheduler/loader.py
"""
Scheduler Loader
Factory function to instantiate SDE schedulers from pipeline schedulers.
"""
import inspect
from typing import Any, Dict, Set, Type

from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .abc import SDESchedulerMixin
from .registry import get_sde_scheduler_class
from ..hparams import SchedulerArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def _get_scheduler_init_keys(scheduler_cls: Type) -> Set[str]:
    """Return constructor keys accepted by a scheduler class or its parents."""
    keys: Set[str] = set()
    for cls in scheduler_cls.mro():
        if cls is object:
            continue
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                keys.add(name)
    return keys


def _filter_scheduler_config(scheduler_cls: Type, config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop checkpoint scheduler config keys unsupported by the target scheduler."""
    accepted_keys = _get_scheduler_init_keys(scheduler_cls)
    if not accepted_keys:
        raise ValueError(f"Could not inspect scheduler constructor for {scheduler_cls.__name__}.")
    filtered_config = {key: value for key, value in config.items() if key in accepted_keys}
    dropped_keys = sorted(set(config) - set(filtered_config))
    if dropped_keys:
        logger.info(
            "Dropped scheduler config keys not accepted by %s: %s",
            scheduler_cls.__name__,
            dropped_keys,
        )
    return filtered_config


def load_scheduler(
    pipeline_scheduler: SchedulerMixin,
    scheduler_args: SchedulerArguments,
) -> SDESchedulerMixin:
    """
    Create an SDE scheduler from a pipeline scheduler and scheduler args.
    
    Merges the original scheduler config with SDE-specific args, then keeps
    only constructor keys accepted by the registered target scheduler.
    
    Args:
        pipeline_scheduler: Scheduler from pipeline.from_pretrained()
        scheduler_args: SchedulerArguments with SDE config
    
    Returns:
        Custom SDE scheduler instance
    
    Example:
        >>> pipe = DiffusionPipeline.from_pretrained("...")
        >>> sde_scheduler = load_scheduler(pipe.scheduler, scheduler_args)
    """
    sde_class = get_sde_scheduler_class(pipeline_scheduler)
    
    # Merge base config with SDE args
    base_config = dict(pipeline_scheduler.config)
    base_config.update(scheduler_args.to_dict())
    base_config = _filter_scheduler_config(sde_class, base_config)
    
    scheduler = sde_class(**base_config)
    logger.info(f"Loaded SDE scheduler: {sde_class.__name__}")
    return scheduler
