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

# src/flow_factory/inference.py
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import torch
from accelerate import Accelerator
from diffusers.utils import export_to_video

from .hparams import Arguments
from .models.registry import get_model_adapter_class
from .samples import BaseSample
from .utils.base import filter_kwargs
from .utils.logger_utils import setup_logger
from .utils.video import numpy_to_video_frames, tensor_to_video_frames

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for video generation."""
    parser = argparse.ArgumentParser(description="Flow-Factory video inference")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training/inference YAML config. Recommended for fine-tuned checkpoints.",
    )
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text. Repeatable.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output mp4 path for one prompt, or output directory/pattern for multiple prompts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional fine-tuned checkpoint path. Omit for pretrained-only inference.",
    )
    parser.add_argument(
        "--resume-type",
        choices=["lora", "full"],
        default=None,
        help="Model checkpoint type. If omitted, local checkpoints are auto-detected.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model.model_name_or_path or provide it when --config is omitted.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override model.model_type or provide it when --config is omitted.",
    )
    parser.add_argument(
        "--finetune-type",
        choices=["full", "lora"],
        default=None,
        help="Override model.finetune_type. Usually inferred from --resume-type.",
    )
    parser.add_argument("--height", type=int, default=None, help="Generated video height.")
    parser.add_argument("--width", type=int, default=None, help="Generated video width.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of generated frames.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG guidance scale.")
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help="Optional second CFG scale for two-stage Wan models.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Prompt length cap.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size.")
    parser.add_argument("--fps", type=int, default=16, help="Saved video FPS.")
    parser.add_argument("--quality", type=float, default=8.0, help="MP4 quality from 0 to 10.")
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=16,
        help="imageio macro block size. Use 1 to disable automatic resizing.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Override mixed precision.",
    )
    parser.add_argument(
        "--attn-backend",
        type=str,
        default=None,
        help="Optional attention backend override, e.g. sage, flash, xformers.",
    )
    parser.add_argument(
        "--inference-kwargs-json",
        type=str,
        default=None,
        help="Extra adapter.inference kwargs as a JSON object.",
    )
    return parser.parse_args()


def _read_prompts(prompt_args: Optional[List[str]], prompt_file: Optional[str]) -> List[str]:
    """Load prompts from repeated --prompt values and an optional prompt file."""
    prompts = list(prompt_args or [])
    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if not prompts:
        raise ValueError("Provide at least one prompt via --prompt or --prompt-file.")
    return prompts


def _detect_local_resume_type(
    checkpoint: Optional[str],
    target_components: List[str],
) -> Optional[Literal["lora", "full"]]:
    """Detect local checkpoint type before adapter construction."""
    if checkpoint is None:
        return None
    checkpoint_path = Path(os.path.expanduser(checkpoint))
    if not checkpoint_path.exists():
        return None

    candidate_paths = (
        [checkpoint_path / component for component in target_components]
        if len(target_components) > 1
        else [checkpoint_path]
    )
    for candidate in candidate_paths:
        if (candidate / "adapter_config.json").exists():
            return "lora"
    return "full"


def _minimal_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a minimal config when no YAML config is provided."""
    if args.model_path is None or args.model_type is None:
        raise ValueError("--model-path and --model-type are required when --config is omitted.")
    return {
        "mixed_precision": args.mixed_precision or "bf16",
        "model": {
            "model_name_or_path": args.model_path,
            "model_type": args.model_type,
            "finetune_type": args.finetune_type or "full",
            "target_components": "transformer",
            "target_modules": "default",
            "resume_path": args.checkpoint,
            "resume_type": args.resume_type,
        },
        "train": {
            "trainer_type": "opd",
            "per_device_batch_size": 1,
            "group_size": 1,
            "unique_sample_num_per_epoch": 1,
            "gradient_step_per_epoch": 1,
            "num_inference_steps": args.num_inference_steps or 50,
        },
        "scheduler": {"dynamics_type": "ODE"},
        "log": {"logging_backend": "none"},
    }


def _load_config(args: argparse.Namespace) -> Arguments:
    """Load config and apply inference-time overrides."""
    if args.config is None:
        config = Arguments.from_dict(_minimal_config_dict(args))
    else:
        config = Arguments.load_from_yaml(args.config)

    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.model_path is not None:
        config.model_args.model_name_or_path = args.model_path
    if args.model_type is not None:
        config.model_args.model_type = args.model_type
    if args.attn_backend is not None:
        config.model_args.attn_backend = args.attn_backend
    if args.checkpoint is not None:
        config.model_args.resume_path = os.path.expanduser(args.checkpoint)
    if args.resume_type is not None:
        config.model_args.resume_type = args.resume_type
    if args.finetune_type is not None:
        config.model_args.finetune_type = args.finetune_type

    detected_resume_type = _detect_local_resume_type(
        config.model_args.resume_path,
        config.model_args.target_components,
    )
    if config.model_args.resume_type is None and detected_resume_type is not None:
        config.model_args.resume_type = detected_resume_type

    if config.model_args.resume_type == "state":
        raise ValueError(
            "Training-state checkpoints are not supported for inference. "
            "Use a model-only LoRA or full checkpoint and pass --resume-type lora/full."
        )
    if config.model_args.resume_type == "full":
        config.model_args.finetune_type = "full"
    elif config.model_args.resume_type == "lora":
        config.model_args.finetune_type = "lora"
    elif config.model_args.resume_path is None and config.model_args.finetune_type == "lora":
        logger.info(
            "No checkpoint provided; switching finetune_type to 'full' "
            "for pretrained inference."
        )
        config.model_args.finetune_type = "full"

    return config


def _parse_extra_kwargs(raw_json: Optional[str]) -> Dict[str, Any]:
    """Parse optional JSON kwargs for adapter.inference."""
    if raw_json is None:
        return {}
    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise ValueError("--inference-kwargs-json must decode to a JSON object.")
    return value


def _chunk(items: List[str], batch_size: int) -> List[List[str]]:
    """Split prompts into fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]


def _slugify(text: str, max_length: int = 48) -> str:
    """Create a compact filename-safe prompt slug."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("._-")
    return (slug or "prompt")[:max_length]


def _resolve_output_path(output: str, prompt: str, index: int, total: int) -> str:
    """Resolve output path for a generated sample."""
    output_path = Path(output)
    has_pattern = "{index}" in output or "{slug}" in output
    if has_pattern:
        resolved = output.format(index=index, slug=_slugify(prompt))
    elif total == 1 and output_path.suffix:
        resolved = str(output_path)
    elif output_path.suffix:
        raise ValueError(
            "Multiple prompts require --output to be a directory or a pattern containing "
            "{index} and/or {slug}."
        )
    else:
        resolved = str(output_path / f"{index:04d}_{_slugify(prompt)}.mp4")
    if Path(resolved).suffix == "":
        resolved = f"{resolved}.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
    return resolved


def _sample_to_frames(sample: BaseSample) -> List[Any]:
    """Convert a generated sample video to PIL frames."""
    if sample.video is None:
        raise ValueError("Generated sample does not contain a video field.")
    video = sample.video
    if isinstance(video, torch.Tensor):
        return tensor_to_video_frames(video)[0]
    if hasattr(video, "ndim"):
        return numpy_to_video_frames(video)[0]
    if isinstance(video, list):
        return video
    raise TypeError(f"Unsupported video output type: {type(video)}.")


def _first_config_value(config: Arguments, key: str) -> Any:
    """Read a generation default from eval args, then training args."""
    value = getattr(config.eval_args, key, None)
    if value is not None:
        return value
    return getattr(config.training_args, key, None)


def _resolve_generation_value(config: Arguments, args: argparse.Namespace, key: str) -> Any:
    """Resolve one generation value with CLI taking precedence over config."""
    cli_value = getattr(args, key)
    if cli_value is not None:
        return cli_value
    return _first_config_value(config, key)


def _build_inference_kwargs(
    args: argparse.Namespace,
    config: Arguments,
    prompts: List[str],
    generator: Optional[torch.Generator],
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build adapter.inference kwargs for one prompt batch."""
    kwargs: Dict[str, Any] = {
        "prompt": prompts,
        "compute_log_prob": False,
        "trajectory_indices": None,
        "generator": generator,
    }
    optional_values = {
        "negative_prompt": args.negative_prompt,
        "height": _resolve_generation_value(config, args, "height"),
        "width": _resolve_generation_value(config, args, "width"),
        "num_frames": _resolve_generation_value(config, args, "num_frames"),
        "num_inference_steps": _resolve_generation_value(config, args, "num_inference_steps"),
        "guidance_scale": _resolve_generation_value(config, args, "guidance_scale"),
        "guidance_scale_2": args.guidance_scale_2,
        "max_sequence_length": args.max_sequence_length,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    kwargs.update(extra_kwargs)
    return kwargs


def _freeze_adapter(adapter: Any) -> None:
    """Freeze all adapter components for inference."""
    for name in adapter._resolve_component_names(None):
        component = adapter.get_component(name)
        if component is not None and hasattr(component, "requires_grad_"):
            component.requires_grad_(False)
            component.eval()


def _component_devices(component: torch.nn.Module) -> Set[torch.device]:
    """Collect parameter and buffer devices for a module."""
    devices = {
        tensor.device
        for tensor in itertools.chain(
            component.parameters(recurse=True),
            component.buffers(recurse=True),
        )
    }
    return devices


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices while allowing index-free expected devices."""
    if actual.type != expected.type:
        return False
    if expected.index is None:
        return True
    return actual.index == expected.index


def _load_components_for_inference(
    adapter: Any,
    components: Union[str, List[str]],
    device: torch.device,
) -> None:
    """Move all standalone inference components to the target device."""
    names = adapter._resolve_component_names(components)
    for name in names:
        component = adapter.get_component(name)
        if component is not None and hasattr(component, "to"):
            # LoRA loading stores wrapped modules in _components before accelerator.prepare().
            # Standalone inference never prepares them, so we must move them explicitly.
            component.to(device)

    mismatched = []
    for name in names:
        component = adapter.get_component(name)
        if component is None or not isinstance(component, torch.nn.Module):
            continue
        devices = _component_devices(component)
        if devices and any(not _device_matches(module_device, device) for module_device in devices):
            device_list = ", ".join(sorted(str(module_device) for module_device in devices))
            mismatched.append(f"{name}: {device_list}")

    if mismatched:
        raise RuntimeError(
            "Inference components were not fully moved to the accelerator device "
            f"({device}). Mismatched components: {mismatched}."
        )


def run_inference(args: argparse.Namespace) -> List[str]:
    """Run video generation and save outputs."""
    prompts = _read_prompts(args.prompt, args.prompt_file)
    config = _load_config(args)
    extra_kwargs = _parse_extra_kwargs(args.inference_kwargs_json)

    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    adapter_cls = get_model_adapter_class(config.model_args.model_type)
    adapter = adapter_cls(config=config, accelerator=accelerator)
    _load_components_for_inference(
        adapter=adapter,
        components=list(adapter.preprocessing_modules) + list(adapter.inference_modules),
        device=accelerator.device,
    )
    _freeze_adapter(adapter)
    adapter.eval()

    saved_paths: List[str] = []
    sample_index = 0
    with torch.no_grad(), accelerator.autocast():
        for prompt_batch in _chunk(prompts, args.batch_size):
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(
                    args.seed + sample_index
                )
            inference_kwargs = _build_inference_kwargs(
                args,
                config,
                prompt_batch,
                generator,
                extra_kwargs,
            )
            inference_kwargs = filter_kwargs(adapter.inference, **inference_kwargs)
            samples = adapter.inference(**inference_kwargs)

            for local_index, sample in enumerate(samples):
                sample_index += 1
                prompt_text = sample.prompt or prompt_batch[local_index]
                output_path = _resolve_output_path(
                    output=args.output,
                    prompt=prompt_text,
                    index=sample_index,
                    total=len(prompts),
                )
                frames = _sample_to_frames(sample)
                export_to_video(
                    frames,
                    output_video_path=output_path,
                    fps=args.fps,
                    quality=args.quality,
                    macro_block_size=args.macro_block_size,
                )
                saved_paths.append(output_path)
                logger.info("Saved generated video to %s", output_path)

    return saved_paths


def main() -> None:
    """Run the Flow-Factory inference CLI."""
    run_inference(parse_args())


if __name__ == "__main__":
    main()
