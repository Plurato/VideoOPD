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

# src/flow_factory/trainers/opd.py
from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm_
from diffusers.utils.torch_utils import randn_tensor

from .abc import BaseTrainer
from ..hparams import OPDTrainingArguments
from ..samples import BaseSample
from ..teachers import load_opd_teacher
from ..teachers.context import OPDContextBuilder
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    to_broadcast_tensor,
)
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


class OPDTrainer(BaseTrainer):
    """Step-level OPD trainer with frozen teacher velocity matching."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDTrainingArguments
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range
        self.teacher = load_opd_teacher(self.config, self.accelerator)

    @property
    def enable_reward_weighting(self) -> bool:
        """Return whether scalar rewards should modulate OPD loss."""
        return self.training_args.opd_reward_weight > 0.0

    @property
    def enable_kl_loss(self) -> bool:
        """Return whether student reference KL is enabled."""
        return self.training_args.opd_kl_beta > 0.0

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample OPD training timesteps in scheduler scale [0, 1000]."""
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = [
            'logit_normal',
            'uniform',
            'discrete',
            'discrete_with_init',
            'discrete_wo_init',
        ]

        if strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        if strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        if strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        raise ValueError(f"Unknown time_sampling_strategy: {strategy}. Available: {available}")

    def _attach_opd_context(self, samples: List[BaseSample], batch: Dict[str, Any]) -> None:
        """Attach serialized teacher-only context from dataloader metadata to samples."""
        metadata = batch.get("metadata")
        if metadata is None:
            metadata = [{} for _ in samples]
        if len(metadata) != len(samples):
            raise ValueError(
                "Metadata/sample batch mismatch: "
                f"{len(metadata)} metadata rows vs {len(samples)} samples."
            )

        for sample, meta in zip(samples, metadata):
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                raise TypeError(f"Expected metadata rows to be dicts, got {type(meta)}.")
            context = meta.get("opd_context")
            if context is None:
                context = {
                    key: meta[key]
                    for key in self.training_args.teacher_context_keys
                    if key in meta
                }
            sample.extra_kwargs["opd_context"] = OPDContextBuilder.serialize_context(context)

    def evaluate(self) -> None:
        """Evaluate the student model with text prompt only."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None,
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
                self.eval_reward_buffer.add_samples(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split='pointwise')
            rewards = {
                key: torch.as_tensor(value).to(self.accelerator.device)
                for key, value in rewards.items()
            }
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }

            if self.accelerator.is_main_process:
                log_data = {
                    f'eval/reward_{key}_mean': np.mean(value)
                    for key, value in gathered_rewards.items()
                }
                log_data.update(
                    {
                        f'eval/reward_{key}_std': np.std(value)
                        for key, value in gathered_rewards.items()
                    }
                )
                log_data['eval_samples'] = all_samples
                self.log_data(log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    def start(self):
        """Run the OPD six-stage training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

        if self.log_args.save_freq > 0 and self.log_args.save_dir and self.epoch > 0:
            save_dir = os.path.join(
                self.log_args.save_dir,
                str(self.log_args.run_name),
                'checkpoints',
            )
            self.save_checkpoint(save_dir, epoch=self.epoch)

    def sample(self) -> List[BaseSample]:
        """Generate student rollouts and keep only final latents for OPD matching."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = self.training_args.opd_trajectory_indices
        if trajectory_indices is None:
            trajectory_indices = [-1]

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,
                    'trajectory_indices': trajectory_indices,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                self._attach_opd_context(sample_batch, batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)
        return samples

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute optional scalar reward advantages for OPD reward weighting."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Compute optional scalar rewards and advantages."""
        if not self.enable_reward_weighting:
            return
        if not self.reward_models:
            raise ValueError(
                "`opd_reward_weight` is positive, but no training reward model is configured."
            )

        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    def _compute_student_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Return student velocity/noise prediction for one noised latent batch."""
        t_flat = timestep.view(-1)
        excluded = {'all_latents', 'timesteps', 'advantage', 'rewards', 'opd_context'}
        forward_kwargs = {
            **self.training_args,
            't': t_flat,
            't_next': torch.zeros_like(t_flat),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': 0.0,
            **{k: v for k, v in batch.items() if k not in excluded},
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        output = self.adapter.forward(**forward_kwargs)
        return output.noise_pred

    def _compute_opd_loss(
        self,
        student_v_pred: torch.Tensor,
        teacher_v_pred: torch.Tensor,
        noised_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample OPD loss."""
        if student_v_pred.shape != teacher_v_pred.shape:
            raise ValueError(
                "Teacher/student velocity shape mismatch: "
                f"student={tuple(student_v_pred.shape)}, teacher={tuple(teacher_v_pred.shape)}. "
                "Use latent-compatible teacher/student models for OPD."
            )

        reduce_dims = tuple(range(1, student_v_pred.ndim))
        if self.training_args.opd_loss_type == 'velocity_mse':
            return F.mse_loss(
                student_v_pred.float(),
                teacher_v_pred.float(),
                reduction='none',
            ).mean(dim=reduce_dims)
        if self.training_args.opd_loss_type == 'x0_mse':
            sigma = to_broadcast_tensor(flow_match_sigma(timestep.view(-1)), noised_latents)
            student_x0 = noised_latents.float() - sigma.float() * student_v_pred.float()
            teacher_x0 = noised_latents.float() - sigma.float() * teacher_v_pred.float()
            return F.mse_loss(student_x0, teacher_x0, reduction='none').mean(dim=reduce_dims)
        raise ValueError(f"Unknown OPD loss type: {self.training_args.opd_loss_type}.")

    def _apply_reward_weighting(
        self,
        per_sample_loss: torch.Tensor,
        batch: Dict[str, Any],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Apply optional advantage-based weighting to per-sample OPD loss."""
        if not self.enable_reward_weighting:
            return per_sample_loss
        if "advantage" not in batch:
            raise ValueError("OPD reward weighting is enabled, but batch has no `advantage` field.")

        adv = batch["advantage"]
        adv_clip_range = self.training_args.adv_clip_range
        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
        weights = torch.clamp(1.0 + self.training_args.opd_reward_weight * adv, min=0.0)
        loss_info["reward_weight"].append(weights.detach())
        return per_sample_loss * weights

    def optimize(self, samples: List[BaseSample]) -> None:
        """Optimize student with teacher velocity matching on forward-process timesteps."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]
            loss_info = defaultdict(list)

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    sample.to(device)
                    for sample in shuffled_samples[start:start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                batch_size = batch['all_latents'].shape[0]
                clean_latents = batch['all_latents'][:, -1]
                contexts = batch.get("opd_context", ["{}"] * batch_size)
                teacher_context_gen = create_generator(
                    self.training_args.seed,
                    self.epoch,
                    inner_epoch,
                    batch_idx,
                )
                with torch.no_grad():
                    teacher_encoded = self.teacher.encode_prompt(
                        prompts=batch["prompt"],
                        contexts=contexts,
                        negative_prompts=batch.get("negative_prompt"),
                        generator=teacher_context_gen,
                    )

                self.adapter.train()
                all_timesteps = self._sample_timesteps(batch_size)

                with self.autocast():
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            t_flat = all_timesteps[t_idx]
                            sigma = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                            noise = randn_tensor(
                                clean_latents.shape,
                                device=clean_latents.device,
                                dtype=clean_latents.dtype,
                            )
                            noised_latents = (1 - sigma) * clean_latents + sigma * noise

                            student_v_pred = self._compute_student_output(
                                batch=batch,
                                timestep=t_flat,
                                noised_latents=noised_latents,
                            )
                            with torch.no_grad():
                                teacher_v_pred = self.teacher.forward_velocity(
                                    batch=batch,
                                    contexts=contexts,
                                    latents=noised_latents,
                                    timestep=t_flat,
                                    encoded_prompt=teacher_encoded,
                                )

                            per_sample_loss = self._compute_opd_loss(
                                student_v_pred=student_v_pred,
                                teacher_v_pred=teacher_v_pred.detach(),
                                noised_latents=noised_latents,
                                timestep=t_flat,
                            )
                            weighted_loss = self._apply_reward_weighting(
                                per_sample_loss=per_sample_loss,
                                batch=batch,
                                loss_info=loss_info,
                            )
                            opd_loss = self.training_args.opd_teacher_weight * weighted_loss.mean()
                            loss = opd_loss

                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_v_pred = self._compute_student_output(
                                        batch=batch,
                                        timestep=t_flat,
                                        noised_latents=noised_latents,
                                    )
                                kl_div = F.mse_loss(
                                    student_v_pred.float(),
                                    ref_v_pred.float(),
                                    reduction='none',
                                ).mean(dim=tuple(range(1, student_v_pred.ndim)))
                                kl_loss = self.training_args.opd_kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            loss_info['opd_loss'].append(opd_loss.detach())
                            loss_info['unweighted_opd_loss'].append(per_sample_loss.detach())
                            loss_info['loss'].append(loss.detach())

                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
