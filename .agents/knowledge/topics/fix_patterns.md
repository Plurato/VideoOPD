# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Launcher process count exceeds visible GPUs
- **Date**: 2026-05-20
- **Symptom**: Accelerate/DeepSpeed failed during initialization with `ValueError: device_id cuda:7 is out of range. Please use a device index less than the number of accelerators available: 4.`
- **Root Cause**: The launch configuration requested 8 local processes on a node where only 4 CUDA devices were visible, so local rank 7 mapped to a nonexistent `cuda:7`.
- **Fix**: `cli.py` now validates `num_processes / num_machines` against the visible local GPU count before launching, `train.py` validates externally supplied `LOCAL_RANK` before constructing `Accelerator`, and the OPD Wan2.1 example uses `num_processes: 4` with matching sampler geometry comments.
- **Lesson**: Distributed process geometry must match `CUDA_VISIBLE_DEVICES` per node. Validate before distributed initialization because the later DeepSpeed error hides the configuration-level cause.
- **Related Constraint**: N/A

### Scheduler config contains incompatible checkpoint keys
- **Date**: 2026-05-23
- **Symptom**: OPD teacher loading failed while replacing the checkpoint scheduler with `TypeError: FlowMatchEulerDiscreteScheduler.__init__() got an unexpected keyword argument 'beta_end'`.
- **Root Cause**: `scheduler/loader.py` passed the checkpoint scheduler config verbatim into the registered SDE scheduler, but the Wan reward teacher checkpoint can carry scheduler keys such as `beta_end` that are valid for other schedulers and invalid for `FlowMatchEulerDiscreteScheduler`.
- **Fix**: `scheduler/loader.py` now filters merged scheduler config keys against the target scheduler class and its parent constructors before instantiation, while preserving Flow-Factory SDE args such as `noise_level`, `sde_steps`, `num_sde_steps`, `seed`, and `dynamics_type`.
- **Lesson**: Checkpoint scheduler configs are not a stable constructor contract across scheduler families. When wrapping a diffusers scheduler class, instantiate from keys accepted by the target scheduler rather than blindly forwarding every serialized config field.
- **Related Constraint**: N/A

### Standalone inference skips LoRA component device move
- **Date**: 2026-05-23
- **Symptom**: Wan LoRA inference failed with `RuntimeError: Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be the same` inside transformer `patch_embedding`.
- **Root Cause**: LoRA checkpoint loading stores wrapped target components in `BaseAdapter._components`, and `on_load_components()` skips cached components because training treats them as accelerator-managed after `accelerator.prepare()`; standalone inference never prepares them, so the wrapped transformer stayed on CPU.
- **Fix**: `inference.py` now moves all standalone inference components explicitly, including LoRA-wrapped cached components, and fails fast if any parameter or buffer remains off the accelerator device before generation starts.
- **Lesson**: `_components` does not always imply accelerator-managed. Standalone utilities that bypass trainer initialization must not rely on `on_load_components()` for LoRA-wrapped target modules.
- **Related Constraint**: #8

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
