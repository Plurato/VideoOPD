# dataset/opd_wan21

A small, self-contained text-to-video prompt set used as the MVP dataset for
the OPD (Online Process Distillation) trainer with the Wan2.1 T2V family.

## Files

- `train.jsonl` — 98 curated prompts for training rollouts.
- `test.jsonl` — 12 held-out prompts for periodic evaluation.

Each row is a single JSON object with at least a `prompt` field. Optional
`dense_caption` / `scene_graph` fields are also included for a subset of rows;
they are surfaced **only to the frozen teacher prompt encoder** via
`OPDContextBuilder` and never seen by the student. Missing context keys are
gracefully skipped, so adding new prompts without those fields is safe.

```json
{"prompt": "A red kite flies over a green meadow on a windy day.", "dense_caption": "A bright crimson kite with a long ribbon tail rises over a sunlit meadow as the wind bends the tall grass."}
```

## Wiring

`examples/opd/lora/wan21/t2v_reward_teacher.yaml` already points its `data.dataset_dir`
to `dataset/opd_wan21`. The `data.max_dataset_size: 1000` cap leaves room for
the user to extend either split without touching the YAML.

## Why curated prompts (and not a Hub download)

The MVP is meant to be runnable out of the box on a fresh checkout, so we
ship the prompts in-repo. To replace this with a larger corpus, drop a
`train.jsonl` / `test.jsonl` (or `train.txt` / `test.txt`) with the same
schema into a new directory and update the YAML `dataset_dir`.

## License

Prompts are short, author-written text descriptions and are released under the
repository's Apache-2.0 license.
