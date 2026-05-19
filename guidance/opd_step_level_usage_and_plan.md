# Flow-Factory 使用说明与 OPD Step-Level 迁移计划

本文档基于当前仓库的静态代码阅读整理。当前本地机器只用于代码与文档修改，不用于训练、推理、安装依赖或跑测试；所有需要 GPU、完整依赖和模型权重的命令都应在远程服务器上执行。

## 1. 代码仓库如何组织

Flow-Factory 的核心抽象是三条 registry 驱动的主线：

- `trainers/`：训练算法。入口由 `train.trainer_type` 选择，当前包括 `grpo`、`grpo-guard`、`dpo`、`dgpo`、`nft`、`awm`、`crd`。
- `models/`：diffusers pipeline 的 adapter。入口由 `model.model_type` 选择，当前包括 FLUX、SD3.5、Qwen-Image、Z-Image、Wan2 T2V/I2V/V2V、LTX2 T2AV/I2AV。
- `rewards/`：reward model。入口由 `rewards[*].reward_model` 选择，支持本地 reward、VLM-as-judge、remote reward server、pointwise/groupwise reward。

实际训练入口是：

```bash
ff-train path/to/config.yaml
```

它会调用 `src/flow_factory/cli.py`，自动判断是单进程、外部 launcher 进程，还是需要 `accelerate launch`。真正的训练主函数在 `src/flow_factory/train.py`，流程是：

```text
YAML -> Arguments -> load_model(adapter) -> load_trainer(trainer) -> trainer.start()
```

## 2. 训练应该怎么跑

### 2.1 安装与环境

在远程 GPU 服务器上安装：

```bash
pip install -e .
```

如果需要 DeepSpeed、量化或实验日志：

```bash
pip install -e ".[deepspeed]"
pip install -e ".[all]"
pip install -e ".[wandb]"
pip install -e ".[swanlab]"
```

LTX2 依赖仓库内置的 diffusers 子模块：

```bash
git submodule update --init
pip install -e ./diffusers
```

### 2.2 选择一个示例配置

示例配置遵循：

```text
examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml
```

常用起点：

```bash
ff-train examples/grpo/lora/flux1/default.yaml
ff-train examples/grpo/lora/wan21/t2v.yaml
ff-train examples/nft/lora/wan22/t2v.yaml
ff-train examples/grpo/lora/ltx2/t2av.yaml
```

视频模型建议优先从 Wan/LTX2 的现有 YAML 改起，因为这些配置已经显式设置了 `offload_samples_to_cpu: true`，否则 rollout buffer 和优化阶段容易占用大量显存。

### 2.3 YAML 的关键字段

顶层结构：

```yaml
launcher: "accelerate"
config_file: config/deepspeed/deepspeed_zero2.yaml
num_processes: 8
mixed_precision: "bf16"

data:      # dataset, cache, preprocessing, sampler
model:     # model type/path, LoRA/full, checkpoint resume
train:     # algorithm-specific training args
scheduler: # ODE/SDE dynamics
eval:      # validation sampling args
log:       # checkpoint and logging
rewards:   # one or more reward models
```

最小训练命令：

```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

多机时，`ff-train` 会读取 `MASTER_IP`/`MASTER_ADDR`、`MASTER_PORT`、`MACHINE_RANK`/`NODE_RANK`、`NUM_MACHINES`/`NUM_NODES`、`GPUS_PER_NODE` 等环境变量；也可以用命令行覆盖：

```bash
ff-train config.yaml \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500
```

### 2.4 数据格式

Text-to-video 和 text-to-image 最简单，`train.txt` 每行一个 prompt：

```text
A dog running through a neon-lit street.
A close-up video of water drops on a leaf.
```

也可以用 `train.jsonl`：

```jsonl
{"prompt": "A dog running through a neon-lit street."}
{"prompt": "A close-up video of water drops on a leaf.", "negative_prompt": "blur, low quality"}
```

Image/video 条件任务使用 `image`/`images`、`video`/`videos` 字段：

```jsonl
{"prompt": "Animate this image.", "image": "frame_0001.png"}
{"prompt": "Transform this clip.", "video": "clip_001.mp4"}
```

默认目录：

- 图片：`{dataset_dir}/images`
- 视频：`{dataset_dir}/videos`
- 音频：`{dataset_dir}/audios`

可以在 YAML 里用 `image_dir`、`video_dir`、`audio_dir` 覆盖。

### 2.5 一个 epoch 内部发生什么

训练遵循六阶段：

```text
1. Data Preprocessing
2. K-Repeat Sampling
3. Trajectory Generation
4. Reward Computation
5. Advantage Computation
6. Policy Optimization
```

对应到 trainer 方法：

- 初始化时：`get_dataloader()` 调 adapter 的 `preprocess_func()`，把 prompt/image/video/audio 编码后缓存。
- 每个 epoch：`sample()` 用 `adapter.inference()` 生成样本和 latent 轨迹。
- `prepare_feedback()` 用 `RewardBuffer`/`RewardProcessor` 算 reward，再用 `AdvantageProcessor` 算 advantage。
- `optimize()` 调 `adapter.forward()` 做单步 denoising 训练。

最重要的不变量是 train-inference consistency：rollout 阶段 `inference()` 内部每一步也必须调用同一个 `forward()`，优化阶段再用同样的输入重放这个 `forward()`。

### 2.6 选择算法

- `grpo` / `grpo-guard`：coupled paradigm。采样和训练 timestep 绑定，需要 SDE log-prob，`scheduler.dynamics_type` 应使用 `Flow-SDE`、`Dance-SDE` 或 `CPS`。
- `nft` / `awm` / `dpo` / `dgpo` / `crd`：decoupled paradigm。采样可以用 `ODE`，训练时重新采样 flow timestep。
- 视频大模型优先用 LoRA；full fine-tune 和 KL/reference 都会显著增加显存。

### 2.7 Checkpoint

checkpoint 保存位置通常是：

```text
{log.save_dir}/{log.run_name}/checkpoints/checkpoint-{epoch}
```

`log.save_model_only: true` 时只保存模型权重：

- LoRA：保存 PEFT adapter。
- Full：保存目标 transformer/component 的权重。

`model.resume_path` 可以指向本地目录或 Hugging Face repo spec；`model.resume_type` 可设为 `lora`、`full`、`state`，为 `null` 时自动检测。

## 3. 推理应该怎么跑

当前仓库没有统一的 `ff-infer` CLI。推理方式是使用对应的 diffusers pipeline，并加载训练保存的 LoRA 或 full checkpoint。仓库提供了 `inference/example_lora.py` 和 `inference/example_full.py` 作为 FLUX.1 示例。

### 3.1 LoRA 推理模式

```python
import torch
from diffusers import FluxPipeline
from peft import PeftModel

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

checkpoint = "saves/run_name/checkpoints/checkpoint-20"
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    checkpoint,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()

image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("result.png")
```

如果训练配置里有多个 `target_components`，checkpoint 下会出现组件子目录，例如 `transformer/`、`transformer_2/`；推理时需要分别把 LoRA 加载到对应 component。

### 3.2 Full checkpoint 推理模式

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

checkpoint = "saves/run_name/checkpoints/checkpoint-20"
pipe.transformer = FluxTransformer2DModel.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
```

视频模型同理：选择 Wan 或 LTX2 对应 diffusers pipeline，加载训练保存的 LoRA/full component，然后用 pipeline 原生 `__call__()` 做推理。训练时的 Flow-Factory adapter 负责 RL 训练的一致性和轨迹收集，部署推理时不必经过 trainer。

## 4. 针对 OPD Step-Level 项目的改造目标

你的研究目标可以概括为：

```text
Student: text prompt -> video latent trajectory
Teacher: text prompt + extra context -> score/supervise student latent steps
目标: 把 context 中的隐式结构蒸馏进只看 text 的 student
```

这里的 context 可以包括 first/last frame、dense caption、scene graph、depth、pose、optical flow、object tracks 等。teacher 和 student 可以是同一个视频模型、同家族模型，或者 latent 几何兼容的两个 adapter。

这个想法和仓库当前设计最接近的是 `CRDTrainer`、`NFTTrainer`、`AWMTrainer`：

- `CRDTrainer` 已经有 teacher/reference 风格的 no-grad forward、step-level flow loss、centered reward distillation。
- `NFTTrainer` 和 `AWMTrainer` 已经在 student 生成的 final latent 上重新采样 flow timestep，做 dense forward-process 训练。
- `GRPOTrainer` 已经严格实现了 rollout `inference()` 和 training `forward()` 的重放一致性。

推荐不要把 OPD 做成普通 reward model。原因是现有 reward 接口主要返回 per-sample scalar reward，而 OPD 需要 teacher 在 latent/timestep 上给 dense step-level 信号，最好作为新 trainer/algorithm 实现。

## 5. 推荐实现路线

### Phase 0: 定义 MVP 边界

先做最小可验证版本：

- student：`wan2_t2v` 或 `ltx2_t2av`，只输入 text prompt。
- teacher：同家族 adapter，冻结参数，额外读取 context。
- context MVP：先支持 `dense_caption`、`scene_graph`、`first_frame`、`last_frame` 四类；depth/pose/flow/tracks 后续接入。
- 训练目标：teacher velocity matching，即在相同 `x_t, t` 上最小化 `||v_student(x_t,t,text) - stopgrad(v_teacher(x_t,t,text,context))||^2`，再叠加可选 reward/advantage/KL。

### Phase 1: 数据与 context 传递

当前 `GeneralDataset._preprocess_batch()` 会把非 `prompt/negative_prompt/images/videos/audios` 的字段保存在 `metadata`。需要把这些 metadata 显式带到 sample：

1. 在 `BaseTrainer` 增加一个轻量 helper，例如 `_attach_batch_metadata(samples, batch)`。
2. 在各 trainer 的 `sample()` 中，`adapter.inference()` 返回 `sample_batch` 后，把 `batch["metadata"][i]` 写入 `sample.extra_kwargs["opd_context"]`。
3. 对 OPD trainer 先只在自己的 `sample()` 中调用该 helper，避免立刻改动全部 trainer 行为。
4. 约定 JSONL context 格式，例如：

```jsonl
{
  "prompt": "A person opens a red umbrella in the rain.",
  "opd_context": {
    "dense_caption": "The scene contains...",
    "scene_graph": {"objects": [], "relations": []},
    "first_frame": "first/0001.png",
    "last_frame": "last/0001.png",
    "depth": "depth/0001.pt",
    "pose": "pose/0001.json",
    "optical_flow": "flow/0001.pt",
    "object_tracks": "tracks/0001.json"
  }
}
```

对于 first/last frame，不建议复用主任务的 `image` 字段，否则 image-conditioned adapter 可能把它当成 student 条件。应放进 `opd_context`，由 teacher-only context loader 读取。

### Phase 2: 新增 OPD hparams

在 `src/flow_factory/hparams/training_args.py` 增加：

- `OPDTrainingArguments`，建议继承 `TrainingArguments` 或参考 `CRDTrainingArguments`。
- 字段建议：
  - `trainer_type: "opd"`
  - `opd_loss_type: "velocity_mse"`，预留 `"x0_mse"`、`"teacher_logprob"`、`"hybrid"`
  - `opd_teacher_weight: float`
  - `opd_reward_weight: float`
  - `opd_kl_beta: float`
  - `num_train_timesteps`
  - `time_sampling_strategy`
  - `timestep_range`
  - `teacher_guidance_scale`
  - `teacher_context_keys`
  - `teacher_context_dropout`
  - `store_student_trajectory: bool`
  - `opd_trajectory_indices`

同步更新：

- `get_training_args_class()`
- `examples/` 下新增 OPD 示例配置
- `guidance/algorithms.md` 增加 OPD 小节
- `.agents/knowledge/architecture.md` registry 表增加 `opd`

### Phase 3: Teacher 配置与加载

建议新增 `TeacherArguments`，顶层 YAML 增加：

```yaml
teacher:
  model_type: "wan2_i2v"
  model_name_or_path: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
  checkpoint_path: null
  device: "cuda"
  dtype: "bfloat16"
  context_keys: ["first_frame", "last_frame", "dense_caption", "scene_graph"]
  offload: true
```

实现上建议做一个独立 loader，而不是复用 student 的 optimizer 初始化：

- `src/flow_factory/teachers/loader.py`
- `src/flow_factory/teachers/context.py`
- `src/flow_factory/teachers/opd_teacher.py`

teacher 必须：

- 全部冻结参数。
- no-grad forward。
- 不进入 optimizer。
- 可选择 offload 或放到单独 device。
- fail-fast 检查 teacher/student latent shape、scheduler timestep scale、`num_frames`、VAE scale 是否兼容。

### Phase 4: 新增 `OPDTrainer`

新增文件：

```text
src/flow_factory/trainers/opd.py
```

并在 `src/flow_factory/trainers/registry.py` 注册：

```python
"opd": "flow_factory.trainers.opd.OPDTrainer"
```

建议训练流程：

```text
sample():
  student adapter rollout
  adapter.inference(compute_log_prob=False, trajectory_indices=opd_trajectory_indices or [-1])
  attach opd_context from batch metadata
  optional: existing reward_buffer.add_samples()

prepare_feedback():
  optional existing rewards -> advantages
  context validation and grouping stats

optimize():
  for each micro-batch:
    reload sample to GPU
    choose timesteps:
      A. trajectory mode: use stored rollout x_t and t
      B. forward-process mode: x_t = (1 - sigma) * x_1 + sigma * noise
    student_out = student.forward(... prompt-only ...)
    with no_grad:
      teacher_inputs = build_teacher_inputs(batch["opd_context"])
      teacher_out = teacher.forward(... prompt + context ...)
    opd_loss = mse(student_out.noise_pred, teacher_out.noise_pred)
    optional reward weighting: opd_loss *= f(advantage)
    optional KL/reference regularization
    backward + optimizer step
```

两种 timestep 模式的取舍：

- `trajectory mode` 更贴近“teacher 在 student 自己生成的 latent trajectory 上打分”，训练/推理一致性最好，但视频模型存全轨迹会贵。建议先支持稀疏 step，例如 early/mid/late 三个 step。
- `forward-process mode` 更接近 NFT/AWM，显存更稳，容易先跑通；但它不是严格的 student rollout trajectory，只是基于 student final latent 重新加噪。

### Phase 5: Teacher context adapter

不同 context 类型应通过小型 registry 映射到 teacher 输入，而不是硬编码进 trainer：

```text
context dict -> ContextBatch -> teacher adapter kwargs
```

建议接口：

```python
class OPDContextEncoder:
    def load(self, context_batch, device, dtype) -> Dict[str, Any]:
        ...
```

初期映射：

- `dense_caption`：拼接/模板化到 teacher prompt，不给 student。
- `scene_graph`：转成结构化文本，拼接到 teacher prompt。
- `first_frame` / `last_frame`：加载成 image tensor，传给 teacher I2V adapter 的 `condition_images` 或定制 kwargs。
- `depth` / `pose` / `optical_flow` / `object_tracks`：先作为结构文本或 tensor extra kwargs 存储；等选定具体 teacher pipeline 后，再接入 ControlNet/condition encoder/自定义 adapter。

### Phase 6: 示例配置

建议新增：

```text
examples/opd/lora/wan21/t2v_teacher_i2v.yaml
examples/opd/lora/wan22/t2v_teacher_i2v.yaml
```

关键配置建议：

```yaml
train:
  trainer_type: "opd"
  resolution: 256
  num_frames: 5
  num_inference_steps: 10
  per_device_batch_size: 1
  group_size: 4
  unique_sample_num_per_epoch: 32
  offload_samples_to_cpu: true
  num_train_timesteps: 3
  time_sampling_strategy: "discrete"
  timestep_range: 0.9
  opd_loss_type: "velocity_mse"
  opd_teacher_weight: 1.0
  opd_reward_weight: 0.2

scheduler:
  dynamics_type: "ODE"

teacher:
  model_type: "wan2_i2v"
  model_name_or_path: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
  context_keys: ["first_frame", "last_frame", "dense_caption"]
```

## 6. 需要特别注意的工程风险

1. 不要改 `BaseAdapter.forward()` 的抽象签名。teacher context 应通过 adapter-specific kwargs 和 `filter_kwargs()` 进入，或者由 OPD teacher wrapper 做转换。
2. 不要让 student 看到 teacher-only context。数据字段要保存在 `opd_context`，不能误放进 student adapter 的 `images/videos` 条件字段。
3. teacher/student 必须使用兼容的 latent 几何。不同模型家族的 latent channel、scale、timestep convention 不一致时，必须显式 adapter/projection，不能静默训练。
4. 视频样本要默认打开 `offload_samples_to_cpu`，并避免默认存全轨迹。
5. 如果引入新 config 字段，要同步更新 hparams、示例 YAML、guidance 文档和 registry。
6. 如果新增 `encode_context` 之类 adapter API，必须做成非抽象 no-op default，避免强迫所有 adapter 改 stub。
7. 本地不要运行训练、推理、测试或安装命令。远程服务器才是验证环境。

## 7. 建议的第一版开发任务清单

1. 添加 `OPDTrainingArguments` 和 `opd` trainer registry。
2. 添加 `OPDTrainer`，先实现 forward-process mode 的 teacher velocity matching。
3. 添加 `opd_context` metadata 传递 helper，只在 OPDTrainer 中使用。
4. 添加 teacher loader/context loader，先支持 dense caption/scene graph 文本拼接和 first/last frame。
5. 添加 Wan T2V student + Wan I2V teacher 示例 YAML。
6. 在远程服务器上先跑小分辨率、小帧数、小 epoch sanity check：
   - teacher/student forward shape 对齐。
   - loss 能反传到 student。
   - teacher 参数无梯度。
   - student 推理时只用 text prompt。
7. 再接入外部 scalar rewards，把 OPD dense loss 与现有 advantage/reward pipeline 混合。

