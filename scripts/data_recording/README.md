# Data Recording & GR00T Post-Training

Record demonstration data from an RL specialist policy and use it to fine-tune a GR00T vision-language-action model.

## Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Record      │───▶│  2. Convert     │───▶│  3. Fine-tune   │───▶│  4. Evaluate    │
│  (RL Policy)    │    │  (HDF5→LeRobot) │    │  (GR00T)        │    │  (Closed-loop)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
     Isaac Lab              Python              Isaac-GR00T            Isaac Lab
```

## Prerequisites

- Trained RL policy checkpoint (from `scripts/train.py`)
- Isaac Lab environment with `G1-PickPlace-Tracking-v0-Record` task
- [Isaac-GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) repository for fine-tuning

---

## Step 1: Record Demonstration Data

Setup RL environment with appropriate observation group for required data collection. As an example, `G1-PickPlace-Tracking-v0-Record` environment captures RGB images, proprioceptive states, and actions from an RL policy. Observations are defined in `RecordObservationsCfg`.

```bash
python scripts/record.py \
    --task G1-PickPlace-Tracking-v0-Record \
    --checkpoint <path/to/rl/checkpoint.pt> \
    --record \
    --record_output data/recording \
    --num_envs 100 \
    --num_steps 300 \
    --enable_cameras
```

| Argument | Description |
|----------|-------------|
| `--task` | Environment with camera sensor (`G1-PickPlace-Tracking-v0-Record`) |
| `--checkpoint` | Path to trained RL policy |
| `--record` | Enable HDF5 recording |
| `--record_output` | Output directory (creates `data.h5` inside) |
| `--num_envs` | Parallel environments (more = faster collection) |
| `--num_steps` | Total simulation steps to run |

**Output:** `data/recording/data.h5` containing episodes with observations and actions.

### Inspect Recorded Data

Use the provided notebook to visualize trajectories and verify data quality:

```bash
jupyter notebook scripts/data_recording/inspect_data.ipynb
```

| File | Description |
|------|-------------|
| `data_recorder.py` | `MultiEnvDataRecorder` API |
| `inspect_data.ipynb` | Visualization notebook |
| `convert_to_gr00t.py` | HDF5 → LeRobot converter |

---

## Step 2: Convert to GR00T Format

Convert the HDF5 dataset to [LeRobot-compatible format](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md) for GR00T training:

```bash
python scripts/data_recording/convert_to_gr00t.py \
    -i data/recording/data.h5 \
    -o data/gr00t \
    --task "pick up the object"
```

This generates:
- `meta/` — Dataset metadata (episodes.jsonl, info.json, tasks.jsonl)
- `videos/` — MP4 videos from RGB observations
- `data/` — Parquet files with states and actions

---

## Step 3: Configure Data Pipeline

Once the GR00T dataset is generated, configure the data modalities for GR00T model training. The modality configuration specifies which observation channels (state, video, actions) are used and how they map to the training pipeline. You can tune this to find the best-performing combination for your task.

For the pick-place task, edit the `modality.json` file in the GR00T dataset directory as following:
```json
{
  "state": {
    "base_ang_vel": {
      "start": 0,
      "end": 3
    },
    "joint_pos": {
      "start": 6,
      "end": 23
    },
    "joint_vel": {
      "start": 23,
      "end": 38
    }
  },
  "action": {
    "action": {
      "start": 0,
      "end": 21
    }
  },
  "video": {
    "image": {
      "original_key": "observation.images.image"
    }
  },
  "annotation": {
    "human.action.task_description": {}
  }
}
```

Next, add a corresponding data config class to `gr00t/experiment/data_config.py` in the Isaac-GR00T repository. This class defines the data keys, transform pipeline, and observation/action horizons used during training. Below is an example for the G1 pick-place task:

<details>
<summary>G1PPTrackingSimDataConfig (click to expand)</summary>

```python
class G1PPTrackingSimDataConfig(BaseDataConfig):
    """Data config for G1 pick-place simulation dataset.

    This config defines the modalities, transforms, and horizons for GR00T
    post-training on data recorded from the G1 pick-place RL policy.
    Images are captured from an ego camera, and proprioceptive states include
    base angular velocity, upper body joint positions, and joint velocities.
    Actions are absolute joint position targets.
    """

    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.base_ang_vel",
        "state.joint_pos",
        "state.joint_vel",
    ]
    action_keys = [
        "action.action",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=64,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
```

</details>

Register in `DATA_CONFIG_MAP`:

```python
DATA_CONFIG_MAP["g1_pp_tracking_sim"] = G1PPTrackingSimDataConfig()
```

---

## Step 4: Fine-tune GR00T

Run fine-tuning in the [Isaac-GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) repository:

```bash
python scripts/gr00t_finetune.py \
    --dataset-path <path/to/gr00t/data> \
    --data-config g1_pp_tracking_sim \
    --video-backend torchvision_av \
    --num-gpus 1 \
    --max-steps 10000 \
    --output-dir outputs/gr00t-pp-tracking
```

> **Tip:** Increase `--max-steps` and add more data for better performance.

---

## Step 5: Closed-Loop Evaluation

### 5.1 Launch GR00T N1.5 Inference Server

In the [Isaac-GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) repository:

```bash
python scripts/inference_service.py \
    --server \
    --model_path <path/to/finetuned/checkpoint> \
    --embodiment-tag new_embodiment \
    --data-config g1_pp_tracking_sim \
    --denoising-steps 4 \
    --port 6666
```

### 5.2 Run Simulation Client

In **this repository**, connect to the server and run closed-loop evaluation:

```bash
python scripts/record.py \
    --task G1-PickPlace-Tracking-v0-GR00T-Inference \
    --gr00t \
    --gr00t_host localhost \
    --gr00t_port 6666 \
    --gr00t_task_description "Pick up object" \
    --gr00t_action_horizon 4 \
    --num_envs 1 \
    --enable_cameras
```

| Argument | Description |
|----------|-------------|
| `--gr00t` | Use GR00T policy instead of RL checkpoint |
| `--gr00t_host` | Inference server hostname |
| `--gr00t_port` | Inference server port |
| `--gr00t_task_description` | Language instruction for the task |
| `--gr00t_action_horizon` | Steps to execute per action chunk |

> **Note:** Use `--num_envs 1` for easier debugging. Increase for throughput testing.
