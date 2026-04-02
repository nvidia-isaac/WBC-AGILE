# WBC-AGILE Docker 部署指南

基于 NVIDIA 官方 `nvcr.io/nvidia/isaac-lab:2.3.1` 镜像部署 WBC-AGILE RL 训练环境。

---

## 1. 基础镜像信息

| 组件 | 版本 | 镜像内路径 |
|------|------|-----------|
| 基础镜像 | `nvcr.io/nvidia/isaac-lab:2.3.1` | — |
| Isaac Sim | 5.1.0-rc.19 | `/isaac-sim/` |
| Isaac Lab | 2.3.1 | `/workspace/isaaclab/` |
| Python | 3.11.13 | `/isaac-sim/kit/python/bin/python3.11` |
| Kit kernel | 106.x | `/isaac-sim/kit/` |
| Vulkan ICD | headless（不需要 Xvfb） | `/etc/vulkan/icd.d/nvidia_icd.json` |
| USD 资产 S3 根路径 | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1` | 由 `isaaclab.python.headless.kit` 设定 |

> **为什么用 Docker 而非宿主机直接跑？**
> - 宿主机 Isaac Sim 4.5 的 S3 路径上 T1 USD 404、G1 USD 关节版本不匹配；Isaac 5.1 两者均正常
> - 宿主机 Vulkan ICD（`libGLX_nvidia.so.0`）需要 Xvfb 才能 headless 初始化；Docker 镜像的 headless driver 不需要
> - Docker 将运行环境固化为镜像，避免每次配环境

---

## 2. 宿主机前置条件

```
NVIDIA GPU driver          >= 570.x（已验证 570.195.03）
Docker Engine              >= 24.x
NVIDIA Container Toolkit   已安装且配置为 Docker runtime
```

### 2.1 安装 Docker

```bash
# 设置代理（如需）
export http_proxy=http://100.67.186.3:8089
export https_proxy=http://100.67.186.3:8089

# 添加 Docker 官方源
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io
```

### 2.2 安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### 2.3 配置 Docker daemon 代理（如需外网拉镜像）

```bash
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/proxy.conf << 'EOF'
[Service]
Environment="HTTP_PROXY=http://100.67.186.3:8089"
Environment="HTTPS_PROXY=http://100.67.186.3:8089"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF
systemctl daemon-reload && systemctl restart docker
```

### 2.4 验证 GPU 对 Docker 可见

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# 应输出 8 张 RTX 4090
```

---

## 3. 在基础镜像上做了哪些更改

基础镜像 `nvcr.io/nvidia/isaac-lab:2.3.1` 有三个问题需要修复：

### 3.1 删除 `omni.isaac.ml_archive` 中捆绑的旧版 torch/torchvision

**问题**：镜像内 `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/` 包含旧版 torch 和 torchvision，通过 `setup_python_env.sh` 加入 `PYTHONPATH`，与新安装的 PyTorch 产生 CUDA 符号冲突：

```
ImportError: .../ml_archive/pip_prebundle/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12:
undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**修复（Dockerfile）**：

```dockerfile
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch* \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torchvision*
```

**修复（entrypoint.sh 运行时）**：从 `PYTHONPATH` 中移除含 `ml_archive` 的路径：

```bash
source /isaac-sim/setup_python_env.sh
PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'ml_archive' | tr '\n' ':')
export PYTHONPATH
```

### 3.2 安装 PyTorch

**问题**：基础镜像不包含 PyTorch（它是 Isaac Sim 的 streaming/app 镜像，不是 RL 训练镜像）。

**修复**：

```dockerfile
RUN ${PYTHON} -m pip install --timeout 600 --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

安装后版本：`torch 2.6.0+cu124`，`torchvision 0.21.0+cu124`。

> 注：`isaaclab 0.48.0` 声明 `requires torch>=2.7`，实测 2.6 可正常训练。如需严格匹配可改为 `--index-url https://download.pytorch.org/whl/cu126`。

### 3.3 替换 rsl_rl

**问题**：基础镜像预装 `rsl-rl-lib 3.0.1`，其 `OnPolicyRunner.__init__` 要求配置中包含 `obs_groups` 字段，WBC-AGILE 的训练配置没有该字段：

```
File ".../rsl_rl/runners/on_policy_runner.py", line 44, in __init__
    self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)
KeyError: 'obs_groups'
```

WBC-AGILE 自带修改版 rsl_rl v2.3.3（位于 `agile/algorithms/rsl_rl/`），包含自定义的 `OnPolicyRunner`、`StudentTeacher` 等模块。

**修复**：

```dockerfile
# 卸载系统版
RUN ${PYTHON} -m pip uninstall -y rsl-rl-lib

# 安装 WBC-AGILE 的 fork
RUN cd /workspace/WBC-AGILE/agile/algorithms/rsl_rl && \
    ${PYTHON} -m pip install --no-deps --no-build-isolation -e .
```

### 3.4 安装 WBC-AGILE 及依赖

```dockerfile
# agile 包本身（不拉依赖，避免重新下载 torch）
RUN ${PYTHON} -m pip install --no-deps --no-build-isolation -e .

# 缺失的小依赖
RUN ${PYTHON} -m pip install --timeout 300 --no-cache-dir \
    toml wandb==0.22.2 protobuf==3.20.3 tqdm pyyaml "gym>=0.21.0" tensordict==0.8.3
```

> `protobuf==3.20.3` 是 AGILE 的要求，与 Isaac Lab 的 `>=4.25.8` 冲突，但运行时不影响 RL 训练。

### 3.5 设置环境变量

基础镜像的默认 `ENTRYPOINT` 是 `/isaac-sim/runheadless.sh`（启动 streaming server），需替换为训练入口。以下环境变量必须设置，否则 `AppLauncher` 会报 `KeyError`：

```dockerfile
ENV CARB_APP_PATH=/isaac-sim/kit       # SimulationApp.__init__ 需要
ENV EXP_PATH=/isaac-sim/apps           # AppLauncher._resolve_experience_file 需要
ENV ISAAC_PATH=/isaac-sim              # isaacsim bootstrap 需要
ENV ISAACLAB_PATH=/workspace/isaaclab  # Isaac Lab 包路径
```

---

## 4. 最终镜像内路径一览

```
/isaac-sim/                              # Isaac Sim 5.1.0 安装根目录
├── kit/                                 # Kit kernel
│   └── python/bin/python3.11            # Python 解释器
├── apps/                                # Experience (.kit) 文件
├── exts/                                # Isaac Sim 扩展
│   └── omni.isaac.ml_archive/
│       └── pip_prebundle/               # ← 已删除旧 torch/torchvision/nvidia
├── setup_python_env.sh                  # ← entrypoint 中 source 此文件
└── VERSION                              # "5.1.0-rc.19..."

/workspace/
├── isaaclab/                            # Isaac Lab 2.3.1
│   └── apps/
│       └── isaaclab.python.headless.kit # ← 设置 S3 asset root = Isaac/5.1
└── WBC-AGILE/                           # WBC-AGILE 代码
    ├── scripts/train.py                 # 训练入口
    ├── agile/
    │   └── algorithms/rsl_rl/           # ← 自定义 rsl_rl v2.3.3（已 pip install -e）
    ├── logs/                            # ← 挂载出来的训练日志
    └── docker/
        ├── Dockerfile
        └── entrypoint.sh
```

---

## 5. 构建镜像

```bash
cd /root/WBC-AGILE

docker build --network=host \
  --build-arg http_proxy=http://100.67.186.3:8089 \
  --build-arg https_proxy=http://100.67.186.3:8089 \
  -t wbc-agile:latest \
  -f docker/Dockerfile .
```

构建时间约 5 分钟（主要耗时在下载 PyTorch ~2GB）。最终镜像约 35GB。

---

## 6. 运行训练

### 6.1 默认 8 GPU 分布式训练

使用 `torchrun` 启动分布式训练，8 张 GPU 间通过 NCCL 进行梯度同步（all_reduce），只有 rank 0 写 wandb/tensorboard 日志。

```bash
docker run -d --gpus all --network=host \
  --shm-size=16g \
  --name wbc-agile-train \
  -e http_proxy=http://100.67.186.3:8089 \
  -e https_proxy=http://100.67.186.3:8089 \
  -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
  -v $(pwd)/outputs:/workspace/WBC-AGILE/outputs \
  wbc-agile:latest
```

> **`--shm-size=16g` 是必须的**，NCCL 使用 `/dev/shm` 进行 GPU 间通信，Docker 默认 64MB 不够。

默认配置：
- 任务：`Velocity-G1-History-v0`
- 每 GPU 2048 环境，共 16384 环境
- 3000 iterations
- seed 42-49（torchrun 自动按 rank 偏移）
- 训练模式：`torchrun`（NCCL 梯度同步）

通过环境变量自定义：

```bash
docker run -d --gpus all --network=host \
  --shm-size=16g \
  -e TASK=Velocity-T1-v0 \
  -e NUM_ENVS=1024 \
  -e MAX_ITERATIONS=5000 \
  -e NUM_GPUS=4 \
  -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
  wbc-agile:latest
```

启用 W&B 日志：

```bash
docker run -d --gpus all --network=host \
  --shm-size=16g \
  -e WANDB_API_KEY=your_key_here \
  -e LOGGER=wandb \
  -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
  wbc-agile:latest
```

### 6.2 单 GPU 快速测试

```bash
docker run --rm --gpus '"device=0"' --network=host \
  wbc-agile:latest \
  --task Velocity-G1-History-v0 --num_envs 64 --max_iterations 5 --device cuda:0
```

### 6.3 交互式进入容器

```bash
docker run --rm -it --gpus all --network=host \
  wbc-agile:latest bash
```

---

## 7. 监控训练

```bash
# 查看容器状态
docker ps --filter name=wbc-agile-train

# 实时查看某 GPU 日志
docker exec wbc-agile-train tail -f logs/gpu0.log

# 查看所有 GPU 最新进度
docker exec wbc-agile-train tail -1 logs/gpu*.log

# GPU 显存和利用率
docker exec wbc-agile-train nvidia-smi

# 容器主进程日志（看各 GPU 启动/完成状态）
docker logs -f wbc-agile-train

# 停止训练
docker stop wbc-agile-train
docker rm wbc-agile-train
```

---

## 8. 预期性能

测试环境：8 x NVIDIA RTX 4090, Intel Xeon Platinum 8463B, 1TB RAM

| 配置 | 值 |
|------|---|
| 每 GPU 显存占用 | ~8.1 GB / 24.5 GB |
| GPU 利用率 | ~64% |
| 每 iteration 时间 | ~2.6s（2048 envs/GPU） |
| 3000 iterations 总时间 | ~2.3 小时 |
| 每 iteration timesteps | 49152（2048 envs × 24 steps） |
| 8 GPU 总 timesteps | ~1.18 亿 |

---

## 9. 常见问题排查

### Q1: `KeyError: 'CARB_APP_PATH'`

原因：环境变量未设置。确认 Dockerfile 中有：

```dockerfile
ENV CARB_APP_PATH=/isaac-sim/kit
```

或者在 `docker run` 时传入 `-e CARB_APP_PATH=/isaac-sim/kit`。

### Q2: `KeyError: 'EXP_PATH'`

原因：同上。确认 `ENV EXP_PATH=/isaac-sim/apps`。

### Q3: `ImportError: undefined symbol: __nvJitLinkCreate_12_8`

原因：`omni.isaac.ml_archive/pip_prebundle` 中的旧 torch 未删除，或 `PYTHONPATH` 中仍包含 `ml_archive` 路径。

排查：

```bash
docker exec <container> bash -c 'echo $PYTHONPATH | tr ":" "\n" | grep ml_archive'
# 如果有输出，说明 entrypoint.sh 中的 PYTHONPATH 清理未生效
```

### Q4: `KeyError: 'obs_groups'`

原因：系统版 rsl_rl 3.0.1 未卸载，优先于 WBC-AGILE 的 fork 被加载。

排查：

```bash
docker exec <container> /isaac-sim/kit/python/bin/python3.11 -c \
  "import rsl_rl; print(rsl_rl.__version__, rsl_rl.__file__)"
# 应输出 2.3.3 和 /workspace/WBC-AGILE/agile/algorithms/rsl_rl/...
```

### Q5: `ModuleNotFoundError: No module named 'isaacsim'`

原因：使用了 `--entrypoint bash` 但未 `source /isaac-sim/setup_python_env.sh`。

### Q6: NCCL 报 `Error while creating shared memory segment /dev/shm/nccl-...`

原因：Docker 默认 `/dev/shm` 只有 64MB，不够 NCCL GPU 间通信使用。

修复：运行时加 `--shm-size=16g`。

### Q7: 如何使用 W&B 记录训练

分布式训练下只有 rank 0 写 wandb，所以只会产生一个 run。

```bash
docker run -d --gpus all --network=host \
  --shm-size=16g \
  -e WANDB_API_KEY=your_key_here \
  -e LOGGER=wandb \
  -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
  wbc-agile:latest
```
