# Qwen3-VL-8B TensorRT Edge-LLM 部署与 Benchmark 复现流程

本仓库用于记录 `Qwen3-VL-8B-Instruct-INT4AWQ` 在 **NVIDIA TensorRT Edge-LLM** 上的部署、量化导出和 benchmark 执行流程。

仓库目标不是保存最终实验结果，而是保存后续复现实验所需的核心执行文件，包括：

- x86 / RTX 4090 端量化与 ONNX 导出脚本；
- Jetson AGX Orin 端 benchmark 执行脚本；
- INT4 AWQ 量化所需的 Robo2VLM 校准集；
- text-only / vision VQA benchmark 数据集；
- 已构造好的 Edge-LLM 输入 JSON 和 manifest；
- 与执行流程相关的说明文档。

本仓库不上传模型权重、ONNX、TensorRT engine、TensorRT Edge-LLM 官方源码，也不上传已经跑出来的推理结果、profile、summary、quality evaluation、final record 或原始日志。

---

## 1. 仓库结构

推荐最终上传结构如下：

```text
Qwen3-VL-8B-TRT/
├── data/
│   ├── calib_robo2vlm/
│   │   ├── train.jsonl
│   │   ├── README_calib.json
│   │   └── calib_distribution_report.txt
│   │
│   └── benchmark/
│       ├── robot_short_prompts_v3_mixed_lengths.json
│       └── vision_arena_robot_like/
│           ├── robot_extended40_hybrid50_images/
│           ├── vision_arena_robot_hybrid50.json
│           └── vision_arena_robot_hybrid50.local.json
│
├── examples/
│   └── edgellm_inputs/
│       ├── input_robot_text50_repeat3_max50.json
│       ├── manifest_robot_text50_repeat3_max50.jsonl
│       ├── input_vision_hybrid50_repeat3_max720.json
│       └── manifest_vision_hybrid50_repeat3_max720.jsonl
│
├── scripts/
│   ├── x86/
│   │   ├── 00_start_export_container.sh
│   │   ├── 01_install_edgellm_python_cli.sh
│   │   ├── 02_prepare_local_calib.sh
│   │   ├── 03_quantize_int4_awq.sh
│   │   ├── 04_export_onnx.sh
│   │   ├── 05_check_onnx.py
│   │   └── 06_pack_onnx_artifacts.sh
│   │
│   └── benchmark/
│       ├── make_edgellm_inputs.py
│       ├── run_edgellm_bench_with_tegrastats.sh
│       ├── check_edgellm_vision_io.py
│       ├── summarize_edgellm_benchmark.py
│       └── parse_tegrastats_basic.py
│
├── docs/
│   └── 01_x86_quantize_export_onnx.md
│
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

其中：

- `data/`：保存校准集和 benchmark 数据集；
- `examples/edgellm_inputs/`：保存一份已构造好的 Edge-LLM 输入文件，可直接用于运行；
- `scripts/x86/`：保存 x86 端量化与 ONNX 导出脚本；
- `scripts/benchmark/`：保存 Orin 端 benchmark 执行与基础解析脚本；
- `docs/`：保存更详细的 x86 导出流程说明。

---

## 2. 实验环境

### 2.1 x86 导出端

x86 端用于模型量化和 ONNX 导出。

| 项目 | 内容 |
|---|---|
| 设备 | x86 主机 / RTX 4090 |
| 容器 | `nvcr.io/nvidia/pytorch:26.05-py3` |
| 工具 | TensorRT Edge-LLM Python CLI |
| 任务 | INT4 AWQ 量化、ONNX 导出、ONNX 检查与打包 |

### 2.2 Jetson 推理端

Jetson AGX Orin 端用于 TensorRT Edge-LLM C++ Runtime benchmark。

| 项目 | 内容 |
|---|---|
| 设备 | Jetson AGX Orin 64GB |
| JetPack | JetPack 7.2 |
| Jetson Linux | R39.2 |
| CUDA | 13.2 |
| 推理框架 | NVIDIA TensorRT Edge-LLM |
| 推理方式 | C++ Runtime |
| 被测模型 | Qwen3-VL-8B-Instruct-INT4AWQ |
| 测试任务 | text-only / vision VQA |
| 资源监控 | tegrastats |

---

## 3. 总体复现流程

完整流程分为两部分。

第一部分是在 x86 / RTX 4090 主机上完成模型转换：

```text
原始 Qwen3-VL-8B-Instruct 模型
        ↓
Robo2VLM 校准集
        ↓
INT4 AWQ 量化
        ↓
TensorRT Edge-LLM ONNX 导出
        ↓
ONNX 检查与打包
```

第二部分是在 Jetson AGX Orin 上运行 benchmark：

```text
准备 TensorRT Edge-LLM C++ Runtime
        ↓
准备 LLM engine 和 multimodal engine
        ↓
准备 benchmark 仓库和数据集
        ↓
构造或检查 Edge-LLM input JSON
        ↓
运行 text-only benchmark
        ↓
运行 vision VQA benchmark
        ↓
解析 profile / tegrastats
```

---

## 4. Step 1：准备模型和 TensorRT Edge-LLM

本仓库不包含模型权重、ONNX、TensorRT engine 和 TensorRT Edge-LLM 官方源码。

需要在本地或实验设备上单独准备：

```text
TensorRT-Edge-LLM 官方源码
Qwen3-VL-8B-Instruct 原始模型
Qwen3-VL-8B-Instruct-INT4AWQ 量化产物
TensorRT Edge-LLM ONNX 导出产物
TensorRT engine
```

x86 端建议使用类似目录组织：

```text
/mnt/data/pc/edgellm_x86/workspace
/mnt/data/pc/edgellm_x86/models
/mnt/data/pc/edgellm_x86/hf_cache
/mnt/data/pc/edgellm_x86/logs
```

Orin 端建议使用如下目录组织：

```text
/mnt/ssd/TensorRT-Edge-LLM
/mnt/ssd/edgellm_models/Qwen3-VL-8B-Instruct-INT4AWQ
/mnt/ssd/qwen3vl8b_edgellm_benchmark
```

如果脚本中的默认路径与当前设备不同，需要在执行前修改脚本顶部的路径变量，例如：

```text
MODEL_DIR
OUTPUT_DIR
CALIB_DIR
ONNX_DIR
ENGINE_DIR
BENCH_ROOT
```

---

## 5. Step 2：x86 端启动导出容器

x86 端脚本位于：

```text
scripts/x86/
```

各脚本的执行位置如下：

| 脚本 | 执行位置 | 作用 |
|---|---|---|
| `00_start_export_container.sh` | x86 宿主机 | 启动导出容器 |
| `01_install_edgellm_python_cli.sh` | 容器内 | 安装 TensorRT Edge-LLM Python CLI |
| `02_prepare_local_calib.sh` | x86 宿主机 | 将校准集复制到容器 |
| `03_quantize_int4_awq.sh` | 容器内 | 执行 INT4 AWQ 量化 |
| `04_export_onnx.sh` | 容器内 | 导出 ONNX |
| `05_check_onnx.py` | 容器内 | 检查 ONNX |
| `06_pack_onnx_artifacts.sh` | 容器内或宿主机 | 打包 ONNX 产物 |

首先在 x86 宿主机启动导出容器：

```bash
bash scripts/x86/00_start_export_container.sh
```

该脚本用于启动 NVIDIA PyTorch 容器，并挂载 workspace、HuggingFace cache、logs 和 models 目录。

---

## 6. Step 3：安装 TensorRT Edge-LLM Python CLI

进入容器后执行：

```bash
bash scripts/x86/01_install_edgellm_python_cli.sh
```

该脚本用于安装 TensorRT Edge-LLM Python 工具链，并检查以下命令是否可用：

```bash
which tensorrt-edgellm-quantize
which tensorrt-edgellm-export
```

---

## 7. Step 4：准备 Robo2VLM 校准集

本仓库提供完整 Robo2VLM JSONL 校准集：

```text
data/calib_robo2vlm/train.jsonl
```

校准集说明文件：

```text
data/calib_robo2vlm/README_calib.json
```

校准集统计信息：

```text
data/calib_robo2vlm/calib_distribution_report.txt
```

在 x86 宿主机执行：

```bash
bash scripts/x86/02_prepare_local_calib.sh
```

该脚本会将仓库中的校准集复制到导出容器中：

```text
/tmp/calib_robo2vlm/train.jsonl
```

---

## 8. Step 5：INT4 AWQ 量化

在容器内执行：

```bash
bash scripts/x86/03_quantize_int4_awq.sh
```

该脚本调用 TensorRT Edge-LLM 量化命令：

```text
tensorrt-edgellm-quantize
```

完成 Qwen3-VL-8B 的 INT4 AWQ 量化。

---

## 9. Step 6：导出 ONNX

在容器内执行：

```bash
bash scripts/x86/04_export_onnx.sh
```

该脚本调用 TensorRT Edge-LLM 导出命令：

```text
tensorrt-edgellm-export
```

将量化后的 checkpoint 导出为 TensorRT Edge-LLM ONNX 中间格式。

---

## 10. Step 7：检查与打包 ONNX

检查导出的 ONNX：

```bash
python scripts/x86/05_check_onnx.py
```

打包 ONNX 产物：

```bash
bash scripts/x86/06_pack_onnx_artifacts.sh
```

注意：ONNX 文件和 ONNX 压缩包不上传 GitHub，需要单独保存或传输到 Orin。

---

## 11. Step 8：Orin 端构建 TensorRT Edge-LLM Runtime 和 Engine

如果 Jetson AGX Orin 重新刷机，之前编译好的 TensorRT Edge-LLM C++ Runtime、plugin、LLM engine 和 multimodal engine 都可能不存在。因此需要按照下面流程重新恢复执行环境。

整体流程如下：

```text
x86 端导出 ONNX
        ↓
将 ONNX 拷贝到 Orin
        ↓
Orin 上编译 TensorRT Edge-LLM C++ Runtime
        ↓
Orin 上构建 TensorRT engine
        ↓
使用 llm_inference 运行正式 benchmark input JSON
```

需要注意：

- ONNX 文件可以从 x86 端传输到 Orin；
- TensorRT engine 与目标硬件强绑定，建议在最终运行推理的 Orin 上构建；
- Qwen3-VL 是 VLM 模型，即使只运行 text-only benchmark，也需要准备 multimodal engine，并在推理时传入 `--multimodalEngineDir`。

---

### 11.1 检查 Orin 系统环境

在 Orin 上执行：

```bash
cat /etc/nv_tegra_release
uname -a
nvcc --version
df -h
free -h
```

本项目期望环境大致为：

```text
Jetson AGX Orin 64GB
JetPack 7.2
Jetson Linux R39.2
CUDA 13.2
TensorRT 10.x+
```

如果刷机后 `/mnt/ssd` 没有挂载，先检查 SSD：

```bash
lsblk -f
df -h
```

如果 SSD 分区存在但没有挂载，可以按实际设备名挂载，例如：

```bash
sudo mkdir -p /mnt/ssd
sudo mount /dev/nvme0n1p1 /mnt/ssd
df -h /mnt/ssd
```

---

### 11.2 安装 Orin 端构建依赖

在 Orin 上安装 C++ Runtime 编译所需依赖：

```bash
sudo apt update
sudo apt install -y cmake build-essential git \
    cuda-toolkit-13-2 \
    libnvinfer-headers-dev libnvinfer-dev libnvonnxparsers-dev

export PATH=/usr/local/cuda/bin:$PATH
```

检查 CUDA 和 JetPack 版本：

```bash
nvcc --version
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core cuda-toolkit-13-2
```

注意：Jetson 上建议使用 NVIDIA 仓库中的 `cuda-toolkit-*` 包，不要安装 Ubuntu 源中的 `nvidia-cuda-toolkit`，避免和 JetPack 自带 CUDA 库冲突。

---

### 11.3 下载或恢复 TensorRT-Edge-LLM 官方源码

本仓库不上传 TensorRT-Edge-LLM 官方源码。刷机后需要重新 clone 官方源码，或者从之前备份恢复。

建议源码目录为：

```text
/mnt/ssd/TensorRT-Edge-LLM
```

重新 clone：

```bash
cd /mnt/ssd
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

如果之前通过 Docker 或 root 用户操作过源码目录，建议修复权限：

```bash
sudo chown -R $(whoami):$(whoami) /mnt/ssd/TensorRT-Edge-LLM
```

---

### 11.4 编译 TensorRT Edge-LLM C++ Runtime

进入源码目录：

```bash
cd /mnt/ssd/TensorRT-Edge-LLM
rm -rf build
mkdir build
cd build
```

针对 Jetson Orin 使用如下 CMake 配置：

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

开始编译：

```bash
make -j$(nproc)
```

如果编译时内存压力较大，可以降低并行度：

```bash
make -j4
```

---

### 11.5 检查 C++ Runtime 和 plugin

回到源码根目录：

```bash
cd /mnt/ssd/TensorRT-Edge-LLM
```

检查 `llm_build`、`llm_inference` 和 plugin：

```bash
ls -lh build/examples/llm/llm_build
ls -lh build/examples/llm/llm_inference
ls -lh build/libNvInfer_edgellm_plugin.so
```

对于 Qwen3-VL 这类视觉语言模型，还需要检查 visual engine 构建工具：

```bash
ls -lh build/examples/multimodal/visual_build
```

如果这些文件不存在，说明 C++ Runtime、plugin 或 multimodal 构建工具没有编译成功，需要回到前面的 CMake / make 步骤排查。

---

### 11.6 设置 TensorRT Edge-LLM 环境变量

建议统一设置以下变量：

```bash
export EDGELLM_ROOT=/mnt/ssd/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so

export MODEL_NAME=Qwen3-VL-8B-Instruct-INT4AWQ
export MODEL_ROOT=/mnt/ssd/edgellm_models/$MODEL_NAME

export ONNX_ROOT=$MODEL_ROOT/onnx
export ENGINE_ROOT=$MODEL_ROOT/engine
export BENCH_ROOT=/mnt/ssd/qwen3vl8b_edgellm_benchmark
```

检查 plugin：

```bash
ls -lh $EDGELLM_PLUGIN_PATH
```

为了避免每次重新设置，可以写入 `~/.bashrc`：

```bash
cat >> ~/.bashrc << 'EOF'

# TensorRT Edge-LLM
export EDGELLM_ROOT=/mnt/ssd/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so
EOF

source ~/.bashrc
```

---

### 11.7 准备 x86 端导出的 ONNX 产物

x86 端完成 INT4 AWQ 量化和 ONNX 导出后，需要将 ONNX 产物传到 Orin。

建议 Orin 上模型目录为：

```text
/mnt/ssd/edgellm_models/Qwen3-VL-8B-Instruct-INT4AWQ/
```

建议目录结构为：

```text
/mnt/ssd/edgellm_models/Qwen3-VL-8B-Instruct-INT4AWQ/
├── onnx/
│   ├── llm/
│   └── visual/
└── engine/
```

如果 ONNX 产物在 x86 主机上，可以从 x86 主机执行：

```bash
scp -r /path/to/Qwen3-VL-8B-Instruct-INT4AWQ/onnx \
    nv@<ORIN_IP>:/mnt/ssd/edgellm_models/Qwen3-VL-8B-Instruct-INT4AWQ/
```

拷贝后在 Orin 上检查：

```bash
ls -lah $ONNX_ROOT
find $ONNX_ROOT -maxdepth 3 -type f | head
find $ONNX_ROOT -maxdepth 3 -type d
```

注意：本仓库不上传 ONNX 文件。ONNX 需要单独保存，并在构建 engine 前传输到 Orin。

---

### 11.8 构建 LLM engine

在 Orin 上构建 LLM engine。

推荐先清理缓存，降低内存压力：

```bash
sudo sysctl -w vm.drop_caches=3
```

创建 engine 输出目录：

```bash
mkdir -p $ENGINE_ROOT/llm_unified_4096_8192
```

构建 LLM engine：

```bash
cd $EDGELLM_ROOT

./build/examples/llm/llm_build \
    --onnxDir $ONNX_ROOT/llm \
    --engineDir $ENGINE_ROOT/llm_unified_4096_8192 \
    --maxBatchSize 1 \
    --maxInputLen 4096 \
    --maxKVCacheCapacity 8192
```

如果 engine 构建时显存或统一内存不足，可以降低上下文配置：

```bash
./build/examples/llm/llm_build \
    --onnxDir $ONNX_ROOT/llm \
    --engineDir $ENGINE_ROOT/llm_unified_2048_4096 \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096
```

构建完成后检查：

```bash
ls -lah $ENGINE_ROOT/llm_unified_4096_8192
```

如果你使用的是降低后的配置，则检查对应目录：

```bash
ls -lah $ENGINE_ROOT/llm_unified_2048_4096
```

---

### 11.9 构建 visual / multimodal engine

Qwen3-VL 是视觉语言模型，除了 LLM engine，还需要构建 visual engine。

创建 engine 目录：

```bash
mkdir -p $ENGINE_ROOT
```

构建 visual engine：

```bash
cd $EDGELLM_ROOT

./build/examples/multimodal/visual_build \
    --onnxDir $ONNX_ROOT/visual \
    --engineDir $ENGINE_ROOT
```

构建完成后检查：

```bash
ls -lah $ENGINE_ROOT
find $ENGINE_ROOT -maxdepth 3 -type f | head
```

如果你的 ONNX 目录不是 `visual/`，而是其他名称，例如 `vision/`、`multimodal/`，需要以实际导出目录为准：

```bash
find $ONNX_ROOT -maxdepth 3 -type d
```

然后把 `--onnxDir` 改成实际视觉 ONNX 目录。

---

### 11.10 在 Orin 上准备 benchmark 仓库

在 Orin 上将本仓库 clone 到 `BENCH_ROOT` 对应目录：

```bash
cd /mnt/ssd
git clone https://github.com/WSH-WX/Qwen3-VL-8B-TRT.git qwen3vl8b_edgellm_benchmark
cd /mnt/ssd/qwen3vl8b_edgellm_benchmark
```

如果仓库已经存在，可以更新：

```bash
cd /mnt/ssd/qwen3vl8b_edgellm_benchmark
git pull
```

设置：

```bash
export BENCH_ROOT=/mnt/ssd/qwen3vl8b_edgellm_benchmark
```

检查 benchmark 数据：

```bash
ls -lah $BENCH_ROOT/data/benchmark
ls -lah $BENCH_ROOT/data/benchmark/vision_arena_robot_like
ls -lah $BENCH_ROOT/examples/edgellm_inputs
```

注意：`vision_arena_robot_hybrid50.local.json` 和已生成的 vision input JSON 中可能包含图片绝对路径。若仓库没有 clone 到 `/mnt/ssd/qwen3vl8b_edgellm_benchmark`，需要重新生成 `.local.json` 或重新生成 Edge-LLM input JSON。

---

### 11.11 最终检查 Runtime、plugin、engine 和 benchmark 仓库

运行正式 benchmark 前，至少确认以下文件或目录存在：

```bash
ls -lh $EDGELLM_ROOT/build/examples/llm/llm_inference
ls -lh $EDGELLM_ROOT/build/examples/llm/llm_build
ls -lh $EDGELLM_ROOT/build/examples/multimodal/visual_build
ls -lh $EDGELLM_PLUGIN_PATH

ls -lah $ONNX_ROOT
ls -lah $ENGINE_ROOT
ls -lah $ENGINE_ROOT/llm_unified_4096_8192

ls -lah $BENCH_ROOT/data/benchmark
ls -lah $BENCH_ROOT/examples/edgellm_inputs
```

也可以一次性检查：

```bash
test -x $EDGELLM_ROOT/build/examples/llm/llm_inference && echo "[OK] llm_inference"
test -x $EDGELLM_ROOT/build/examples/llm/llm_build && echo "[OK] llm_build"
test -x $EDGELLM_ROOT/build/examples/multimodal/visual_build && echo "[OK] visual_build"
test -f $EDGELLM_PLUGIN_PATH && echo "[OK] plugin"
test -d $ONNX_ROOT && echo "[OK] ONNX root"
test -d $ENGINE_ROOT && echo "[OK] engine root"
test -d $ENGINE_ROOT/llm_unified_4096_8192 && echo "[OK] LLM engine"
test -d $BENCH_ROOT/data/benchmark && echo "[OK] benchmark dataset"
test -d $BENCH_ROOT/examples/edgellm_inputs && echo "[OK] Edge-LLM inputs"
```

如果这些检查都通过，就可以继续执行正式 text-only 或 vision VQA benchmark。

---

### 11.12 刷机后恢复顺序总结

如果 Orin 完全重新刷机，可以按下面顺序恢复：

```text
1. 挂载 SSD
2. 安装 cmake、build-essential、git、cuda-toolkit-13-2、TensorRT dev 包
3. clone TensorRT-Edge-LLM 官方源码
4. git submodule update --init --recursive
5. cmake 配置 jetson-orin
6. make 编译 TensorRT Edge-LLM C++ Runtime
7. 设置 EDGELLM_PLUGIN_PATH
8. 从 x86 拷贝 ONNX 到 Orin
9. 使用 llm_build 构建 LLM engine
10. 使用 visual_build 构建 visual / multimodal engine
11. clone 本仓库到 BENCH_ROOT
12. 检查 Runtime、plugin、ONNX、engine、数据集和 input JSON
13. 运行正式 text-only / vision VQA benchmark
```

---

## 12. Step 9：准备 Benchmark 数据集

本仓库提供 benchmark 数据集：

```text
data/benchmark/
```

text-only 数据集：

```text
data/benchmark/robot_short_prompts_v3_mixed_lengths.json
```

vision VQA 数据集：

```text
data/benchmark/vision_arena_robot_like/
```

建议保留的 vision VQA 执行文件包括：

```text
vision_arena_robot_hybrid50.json
vision_arena_robot_hybrid50.local.json
robot_extended40_hybrid50_images/
```

其中：

- `vision_arena_robot_hybrid50.json` 是原始 vision VQA 样本；
- `vision_arena_robot_hybrid50.local.json` 是包含本地图片路径的执行版本；
- `robot_extended40_hybrid50_images/` 是 vision VQA 图片目录。

如果仓库 clone 到默认位置：

```text
/mnt/ssd/qwen3vl8b_edgellm_benchmark
```

通常可以直接使用仓库中的 `.local.json` 和 `examples/edgellm_inputs/`。

如果仓库 clone 到其他目录，或者图片路径发生变化，需要重新生成 `.local.json` 或重新生成 Edge-LLM input JSON。

---

## 13. Step 10：构造 Edge-LLM 输入 JSON

TensorRT Edge-LLM C++ Runtime 使用 JSON 文件作为输入。  
因此需要将原始 benchmark 数据转换为 Edge-LLM 可读格式。

输入构造脚本：

```text
scripts/benchmark/make_edgellm_inputs.py
```

该脚本负责：

1. 读取 text-only 或 vision VQA 数据集；
2. 构造 Edge-LLM messages 格式；
3. 生成重复测试请求；
4. 设置 `maxGenerateLength`、`temperature`、`top_p`、`top_k`；
5. 输出 input JSON；
6. 输出 manifest JSONL，记录每个请求与原始样本的对应关系。

本仓库已经提供一份生成好的输入文件：

```text
examples/edgellm_inputs/input_robot_text50_repeat3_max50.json
examples/edgellm_inputs/manifest_robot_text50_repeat3_max50.jsonl
examples/edgellm_inputs/input_vision_hybrid50_repeat3_max720.json
examples/edgellm_inputs/manifest_vision_hybrid50_repeat3_max720.jsonl
```

这些文件不是实验结果，而是 benchmark 执行输入。  
在默认路径下可以直接使用；如果 benchmark 数据路径、图片路径或生成参数发生变化，应重新生成。

先查看脚本参数：

```bash
cd $BENCH_ROOT
python scripts/benchmark/make_edgellm_inputs.py --help
```

重新生成 text-only input JSON：

```bash
python scripts/benchmark/make_edgellm_inputs.py \
    --mode text \
    --dataset data/benchmark/robot_short_prompts_v3_mixed_lengths.json \
    --output examples/edgellm_inputs/input_robot_text50_repeat3_max50.json \
    --manifest examples/edgellm_inputs/manifest_robot_text50_repeat3_max50.jsonl \
    --repeat 3 \
    --max-generate-length 50
```

重新生成 vision VQA input JSON：

```bash
python scripts/benchmark/make_edgellm_inputs.py \
    --mode vision \
    --dataset data/benchmark/vision_arena_robot_like/vision_arena_robot_hybrid50.local.json \
    --output examples/edgellm_inputs/input_vision_hybrid50_repeat3_max720.json \
    --manifest examples/edgellm_inputs/manifest_vision_hybrid50_repeat3_max720.jsonl \
    --repeat 3 \
    --max-generate-length 720
```

如果当前脚本参数名和上述示例不同，以 `--help` 输出为准。

---

## 14. Step 11：检查 vision 输入与图片路径

运行 vision VQA 前，建议先检查 input JSON、manifest 和图片路径是否一致。

检查脚本：

```text
scripts/benchmark/check_edgellm_vision_io.py
```

先查看脚本参数：

```bash
cd $BENCH_ROOT
python scripts/benchmark/check_edgellm_vision_io.py --help
```

如果脚本支持显式传参，可以按当前路径检查：

```bash
python scripts/benchmark/check_edgellm_vision_io.py \
    --input examples/edgellm_inputs/input_vision_hybrid50_repeat3_max720.json \
    --manifest examples/edgellm_inputs/manifest_vision_hybrid50_repeat3_max720.jsonl
```

该脚本用于检查：

```text
图片路径是否存在
input JSON 是否可读
manifest JSONL 是否可读
样本数量是否一致
vision 输入是否包含 image + text
```

如果脚本参数与上述示例不同，以 `--help` 输出为准。

---

## 15. Step 12：运行 text-only Benchmark

text-only benchmark 输入文件：

```text
examples/edgellm_inputs/input_robot_text50_repeat3_max50.json
```

对应 manifest：

```text
examples/edgellm_inputs/manifest_robot_text50_repeat3_max50.jsonl
```

推荐通过统一脚本运行：

```text
scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

执行前先检查脚本顶部路径变量：

```bash
cd $BENCH_ROOT
sed -n '1,160p' scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

至少需要确认或修改：

```text
EDGELLM_ROOT
LLM_INFERENCE_BIN
EDGELLM_PLUGIN_PATH
ENGINE_DIR
MULTIMODAL_ENGINE_DIR
INPUT_FILE
OUTPUT_FILE
PROFILE_FILE
RUNTIME_LOG
TEGRASTATS_LOG
```

运行 text-only benchmark：

```bash
bash scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

运行时需要确保：

```text
ENGINE_DIR 指向 LLM engine 目录
MULTIMODAL_ENGINE_DIR 指向 visual / multimodal engine 目录
INPUT_FILE 指向 input_robot_text50_repeat3_max50.json
OUTPUT_FILE 指向本地输出 JSON
PROFILE_FILE 指向本地 profile JSON
RUNTIME_LOG 指向本地 runtime log
TEGRASTATS_LOG 指向本地 tegrastats log
```

本仓库不保存运行生成的 output、profile 和 log。  
这些文件建议只保存在 Orin 本地实验目录中。

---

## 16. Step 13：运行 vision VQA Benchmark

vision VQA benchmark 输入文件：

```text
examples/edgellm_inputs/input_vision_hybrid50_repeat3_max720.json
```

对应 manifest：

```text
examples/edgellm_inputs/manifest_vision_hybrid50_repeat3_max720.jsonl
```

运行方式同样使用：

```text
scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

执行前检查脚本顶部路径变量：

```bash
cd $BENCH_ROOT
sed -n '1,160p' scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

运行 vision VQA benchmark 时需要确保：

```text
ENGINE_DIR 指向 LLM engine 目录
MULTIMODAL_ENGINE_DIR 指向 visual / multimodal engine 目录
INPUT_FILE 指向 input_vision_hybrid50_repeat3_max720.json
OUTPUT_FILE 指向本地输出 JSON
PROFILE_FILE 指向本地 profile JSON
RUNTIME_LOG 指向本地 runtime log
TEGRASTATS_LOG 指向本地 tegrastats log
```

vision 输入中包含 image + text，结构类似：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个简洁、可靠的机器人视觉助手。请只根据图像和用户问题回答。"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "/path/to/image.jpg"
        },
        {
          "type": "text",
          "text": "用户问题"
        }
      ]
    }
  ]
}
```

运行：

```bash
bash scripts/benchmark/run_edgellm_bench_with_tegrastats.sh
```

本仓库不保存运行生成的 vision output、profile 和 log。  
这些文件建议只保存在 Orin 本地实验目录中。

---

## 17. Step 14：解析 profile 和 tegrastats

如果需要复现实验指标，可以在本地运行解析脚本。

profile 汇总脚本：

```text
scripts/benchmark/summarize_edgellm_benchmark.py
```

tegrastats 解析脚本：

```text
scripts/benchmark/parse_tegrastats_basic.py
```

先查看脚本参数：

```bash
cd $BENCH_ROOT
python scripts/benchmark/summarize_edgellm_benchmark.py --help
python scripts/benchmark/parse_tegrastats_basic.py --help
```

根据本地运行生成的 profile 和 tegrastats log 解析指标。

主要解析指标包括：

```text
prefill average time
generation average time per token
decode throughput
vision encoder average time
peak unified memory
RAM used
SWAP used
GR3D_FREQ
CPU temperature
GPU temperature
TJ temperature
```

注意：

- Edge-LLM C++ Runtime 输出的是 aggregate profile；
- Edge-LLM profile 不等同于 vLLM streaming API 的逐请求 TTFT / E2E / P50 / P95；
- output、profile、summary 和 final record 不上传 GitHub，只在本地实验目录保存。

---

## 18. 不上传的文件

以下内容不是复现执行流程的刚需，不上传 GitHub：

```text
runs/
outputs/
profiles/
reports/
quality_eval/
logs/
final_archives/
*.onnx
*.onnx.data
*.engine
*.safetensors
*.bin
*.pt
*.pth
*.tar.gz
*.pyc
__pycache__/
```

---

## 19. 复用建议

后续更换模型或数据集时，主要修改：

```text
模型目录
ONNX 目录
LLM engine 目录
multimodal engine 目录
benchmark 数据集
input JSON 名称
output JSON 名称
profile JSON 名称
```

推荐复用顺序：

```text
1. 准备模型、ONNX 和 TensorRT engine
2. 检查 Orin 环境和 TensorRT Edge-LLM Runtime
3. 检查 plugin、LLM engine、multimodal engine
4. clone benchmark 仓库到 BENCH_ROOT
5. 检查 benchmark 数据集
6. 构造或检查 text / vision input JSON
7. 检查 vision 图片路径
8. 运行 text-only benchmark
9. 运行 vision VQA benchmark
10. 解析 profile 和 tegrastats
11. 本地保存输出结果和日志
```

---

## 20. 常见注意事项

### 20.1 Qwen3-VL text-only 也需要 multimodal engine

即使没有图片输入，Qwen3-VL 仍然是 VLM 模型，TensorRT Edge-LLM C++ Runtime 初始化时也需要 multimodal engine。

因此 text-only 和 vision benchmark 都需要传入：

```bash
--multimodalEngineDir
```

### 20.2 Edge-LLM profile 与 vLLM API 指标口径不同

vLLM API 可以记录逐请求 TTFT / E2E / P50 / P95。

Edge-LLM C++ Runtime 输出的是 aggregate profile，更适合记录：

```text
prefill average time
generation average time per token
decode throughput
vision encoder average time
peak unified memory
```

### 20.3 不直接上传模型和 engine

本仓库只保存复现执行流程所需的脚本、数据集、输入文件和说明文档。

模型、ONNX、TensorRT engine 和官方源码需要单独准备。

### 20.4 结果文件只保留本地

推理输出、profile、summary、quality evaluation、final record、runtime log 和 tegrastats log 只保留在本地实验目录中，不上传 GitHub。
