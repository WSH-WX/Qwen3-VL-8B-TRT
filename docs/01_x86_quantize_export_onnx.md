# x86 / RTX 4090 端：Qwen3-VL-8B INT4 AWQ 量化与 ONNX 导出

本文档记录在 x86 / RTX 4090 主机上，将 `Qwen3-VL-8B-Instruct` 转换为 TensorRT Edge-LLM 可用 ONNX 中间格式的流程。

---

## 1. 目标

x86 端负责：

```text
Qwen3-VL-8B-Instruct 原始模型
        ↓
Robo2VLM JSONL 校准集
        ↓
INT4 AWQ 量化
        ↓
TensorRT Edge-LLM ONNX 导出
        ↓
ONNX 检查
        ↓
ONNX 打包与 SHA256 校验
```

TensorRT engine 不在 x86 上构建，而是在最终运行推理的 Jetson AGX Orin 目标设备上构建。

---

## 2. 使用环境

| 项目 | 内容 |
|---|---|
| 主机 | x86 / RTX 4090 |
| 容器镜像 | `nvcr.io/nvidia/pytorch:26.05-py3` |
| 工具 | TensorRT Edge-LLM Python CLI |
| 作用 | 执行 INT4 AWQ 量化、ONNX 导出和 ONNX 检查 |

---

## 3. 目录规划

建议 x86 端使用类似目录：

```text
/mnt/data/pc/edgellm_x86/
├── workspace/
├── hf_cache/
├── logs/
├── models/
│   ├── Qwen3-VL-8B-Instruct/
│   └── Qwen3-VL-8B-Instruct-EdgeLLM/
└── artifacts/
```

其中：

- `workspace/`：放 TensorRT-Edge-LLM 源码和工作目录；
- `hf_cache/`：HuggingFace 缓存；
- `models/`：原始模型、量化模型和导出产物；
- `logs/`：本地日志；
- `artifacts/`：ONNX 打包结果。

本仓库不上传模型权重、量化 checkpoint、ONNX 文件或压缩包。

---

## 4. 脚本执行位置

| 脚本 | 执行位置 | 作用 |
|---|---|---|
| `00_start_export_container.sh` | x86 宿主机 | 启动导出容器 |
| `01_install_edgellm_python_cli.sh` | 容器内 | 安装 TensorRT Edge-LLM Python CLI |
| `02_prepare_local_calib.sh` | x86 宿主机 | 将校准集复制到容器 |
| `03_quantize_int4_awq.sh` | 容器内 | INT4 AWQ 量化 |
| `04_export_onnx.sh` | 容器内 | 导出 ONNX |
| `05_check_onnx.py` | 容器内 | 检查 ONNX |
| `06_pack_onnx_artifacts.sh` | 容器内或宿主机 | 打包 ONNX 产物 |

---

## 5. 启动导出容器

在 x86 宿主机执行：

```bash
bash scripts/x86/00_start_export_container.sh
```

该脚本用于启动 NVIDIA PyTorch 容器，并挂载 workspace、HuggingFace cache、logs 和 models 目录。

如果当前机器目录与脚本默认路径不同，需要先修改脚本中的挂载路径。

---

## 6. 安装 TensorRT Edge-LLM Python CLI

进入容器后执行：

```bash
bash scripts/x86/01_install_edgellm_python_cli.sh
```

安装完成后检查：

```bash
which tensorrt-edgellm-quantize
which tensorrt-edgellm-export
```

如果命令不存在，需要重新检查 TensorRT Edge-LLM Python 环境是否安装成功。

---

## 7. 准备本地 JSONL 校准集

本仓库上传完整校准集：

```text
data/calib_robo2vlm/train.jsonl
```

校准集说明和统计文件：

```text
data/calib_robo2vlm/README_calib.json
data/calib_robo2vlm/calib_distribution_report.txt
```

在 x86 宿主机执行：

```bash
bash scripts/x86/02_prepare_local_calib.sh
```

该脚本会将仓库中的校准集复制到导出容器中，默认目标路径类似：

```text
/tmp/calib_robo2vlm/train.jsonl
```

如果容器名称或校准集路径不同，需要修改脚本顶部变量。

---

## 8. INT4 AWQ 量化

在容器内执行：

```bash
bash scripts/x86/03_quantize_int4_awq.sh
```

该脚本调用：

```text
tensorrt-edgellm-quantize
```

如果 TensorRT Edge-LLM 版本变化，参数名可能不同，以当前环境中的帮助信息为准：

```bash
tensorrt-edgellm-quantize llm --help
```

---

## 9. 导出 ONNX

在容器内执行：

```bash
bash scripts/x86/04_export_onnx.sh
```

该脚本调用：

```text
tensorrt-edgellm-export
```

如果 CLI 参数变化，先查看：

```bash
tensorrt-edgellm-export --help
```

---

## 10. 检查 ONNX

在容器内执行：

```bash
python3 scripts/x86/05_check_onnx.py
```

大模型 ONNX 通常使用 external data，因此 checker 推荐直接传入路径：

```python
onnx.checker.check_model(str(path))
```

如果检查失败，先确认：

```text
ONNX 文件是否完整
external data 是否在同一目录
当前 Python 环境是否安装 onnx
脚本中的 ONNX 路径是否正确
```

---

## 11. 打包 ONNX 产物

在容器内或 x86 宿主机执行：

```bash
bash scripts/x86/06_pack_onnx_artifacts.sh
```

该脚本会将 ONNX 目录打包并生成 SHA256 校验文件。

生成的 ONNX 压缩包和 SHA256 文件只在本地保存，不上传 GitHub。  
之后需要将 ONNX 产物传输到 Orin，并在 Orin 上构建 TensorRT engine。

---

## 12. 不上传内容

以下内容不要上传到 GitHub：

```text
模型权重
量化 checkpoint
ONNX 文件
ONNX external data
ONNX 压缩包
TensorRT engine
原始 history
代理配置
日志大文件
```

GitHub 只上传：

```text
复现脚本
校准集 JSONL
benchmark 数据集
Edge-LLM input JSON
配置说明
Markdown 文档
```
