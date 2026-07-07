# x86 / RTX 4090 端：Qwen3-VL-8B INT4 AWQ 量化与 ONNX 导出



本文档记录在 x86 RTX 4090 主机上，将 `Qwen3-VL-8B-Instruct` 转换为 TensorRT Edge-LLM 可用 ONNX 中间格式的流程。



## 1. 目标



x86 端负责：



```text

Qwen3-VL-8B-Instruct 原始模型

→ INT4 AWQ 量化

→ TensorRT Edge-LLM ONNX 导出

→ ONNX 检查

→ ONNX 打包与 sha256 校验

```



TensorRT engine 不在 x86 上构建，而是在 Jetson AGX Orin 目标设备上构建。



## 2. 使用环境



| 项目   | 内容                                            |

| ---- | --------------------------------------------- |

| 主机   | x86 / RTX 4090                                |

| 容器镜像 | `nvcr.io/nvidia/pytorch:26.05-py3`            |

| 作用   | 安装 TensorRT Edge-LLM Python CLI，执行量化和 ONNX 导出 |



## 3. 目录规划



```text

/mnt/data/pc/edgellm\_x86/

├── workspace/

├── hf\_cache/

├── logs/

├── models/

│   ├── Qwen3-VL-8B-Instruct/

│   └── Qwen3-VL-8B-Instruct-EdgeLLM/

└── artifacts/

```



## 4. 启动导出容器



```bash

bash scripts/x86/00\_start\_export\_container.sh

```



## 5. 安装 TensorRT Edge-LLM Python CLI



容器内执行：



```bash

bash scripts/x86/01\_install\_edgellm\_python\_cli.sh

```



## 6. 准备本地 JSONL 校准集



x86 宿主机执行：



```bash

bash scripts/x86/02\_prepare\_local\_calib.sh

```



完整校准集不上传 GitHub，仓库中只保留 `examples/calib/train.sample.jsonl` 作为格式示例。



## 7. INT4 AWQ 量化



容器内执行：



```bash

bash scripts/x86/03\_quantize\_int4\_awq.sh

```



如果 TensorRT Edge-LLM 版本变化，参数名可能不同，以当前环境中的帮助信息为准：



```bash

tensorrt-edgellm-quantize llm --help

```



## 8. 导出 ONNX



容器内执行：



```bash

bash scripts/x86/04\_export\_onnx.sh

```



如果 CLI 参数变化，先查看：



```bash

tensorrt-edgellm-export --help

```



## 9. 检查 ONNX



容器内执行：



```bash

python3 scripts/x86/05\_check\_onnx.py

```



大模型 ONNX 通常使用 external data，因此 checker 推荐直接传入路径：



```python

onnx.checker.check\_model(str(path))

```



## 10. 打包 ONNX 产物



x86 宿主机执行：



```bash

bash scripts/x86/06\_pack\_onnx\_artifacts.sh

```



生成的 ONNX 压缩包和 sha256 文件只在本地保存，不上传 GitHub。



## 11. 不上传内容



以下内容不要上传到 GitHub：



```text

模型权重

量化 checkpoint

ONNX 文件

ONNX 压缩包

完整校准集

原始 history

代理配置

日志大文件

```



GitHub 只上传：



```text

复现脚本

配置模板

少量 sample

中文流程文档

```




