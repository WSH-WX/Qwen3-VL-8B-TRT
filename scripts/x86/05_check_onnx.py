#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import onnx

onnx_root = Path("/models/Qwen3-VL-8B-Instruct-EdgeLLM/onnx")

targets = {
    "LLM ONNX": onnx_root / "llm" / "model.onnx",
    "Visual ONNX": onnx_root / "visual" / "model.onnx",
}

for name, path in targets.items():
    print("=" * 80)
    print(name)
    print(path)

    if not path.exists():
        print("MISSING")
        continue

    # external data ONNX 必须直接传路径给 checker。
    # 不建议先 onnx.load() 再 check，否则大模型 external data 容易出错。
    onnx.checker.check_model(str(path))

    model = onnx.load(str(path), load_external_data=False)

    domains = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    custom_ops = Counter(
        f"{node.domain}:{node.op_type}"
        for node in model.graph.node
        if node.domain
    )

    print("Domains:")
    for k, v in domains.items():
        print(f"  {k}: {v}")

    print("Custom ops:")
    for k, v in custom_ops.items():
        print(f"  {k}: {v}")

print("=" * 80)
print("ONNX check finished.")
