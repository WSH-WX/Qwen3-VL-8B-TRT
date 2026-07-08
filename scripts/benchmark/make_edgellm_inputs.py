#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


TEXT_SYSTEM_PROMPT = "你是一个简洁、自然的机器人助手。回答要适合语音播报，优先用一到两句话。"

VISION_SYSTEM_PROMPT = (
    "你是一个简洁、可靠的机器人视觉助手。"
    "请只根据图像和用户问题回答。"
    "回答应清楚、实用，优先服务于机器人感知、导航、巡检或交互任务。"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_text(args):
    data = load_json(args.text_dataset)
    if not isinstance(data, list):
        raise ValueError("text dataset must be a JSON list")

    requests = []
    manifest = []
    request_idx = 0

    for run_idx in range(args.repeat):
        for sample_idx, item in enumerate(data):
            prompt = str(item["prompt"]).strip()
            if not prompt:
                raise ValueError(f"empty prompt at sample_idx={sample_idx}")

            messages = []
            if args.system_prompt:
                messages.append({
                    "role": "system",
                    "content": args.system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })

            requests.append({
                "messages": messages
            })

            manifest.append({
                "request_idx": request_idx,
                "run_idx": run_idx,
                "sample_idx": sample_idx,
                "prompt_id": item.get("id", f"text_prompt_{sample_idx:03d}"),
                "domain": item.get("domain"),
                "intent": item.get("intent"),
                "difficulty": item.get("difficulty"),
                "asr_noise": item.get("asr_noise"),
                "expected_response_type": item.get("expected_response_type"),
                "prompt_approx_len": item.get("prompt_approx_len"),
                "prompt": prompt,
            })
            request_idx += 1

    out = {
        "batch_size": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_generate_length": args.max_generate_length,
        "requests": requests,
    }

    write_json(args.output_input_json, out)
    write_jsonl(args.output_manifest_jsonl, manifest)

    print("TEXT input written:", args.output_input_json)
    print("TEXT manifest written:", args.output_manifest_jsonl)
    print("samples:", len(data))
    print("repeat:", args.repeat)
    print("total_requests:", len(requests))
    print("max_generate_length:", args.max_generate_length)


def make_vision(args):
    data = load_json(args.vision_dataset)
    if not isinstance(data, list):
        raise ValueError("vision dataset must be a JSON list")

    requests = []
    manifest = []
    missing = []
    request_idx = 0

    for run_idx in range(args.repeat):
        for sample_idx, item in enumerate(data):
            prompt = str(item["prompt"]).strip()
            image_path = Path(str(item["image_path"]))

            if not prompt:
                raise ValueError(f"empty prompt at sample_idx={sample_idx}")
            if not image_path.exists():
                missing.append(str(image_path))

            messages = []
            if args.system_prompt:
                messages.append({
                    "role": "system",
                    "content": args.system_prompt
                })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path)
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            })

            requests.append({
                "messages": messages
            })

            manifest.append({
                "request_idx": request_idx,
                "run_idx": run_idx,
                "sample_idx": sample_idx,
                "prompt_id": item.get("id", f"vision_prompt_{sample_idx:03d}"),
                "source_dataset": item.get("source_dataset"),
                "source_index": item.get("source_index"),
                "question_id": item.get("question_id"),
                "cluster_name": item.get("cluster_name"),
                "source_type": item.get("source_type"),
                "synthetic_prompt": item.get("synthetic_prompt"),
                "synthetic_kind": item.get("synthetic_kind"),
                "image_path": str(image_path),
                "prompt": prompt,
            })
            request_idx += 1

    if missing:
        print("missing_images:", len(missing))
        for p in missing[:20]:
            print(p)
        raise SystemExit(1)

    out = {
        "batch_size": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_generate_length": args.max_generate_length,
        "requests": requests,
    }

    write_json(args.output_input_json, out)
    write_jsonl(args.output_manifest_jsonl, manifest)

    print("VISION input written:", args.output_input_json)
    print("VISION manifest written:", args.output_manifest_jsonl)
    print("samples:", len(data))
    print("repeat:", args.repeat)
    print("total_requests:", len(requests))
    print("max_generate_length:", args.max_generate_length)


def main():
    parser = argparse.ArgumentParser(description="Create TensorRT Edge-LLM input JSON from old vLLM benchmark datasets.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_text = sub.add_parser("text")
    p_text.add_argument("--text-dataset", type=Path, required=True)
    p_text.add_argument("--output-input-json", type=Path, required=True)
    p_text.add_argument("--output-manifest-jsonl", type=Path, required=True)
    p_text.add_argument("--repeat", type=int, default=3)
    p_text.add_argument("--max-generate-length", type=int, default=50)
    p_text.add_argument("--temperature", type=float, default=0.0)
    p_text.add_argument("--top-p", type=float, default=1.0)
    p_text.add_argument("--top-k", type=int, default=1)
    p_text.add_argument("--system-prompt", type=str, default=TEXT_SYSTEM_PROMPT)

    p_vision = sub.add_parser("vision")
    p_vision.add_argument("--vision-dataset", type=Path, required=True)
    p_vision.add_argument("--output-input-json", type=Path, required=True)
    p_vision.add_argument("--output-manifest-jsonl", type=Path, required=True)
    p_vision.add_argument("--repeat", type=int, default=3)
    p_vision.add_argument("--max-generate-length", type=int, default=720)
    p_vision.add_argument("--temperature", type=float, default=0.0)
    p_vision.add_argument("--top-p", type=float, default=1.0)
    p_vision.add_argument("--top-k", type=int, default=1)
    p_vision.add_argument("--system-prompt", type=str, default=VISION_SYSTEM_PROMPT)

    args = parser.parse_args()

    if args.mode == "text":
        make_text(args)
    elif args.mode == "vision":
        make_vision(args)


if __name__ == "__main__":
    main()
