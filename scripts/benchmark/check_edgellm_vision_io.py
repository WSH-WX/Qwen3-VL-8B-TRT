#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_check(name, ok, actual="", expected=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}")
    if actual != "" or expected != "":
        print(f"       actual  : {actual}")
        print(f"       expected: {expected}")
    return ok


def extract_user_content(req):
    messages = req.get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


def extract_image_and_text_from_request(req):
    content = extract_user_content(req)
    image_path = None
    text = None

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                image_path = part.get("image")
            elif part.get("type") == "text":
                text = part.get("text")
            elif part.get("type") == "image_url":
                image_path = part.get("image_url", {}).get("url")
    elif isinstance(content, str):
        text = content

    return image_path, text


def get_run_index(meta, i, prompt_count):
    for key in ["run_index_for_prompt", "repeat_index", "run_index"]:
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass
    return i // prompt_count + 1


def get_prompt_id(meta, i):
    return str(
        meta.get("prompt_id")
        or meta.get("id")
        or meta.get("sample_id")
        or f"vision_prompt_{i + 1:03d}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-dataset", default="/mnt/ssd/qwen3vl8b_edgellm_benchmark/datasets/vision_arena_robot_like/vision_arena_robot_hybrid50.local.json")
    parser.add_argument("--input-json", default="/mnt/ssd/qwen3vl8b_edgellm_benchmark/inputs/input_vision_hybrid50_repeat3_max720.json")
    parser.add_argument("--manifest-jsonl", default="/mnt/ssd/qwen3vl8b_edgellm_benchmark/inputs/manifest_vision_hybrid50_repeat3_max720.jsonl")
    parser.add_argument("--output-json", default="/mnt/ssd/qwen3vl8b_edgellm_benchmark/outputs/output_vision_hybrid50_repeat3_max720.json")
    parser.add_argument("--profile-json", default="/mnt/ssd/qwen3vl8b_edgellm_benchmark/profiles/profile_vision_hybrid50_repeat3_max720.json")
    parser.add_argument("--expected-prompts", type=int, default=50)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-requests", type=int, default=150)
    parser.add_argument("--expected-max-generate-length", type=int, default=720)
    parser.add_argument("--check-output", action="store_true", help="同时检查 output/profile，适合 benchmark 跑完后使用。")
    args = parser.parse_args()

    all_ok = True

    dataset_path = Path(args.vision_dataset)
    input_path = Path(args.input_json)
    manifest_path = Path(args.manifest_jsonl)
    output_path = Path(args.output_json)
    profile_path = Path(args.profile_json)

    print("=" * 100)
    print("1. Check required input files")
    print("=" * 100)

    for name, path in [
        ("vision_dataset", dataset_path),
        ("input_json", input_path),
        ("manifest_jsonl", manifest_path),
    ]:
        ok = path.exists() and path.stat().st_size > 0
        all_ok &= print_check(name, ok, f"{path} ({path.stat().st_size if path.exists() else 'missing'} bytes)", "exists and non-empty")

    if not all_ok:
        print("\nInput files missing. Stop.")
        sys.exit(1)

    dataset = load_json(dataset_path)
    input_data = load_json(input_path)
    manifest = load_jsonl(manifest_path)

    print("\n" + "=" * 100)
    print("2. Check vision dataset format")
    print("=" * 100)

    all_ok &= print_check("dataset is list", isinstance(dataset, list), type(dataset).__name__, "list")
    all_ok &= print_check("dataset sample count", len(dataset) == args.expected_prompts, len(dataset), args.expected_prompts)

    missing_image = []
    empty_prompt = []
    image_paths = []
    prompt_ids = []

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            all_ok &= print_check(f"dataset item {idx} is dict", False, type(item).__name__, "dict")
            continue

        img = item.get("image_path")
        prompt = str(item.get("prompt", "")).strip()
        pid = str(item.get("id", f"vision_prompt_{idx + 1:03d}"))

        prompt_ids.append(pid)

        if not prompt:
            empty_prompt.append(idx)

        if not img or not Path(str(img)).exists():
            missing_image.append((idx, img))
        else:
            image_paths.append(str(img))

    all_ok &= print_check("missing images", len(missing_image) == 0, missing_image[:5], 0)
    all_ok &= print_check("empty prompts", len(empty_prompt) == 0, empty_prompt[:5], 0)
    all_ok &= print_check("unique prompt ids", len(set(prompt_ids)) == args.expected_prompts, len(set(prompt_ids)), args.expected_prompts)
    print("unique image paths:", len(set(image_paths)))

    print("\n" + "=" * 100)
    print("3. Check Edge-LLM input JSON format")
    print("=" * 100)

    requests = input_data.get("requests", [])
    all_ok &= print_check("input requests count", len(requests) == args.expected_requests, len(requests), args.expected_requests)
    all_ok &= print_check("max_generate_length", input_data.get("max_generate_length") == args.expected_max_generate_length, input_data.get("max_generate_length"), args.expected_max_generate_length)
    all_ok &= print_check("temperature", float(input_data.get("temperature", -1)) == 0.0, input_data.get("temperature"), 0.0)
    all_ok &= print_check("top_p", float(input_data.get("top_p", -1)) == 1.0, input_data.get("top_p"), 1.0)
    all_ok &= print_check("top_k", int(input_data.get("top_k", -1)) == 1, input_data.get("top_k"), 1)

    bad_format = []
    missing_req_images = []
    empty_req_text = []

    for i, req in enumerate(requests):
        img, text = extract_image_and_text_from_request(req)

        if not img or not text:
            bad_format.append((i, img, text))
        if img and not Path(str(img)).exists():
            missing_req_images.append((i, img))
        if not text or not str(text).strip():
            empty_req_text.append(i)

    all_ok &= print_check("request multimodal format image+text", len(bad_format) == 0, bad_format[:3], 0)
    all_ok &= print_check("request image paths exist", len(missing_req_images) == 0, missing_req_images[:3], 0)
    all_ok &= print_check("request text non-empty", len(empty_req_text) == 0, empty_req_text[:3], 0)

    print("\n" + "=" * 100)
    print("4. Check manifest alignment")
    print("=" * 100)

    all_ok &= print_check("manifest rows count", len(manifest) == args.expected_requests, len(manifest), args.expected_requests)

    if manifest:
        manifest_prompt_ids = [get_prompt_id(m, i) for i, m in enumerate(manifest)]
        run_counter = Counter(get_run_index(m, i, args.expected_prompts) for i, m in enumerate(manifest))
        unique_manifest_ids = set(manifest_prompt_ids)

        all_ok &= print_check("manifest unique prompt count", len(unique_manifest_ids) == args.expected_prompts, len(unique_manifest_ids), args.expected_prompts)
        expected_run_dist = {i: args.expected_prompts for i in range(1, args.expected_runs + 1)}
        all_ok &= print_check("manifest repeat distribution", dict(run_counter) == expected_run_dist, dict(run_counter), expected_run_dist)

    if args.check_output:
        print("\n" + "=" * 100)
        print("5. Check output/profile after benchmark")
        print("=" * 100)

        for name, path in [
            ("output_json", output_path),
            ("profile_json", profile_path),
        ]:
            ok = path.exists() and path.stat().st_size > 0
            all_ok &= print_check(name, ok, f"{path} ({path.stat().st_size if path.exists() else 'missing'} bytes)", "exists and non-empty")

        if output_path.exists() and output_path.stat().st_size > 0:
            output_data = load_json(output_path)
            responses = output_data.get("responses", [])
            finish = Counter(r.get("finish_reason") for r in responses)
            non_empty = sum(1 for r in responses if str(r.get("output_text", "")).strip())

            all_ok &= print_check("output responses count", len(responses) == args.expected_requests, len(responses), args.expected_requests)
            all_ok &= print_check("non-empty output_text count", non_empty == len(responses), non_empty, len(responses))
            print("finish_reason_count:", dict(finish))

        if profile_path.exists() and profile_path.stat().st_size > 0:
            profile = load_json(profile_path)
            print("profile_top_keys:", list(profile.keys()) if isinstance(profile, dict) else type(profile).__name__)

    print("\n" + "=" * 100)
    print("Final verdict")
    print("=" * 100)

    if all_ok:
        print("ALL CHECKS PASSED.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED. Please inspect FAIL items above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
