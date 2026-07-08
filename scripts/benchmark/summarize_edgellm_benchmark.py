#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def basic_stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {
        "min": min(xs),
        "mean": mean(xs),
        "p50": median(xs),
        "p95": pct(xs, 95),
        "max": max(xs),
        "count": len(xs),
    }


def try_load_tokenizer(tokenizer_json: Path):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tokenizer_json))
    except Exception as e:
        print(f"[WARN] Cannot load tokenizer package or tokenizer JSON: {e}")
        print("[WARN] Falling back to rough token counting.")
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids)
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def parse_wall_time_to_seconds(s: str):
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return None


def parse_runtime_log(path: Path):
    result = {
        "process_elapsed_seconds_time_v": None,
        "processing_elapsed_seconds_log": None,
    }

    if not path.exists():
        return result

    lines = path.read_text(errors="ignore").splitlines()

    for line in lines:
        m = re.search(r"Elapsed \(wall clock\) time.*:\s+([0-9:.]+)", line)
        if m:
            result["process_elapsed_seconds_time_v"] = parse_wall_time_to_seconds(m.group(1))

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    start_dt = None
    end_dt = None

    for line in lines:
        m = re.match(r"\[(\d\d:\d\d:\d\d\.\d+)\]", line)
        if not m:
            continue

        try:
            t = datetime.strptime(date_prefix + " " + m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            continue

        if "Processing " in line and "batched requests" in line and start_dt is None:
            start_dt = t

        if "Processing complete:" in line:
            end_dt = t

    if start_dt and end_dt:
        elapsed = (end_dt - start_dt).total_seconds()
        if elapsed < 0:
            elapsed += 24 * 3600
        result["processing_elapsed_seconds_log"] = elapsed

    return result


def recursive_find_numeric(obj, path=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)
            found.extend(recursive_find_numeric(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            found.extend(recursive_find_numeric(v, new_path))
    elif isinstance(obj, (int, float)):
        found.append((path, obj))
    return found


def inspect_profile(profile_path: Path):
    if not profile_path or not profile_path.exists():
        return {
            "profile_exists": False,
            "latency_candidate_keys": [],
        }

    try:
        obj = load_json(profile_path)
    except Exception as e:
        return {
            "profile_exists": True,
            "profile_parse_error": str(e),
            "latency_candidate_keys": [],
        }

    numeric = recursive_find_numeric(obj)
    key_words = [
        "ttft", "first", "token", "latency", "e2e", "elapsed",
        "prefill", "decode", "tpot", "time", "duration"
    ]

    candidates = []
    for k, v in numeric:
        lk = k.lower()
        if any(w in lk for w in key_words):
            candidates.append({"path": k, "value": v})

    return {
        "profile_exists": True,
        "profile_top_type": type(obj).__name__,
        "latency_candidate_keys": candidates[:200],
    }


def parse_edgellm_profile_metrics(profile_path: Path, output_tokens_from_json):
    """
    Parse TensorRT Edge-LLM llm_inference --dumpProfile output.

    Current Edge-LLM profile provides aggregate metrics:
      - prefill.average_time_per_run_ms
      - generation.average_time_per_token_ms
      - generation.tokens_per_second
      - generation.average_tokens_per_run

    It does not provide per-request P50/P95 latency in the observed schema.
    """
    if not profile_path or not profile_path.exists() or profile_path.stat().st_size == 0:
        return {
            "profile_exists": False,
            "note": "profile file is missing or empty",
        }

    obj = load_json(profile_path)
    if not isinstance(obj, dict):
        return {
            "profile_exists": True,
            "profile_schema": type(obj).__name__,
            "note": "unsupported profile schema for direct Edge-LLM metric parsing",
        }

    prefill = obj.get("prefill") or {}
    generation = obj.get("generation") or {}

    prefill_avg_ms = prefill.get("average_time_per_run_ms")
    prefill_avg_tokens = prefill.get("average_tokens_per_run")
    prefill_tokens_per_s = prefill.get("tokens_per_second")
    prefill_total_runs = prefill.get("total_runs")
    prefill_computed_tokens = prefill.get("computed_tokens")
    prefill_reused_tokens = prefill.get("reused_tokens")

    gen_avg_tpot_ms = generation.get("average_time_per_token_ms")
    gen_avg_tokens_per_run = generation.get("average_tokens_per_run")
    gen_tokens_per_s = generation.get("tokens_per_second")
    gen_total_runs = generation.get("total_runs")
    gen_generated_tokens = generation.get("generated_tokens")

    approx_e2e_ms = None
    approx_ttft_ms = None

    if isinstance(prefill_avg_ms, (int, float)) and isinstance(gen_avg_tpot_ms, (int, float)):
        # Approximate first-token latency: prefill + one decode step.
        approx_ttft_ms = prefill_avg_ms + gen_avg_tpot_ms

    if (
        isinstance(prefill_avg_ms, (int, float))
        and isinstance(gen_avg_tpot_ms, (int, float))
        and isinstance(gen_avg_tokens_per_run, (int, float))
    ):
        # Approximate per-request runtime E2E under aggregate profile:
        # prefill + average generation token time * average generated tokens.
        approx_e2e_ms = prefill_avg_ms + gen_avg_tpot_ms * gen_avg_tokens_per_run

    output_token_total_from_json = sum(output_tokens_from_json) if output_tokens_from_json else 0

    token_count_check = {
        "output_json_total_output_tokens": output_token_total_from_json,
        "profile_generated_tokens": gen_generated_tokens,
        "match": (
            int(round(gen_generated_tokens)) == int(output_token_total_from_json)
            if isinstance(gen_generated_tokens, (int, float))
            else None
        ),
    }

    multimodal = obj.get("multimodal") or {}
    stages = obj.get("stages") or []

    stage_by_id = {}
    if isinstance(stages, list):
        for st in stages:
            if isinstance(st, dict) and st.get("stage_id"):
                stage_by_id[str(st.get("stage_id"))] = st

    vision_stage = stage_by_id.get("vision_encoder") or {}
    llm_prefill_stage = stage_by_id.get("llm_prefill") or {}
    llm_generation_stage = stage_by_id.get("llm_generation") or {}

    total_images = multimodal.get("total_images")
    total_image_tokens = multimodal.get("total_image_tokens")
    total_multimodal_tokens = multimodal.get("total_multimodal_tokens")
    multimodal_avg_time_per_token_ms = multimodal.get("average_time_per_token_ms")

    image_tokens_per_run = None
    multimodal_tokens_per_run = None
    approx_vision_encoder_time_ms_from_tokens = None

    if isinstance(total_images, (int, float)) and total_images:
        if isinstance(total_image_tokens, (int, float)):
            image_tokens_per_run = total_image_tokens / total_images
        if isinstance(total_multimodal_tokens, (int, float)):
            multimodal_tokens_per_run = total_multimodal_tokens / total_images

    if (
        isinstance(image_tokens_per_run, (int, float))
        and isinstance(multimodal_avg_time_per_token_ms, (int, float))
    ):
        approx_vision_encoder_time_ms_from_tokens = image_tokens_per_run * multimodal_avg_time_per_token_ms

    vision_encoder_avg_ms = vision_stage.get("average_time_per_run_ms")
    vision_encoder_gpu_stats = vision_stage.get("gpu_time_stats")
    llm_prefill_gpu_stats = llm_prefill_stage.get("gpu_time_stats")
    llm_generation_gpu_stats = llm_generation_stage.get("gpu_time_stats")

    return {
        "profile_exists": True,
        "profile_schema": "edge_llm_aggregate_profile",

        "prefill_average_time_per_run_ms": prefill_avg_ms,
        "prefill_average_tokens_per_run": prefill_avg_tokens,
        "prefill_tokens_per_second": prefill_tokens_per_s,
        "prefill_total_runs": prefill_total_runs,
        "prefill_computed_tokens": prefill_computed_tokens,
        "prefill_reused_tokens": prefill_reused_tokens,

        "generation_average_time_per_token_ms": gen_avg_tpot_ms,
        "generation_average_tokens_per_run": gen_avg_tokens_per_run,
        "generation_tokens_per_second": gen_tokens_per_s,
        "generation_total_runs": gen_total_runs,
        "generation_generated_tokens": gen_generated_tokens,

        "approx_runtime_ttft_ms": approx_ttft_ms,
        "approx_runtime_e2e_ms": approx_e2e_ms,
        "approx_runtime_tpot_ms_per_token": gen_avg_tpot_ms,
        "approx_runtime_decode_tokens_per_second": gen_tokens_per_s,

        "multimodal_total_images": total_images,
        "multimodal_total_image_tokens": total_image_tokens,
        "multimodal_total_multimodal_tokens": total_multimodal_tokens,
        "multimodal_average_time_per_token_ms": multimodal_avg_time_per_token_ms,
        "multimodal_image_tokens_per_run": image_tokens_per_run,
        "multimodal_tokens_per_run": multimodal_tokens_per_run,
        "vision_encoder_average_time_per_run_ms": vision_encoder_avg_ms,
        "vision_encoder_gpu_time_stats": vision_encoder_gpu_stats,
        "approx_vision_encoder_time_ms_from_image_tokens": approx_vision_encoder_time_ms_from_tokens,
        "llm_prefill_gpu_time_stats": llm_prefill_gpu_stats,
        "llm_generation_gpu_time_stats": llm_generation_gpu_stats,

        "peak_unified_memory_bytes": obj.get("peak_unified_memory_bytes"),
        "peak_unified_memory_mb": obj.get("peak_unified_memory_mb"),

        "token_count_check": token_count_check,

        "notes": {
            "prefill_average_time_per_run_ms": "Official Edge-LLM aggregate prefill time per request.",
            "generation_average_time_per_token_ms": "Official Edge-LLM aggregate generation time per output token; can be reported as Runtime TPOT mean.",
            "generation_tokens_per_second": "Official Edge-LLM aggregate decode throughput.",
            "approx_runtime_ttft_ms": "Approximation: prefill average time + one average generation token time. Profile does not expose exact per-request TTFT.",
            "approx_runtime_e2e_ms": "Approximation: prefill average time + average generation TPOT * average generated tokens per run. Profile does not expose per-request E2E distribution.",
            "p50_p95": "Observed profile schema only contains aggregate means, not per-request P50/P95.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--input-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--runtime-log", type=Path, required=True)
    ap.add_argument("--profile-json", type=Path, default=None)
    ap.add_argument("--tegrastats-summary-json", type=Path, default=None)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--max-generate-length", type=int, required=True)
    ap.add_argument("--output-summary-json", type=Path, required=True)
    args = ap.parse_args()

    input_data = load_json(args.input_json)
    output_data = load_json(args.output_json)
    manifest = load_jsonl(args.manifest_jsonl)

    requests = input_data.get("requests", [])
    responses = output_data.get("responses", [])

    tokenizer = try_load_tokenizer(args.tokenizer_json)

    finish_reasons = Counter()
    input_tokens = []
    output_tokens = []
    max_length_count = 0

    for r in responses:
        finish = r.get("finish_reason")
        finish_reasons[finish] += 1
        if finish == "max-length":
            max_length_count += 1

        formatted = r.get("formatted_complete_request", "")
        out_text = r.get("output_text", "")

        input_tokens.append(count_tokens(formatted, tokenizer))
        output_tokens.append(count_tokens(out_text, tokenizer))

    total_output_tokens = sum(output_tokens)

    runtime_info = parse_runtime_log(args.runtime_log)
    profile_info = inspect_profile(args.profile_json)
    edgellm_profile_metrics = parse_edgellm_profile_metrics(args.profile_json, output_tokens)

    tegra_summary = None
    if args.tegrastats_summary_json and args.tegrastats_summary_json.exists():
        tegra_summary = load_json(args.tegrastats_summary_json)

    processing_elapsed = runtime_info.get("processing_elapsed_seconds_log")
    process_elapsed = runtime_info.get("process_elapsed_seconds_time_v")

    approx_avg_e2e_processing = None
    approx_avg_e2e_process = None
    approx_e2e_toks_per_s_processing = None

    if responses and processing_elapsed:
        approx_avg_e2e_processing = processing_elapsed / len(responses)
        approx_e2e_toks_per_s_processing = total_output_tokens / processing_elapsed if processing_elapsed > 0 else None

    if responses and process_elapsed:
        approx_avg_e2e_process = process_elapsed / len(responses)

    report_metrics = {
        "token_source": "edge_llm_profile_when_available",
        "input_tokens_mean": (
            edgellm_profile_metrics.get("prefill_average_tokens_per_run")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "output_tokens_mean": (
            edgellm_profile_metrics.get("generation_average_tokens_per_run")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "prefill_time_mean_ms": (
            edgellm_profile_metrics.get("prefill_average_time_per_run_ms")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "runtime_ttft_approx_ms": (
            edgellm_profile_metrics.get("approx_runtime_ttft_ms")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "runtime_e2e_approx_ms": (
            edgellm_profile_metrics.get("approx_runtime_e2e_ms")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "runtime_tpot_ms_per_token": (
            edgellm_profile_metrics.get("approx_runtime_tpot_ms_per_token")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "runtime_decode_tokens_per_second": (
            edgellm_profile_metrics.get("approx_runtime_decode_tokens_per_second")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "profile_peak_unified_memory_mb": (
            edgellm_profile_metrics.get("peak_unified_memory_mb")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "vision_encoder_time_mean_ms": (
            edgellm_profile_metrics.get("vision_encoder_average_time_per_run_ms")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "image_tokens_mean": (
            edgellm_profile_metrics.get("multimodal_image_tokens_per_run")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "multimodal_tokens_mean": (
            edgellm_profile_metrics.get("multimodal_tokens_per_run")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "multimodal_time_per_token_ms": (
            edgellm_profile_metrics.get("multimodal_average_time_per_token_ms")
            if isinstance(edgellm_profile_metrics, dict) else None
        ),
        "note": (
            "These are the recommended report-level Edge-LLM metrics. "
            "Token counts and runtime metrics are taken from Edge-LLM profile when available. "
            "Basic token stats from output_text may be underestimated if Python tokenizers is unavailable."
        ),
    }

    summary = {
        "task_name": args.task_name,
        "input_json": str(args.input_json),
        "output_json": str(args.output_json),
        "manifest_jsonl": str(args.manifest_jsonl),
        "runtime_log": str(args.runtime_log),
        "profile_json": str(args.profile_json) if args.profile_json else None,
        "tokenizer_json": str(args.tokenizer_json),
        "max_generate_length": args.max_generate_length,

        "num_input_requests": len(requests),
        "num_manifest_rows": len(manifest),
        "num_responses": len(responses),
        "num_failed_estimated": len(requests) - len(responses),

        "finish_reason_count": dict(finish_reasons),
        "max_length_count": max_length_count,
        "max_length_ratio": max_length_count / len(responses) if responses else None,

        "input_tokens": basic_stats(input_tokens),
        "output_tokens": basic_stats(output_tokens),
        "total_output_tokens": total_output_tokens,

        "report_metrics": report_metrics,

        "runtime_log_metrics": runtime_info,
        "approx_avg_e2e_seconds_from_processing_log": approx_avg_e2e_processing,
        "approx_avg_e2e_seconds_from_process_time": approx_avg_e2e_process,
        "approx_end_to_end_output_tokens_per_second_from_processing_log": approx_e2e_toks_per_s_processing,

        "profile_inspection": profile_info,
        "edgellm_profile_metrics": edgellm_profile_metrics,
        "tegrastats_summary": tegra_summary,

        "metric_notes": {
            "latency_boundary": "Edge-LLM latency is runtime-side / process-side, not vLLM OpenAI API streaming latency.",
            "token_count": "Token counts are recomputed from formatted_complete_request and output_text using tokenizer.json when available.",
            "ttft_tpot": "Precise TTFT/TPOT require Edge-LLM profile schema. This script first exposes profile latency candidate keys for follow-up adaptation.",
            "approx_e2e": "The processing-log E2E is a log-level approximation over all requests and should not be reported as per-request API E2E."
        }
    }

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", args.output_summary_json)


if __name__ == "__main__":
    main()
