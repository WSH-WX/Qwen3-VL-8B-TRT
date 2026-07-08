#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def stats(xs):
    if not xs:
        return None
    return {
        "min": min(xs),
        "mean": mean(xs),
        "max": max(xs),
    }


def parse(path: Path):
    lines = path.read_text(errors="ignore").splitlines()

    ram_used = []
    ram_total = []
    swap_used = []
    swap_total = []
    gr3d = []
    emc = []
    vdd_in_mw = []
    temps = {}

    for line in lines:
        m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if m:
            ram_used.append(int(m.group(1)))
            ram_total.append(int(m.group(2)))

        m = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
        if m:
            swap_used.append(int(m.group(1)))
            swap_total.append(int(m.group(2)))

        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            gr3d.append(int(m.group(1)))

        m = re.search(r"EMC_FREQ\s+(\d+)%", line)
        if m:
            emc.append(int(m.group(1)))

        m = re.search(r"VDD_IN\s+(\d+)mW", line)
        if m:
            vdd_in_mw.append(int(m.group(1)))

        for name, val in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", line):
            temps.setdefault(name, []).append(float(val))

    result = {
        "file": str(path),
        "num_lines": len(lines),
        "ram_used_mb": stats(ram_used),
        "ram_total_mb_max": max(ram_total) if ram_total else None,
        "swap_used_mb": stats(swap_used),
        "swap_total_mb_max": max(swap_total) if swap_total else None,
        "gr3d_freq_percent": stats(gr3d),
        "emc_freq_percent": stats(emc),
        "vdd_in_mw": stats(vdd_in_mw),
        "temperatures_c": {k: stats(v) for k, v in temps.items()},
    }

    if result["ram_used_mb"] and result["ram_total_mb_max"]:
        result["ram_peak_percent"] = result["ram_used_mb"]["max"] / result["ram_total_mb_max"] * 100
    else:
        result["ram_peak_percent"] = None

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tegrastats_log", type=Path)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    result = parse(args.tegrastats_log)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote:", args.output_json)


if __name__ == "__main__":
    main()
