#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


# 解析: {obj_id}_{yaw}_{elev}.png
NAME_RE = re.compile(r"^(.+)_(-?\d+)_(-?\d+)\.png$")


def parse_name(name: str):
    m = NAME_RE.match(name)
    if m is None:
        return None
    obj_id = m.group(1)
    yaw = int(m.group(2))
    elev = int(m.group(3))
    return obj_id, yaw, elev


def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(records, desc="Writing jsonl", dynamic_ncols=True):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orig_dir",
        type=Path,
        default=Path("/data/home/2024120101018/linzhuohang/3DGen/data/testset/20260308_221610/orig"),
    )
    ap.add_argument(
        "--orig15_dir",
        type=Path,
        default=Path("/data/home/2024120101018/linzhuohang/3DGen/data/testset/20260308_221610/orig_15"),
    )
    ap.add_argument(
        "--out_jsonl",
        type=Path,
        default=Path("/data/home/2024120101018/linzhuohang/3DGen/data/testset/20260308_221610/orig_to15_manifest.jsonl"),
    )
    ap.add_argument(
        "--skip_ref_elev_15",
        action="store_true",
        help="如果开启，则跳过 ref_elev=15 的样本",
    )
    args = ap.parse_args()

    assert args.orig_dir.exists(), f"orig_dir not found: {args.orig_dir}"
    assert args.orig15_dir.exists(), f"orig15_dir not found: {args.orig15_dir}"

    orig_files = sorted(args.orig_dir.glob("*.png"))

    total_orig_png = 0
    bad_name_count = 0
    missing_target_count = 0
    skipped_ref15_count = 0
    ok_count = 0

    records = []

    pbar = tqdm(orig_files, desc="Scanning orig", dynamic_ncols=True)
    for p in pbar:
        total_orig_png += 1
        pbar.set_postfix_str(p.name)

        parsed = parse_name(p.name)
        if parsed is None:
            bad_name_count += 1
            continue

        obj_id, yaw, elev = parsed

        if args.skip_ref_elev_15 and elev == 15:
            skipped_ref15_count += 1
            continue

        tgt = args.orig15_dir / p.name
        if not tgt.exists():
            missing_target_count += 1
            continue

        sample_id = p.stem
        records.append(
            {
                "sample_id": sample_id,
                "obj_id": obj_id,
                "ref_img": str(p.resolve()),
                "tgt_img": str(tgt.resolve()),
                "ref_yaw": yaw,
                "ref_elev": elev,
                "tgt_yaw": yaw,
                "tgt_elev": 15,
            }
        )
        ok_count += 1

    records = sorted(records, key=lambda x: (x["obj_id"], x["ref_yaw"], x["ref_elev"]))

    write_jsonl(records, args.out_jsonl)

    stats = {
        "orig_dir": str(args.orig_dir),
        "orig15_dir": str(args.orig15_dir),
        "out_jsonl": str(args.out_jsonl),
        "skip_ref_elev_15": bool(args.skip_ref_elev_15),
        "total_orig_png": total_orig_png,
        "bad_name_count": bad_name_count,
        "missing_target_count": missing_target_count,
        "skipped_ref15_count": skipped_ref15_count,
        "ok_count": ok_count,
    }

    stats_path = args.out_jsonl.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] orig -> orig_15 manifest build finished")
    print(f"orig_dir: {args.orig_dir}")
    print(f"orig15_dir: {args.orig15_dir}")
    print(f"out_jsonl: {args.out_jsonl}")
    print(f"stats_json: {stats_path}")
    print("-" * 80)
    print(f"total_orig_png: {total_orig_png}")
    print(f"bad_name_count: {bad_name_count}")
    print(f"missing_target_count: {missing_target_count}")
    print(f"skipped_ref15_count: {skipped_ref15_count}")
    print(f"ok_count: {ok_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()