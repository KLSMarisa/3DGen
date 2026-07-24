import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


def _to_hashable(item: Any) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def deduplicate_list(items: Iterable[Any]) -> Tuple[list[Any], int]:
    seen = set()
    unique_items: list[Any] = []
    duplicate_count = 0

    for item in items:
        key = _to_hashable(item)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_items.append(item)

    return unique_items, duplicate_count


def process_json_file(input_path: Path, output_path: Path, global_seen: Optional[set[str]] = None) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} 不是 JSON 数组，无法按条目去重")

    seen_in_file = set()
    unique_items: list[Any] = []
    duplicate_in_file = 0
    duplicate_across_files = 0

    for item in data:
        key = _to_hashable(item)

        if key in seen_in_file:
            duplicate_in_file += 1
            continue

        if global_seen is not None and key in global_seen:
            duplicate_across_files += 1
            continue

        seen_in_file.add(key)
        if global_seen is not None:
            global_seen.add(key)
        unique_items.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(unique_items, f, ensure_ascii=False, indent=2)

    return duplicate_in_file, duplicate_across_files


def main() -> None:
    parser = argparse.ArgumentParser(description="对 data_queue 下 JSON 条目去重并保存")
    parser.add_argument(
        "--data-queue-dir",
        default="data_queue",
        help="data_queue 目录路径（默认：data_queue）",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖原文件（默认保存为 *_dedup.json）",
    )
    parser.add_argument(
        "--across-files",
        action="store_true",
        help="开启跨文件去重（同一条目仅保留首次出现）",
    )
    args = parser.parse_args()

    data_queue_dir = Path(args.data_queue_dir)
    if not data_queue_dir.exists() or not data_queue_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {data_queue_dir}")

    json_files = sorted(data_queue_dir.glob("*.json"))
    if not json_files:
        print(f"在 {data_queue_dir} 下未找到 .json 文件")
        return

    total_duplicates_in_file = 0
    total_duplicates_across_files = 0
    processed_count = 0
    global_seen = set() if args.across_files else None

    for json_file in json_files:
        output_path = json_file if args.in_place else json_file.with_name(f"{json_file.stem}_dedup.json")
        try:
            duplicate_in_file, duplicate_across_files = process_json_file(json_file, output_path, global_seen)
        except Exception as e:
            print(f"[跳过] {json_file.name}: {e}")
            continue

        processed_count += 1
        total_duplicates_in_file += duplicate_in_file
        total_duplicates_across_files += duplicate_across_files
        print(
            f"{json_file.name}: 去重完成，文件内重复 {duplicate_in_file} 个"
            + (f"，跨文件重复 {duplicate_across_files} 个" if args.across_files else "")
            + f"，输出 -> {output_path.name}"
        )

    print("-" * 60)
    print(f"处理文件数: {processed_count}")
    print(f"文件内重复总数: {total_duplicates_in_file}")
    if args.across_files:
        print(f"跨文件重复总数: {total_duplicates_across_files}")
    print(f"重复条目总数: {total_duplicates_in_file + total_duplicates_across_files}")


if __name__ == "__main__":
    main()