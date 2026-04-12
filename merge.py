#!/usr/bin/env python3
"""NDJSON Dataset Fusion - Merge multiple YOLO .ndjson datasets with class balancing."""

import glob
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ── Terminal colors (ANSI, stdlib only) ─────────────────────────────────────

def _use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    return sys.stdout.isatty()


def style(text: str, *codes: int) -> str:
    """Wrap text in SGR codes; no-op when color disabled."""
    if not _use_color() or not codes:
        return text
    seq = ";".join(str(c) for c in codes)
    return f"\033[{seq}m{text}\033[0m"


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_header(title: str):
    width = 60
    bar = "=" * width
    bar_styled = style(bar, 36) if _use_color() else bar
    title_styled = style(f"  {title}", 1, 96) if _use_color() else f"  {title}"
    print(f"\n{bar_styled}")
    print(title_styled)
    print(bar_styled)


def print_table(headers: list[str], rows: list[list], align: list[str] | None = None):
    """Pretty-print a table. align: list of '<' or '>' per column."""
    if not rows:
        print(style("  (no data)", 2))
        return
    col_widths = [len(h) for h in headers]
    str_rows = [[str(c) for c in r] for r in rows]
    for r in str_rows:
        for i, c in enumerate(r):
            col_widths[i] = max(col_widths[i], len(c))
    if align is None:
        align = ["<"] * len(headers)
    fmt = "  ".join(f"{{:{a}{w}}}" for a, w in zip(align, col_widths))
    sep = "  ".join("-" * w for w in col_widths)
    header_line = fmt.format(*headers)
    print(f"  {style(header_line, 1, 33)}" if _use_color() else f"  {header_line}")
    print(style(f"  {sep}", 2) if _use_color() else f"  {sep}")
    for r in str_rows:
        line = fmt.format(*r)
        if _use_color() and r[1] == "TOTAL":
            line = style(line, 1, 32)
        print(f"  {line}")


def prompt_choice(prompt: str, options: list[str]) -> int:
    """Ask user to pick one option by number. Returns 0-based index."""
    print(f"\n{style(prompt, 1, 97)}")
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")
    while True:
        raw = input(style("  > ", 35)).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(style(f"  Masukkan angka 1-{len(options)}", 31))


def prompt_multi_choice(prompt: str, options: list[str]) -> list[int]:
    """Ask user to pick one or more options (comma-separated). Returns sorted 0-based indices."""
    print(f"\n{style(prompt, 1, 97)}")
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")
    print(f"  {style('[a]', 32, 1)} Pilih semua")
    while True:
        raw = input(style("  > ", 35)).strip().lower()
        if raw == "a":
            return list(range(len(options)))
        parts = [p.strip() for p in raw.split(",")]
        try:
            indices = [int(p) - 1 for p in parts]
            if all(0 <= i < len(options) for i in indices) and indices:
                return sorted(set(indices))
        except ValueError:
            pass
        print(
            style(
                f"  Masukkan angka 1-{len(options)} dipisah koma, atau 'a' untuk semua",
                31,
            )
        )


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_ndjson(filepath: str) -> tuple[dict, list[dict]]:
    """Parse an .ndjson file. Returns (dataset_header, list_of_image_records)."""
    header = None
    images = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "dataset":
                header = record
            elif record.get("type") == "image":
                images.append(record)
    if header is None:
        basename = os.path.basename(filepath)
        header = {"type": "dataset", "name": basename, "class_names": {}}
    return header, images


def build_unified_class_map(
    selected: list[tuple[str, dict, list[dict]]]
) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    """
    Build a unified class name -> new_id map across all selected datasets.
    Returns:
        unified_map: {class_name: unified_id}
        remap_tables: {filepath: {old_id: new_id}}
    """
    unified_map: dict[str, int] = {}
    remap_tables: dict[str, dict[int, int]] = {}

    for filepath, header, _images in selected:
        class_names = header.get("class_names", {})
        local_remap: dict[int, int] = {}
        for old_id_str, name in class_names.items():
            old_id = int(old_id_str)
            if name not in unified_map:
                unified_map[name] = len(unified_map)
            local_remap[old_id] = unified_map[name]
        remap_tables[filepath] = local_remap

    return unified_map, remap_tables


def remap_image(image: dict, remap: dict[int, int]) -> dict:
    """Return a copy of image with class IDs remapped in annotations.boxes."""
    img = json.loads(json.dumps(image))
    boxes = img.get("annotations", {}).get("boxes")
    if boxes:
        for box in boxes:
            old_id = int(box[0])
            box[0] = remap.get(old_id, old_id)
    return img


# ── Counting ─────────────────────────────────────────────────────────────────

def count_distribution(
    images: list[dict], unified_map: dict[str, int]
) -> tuple[dict[int, int], dict[int, int], int]:
    """
    Count annotations and images per class.
    Returns:
        ann_per_class: {class_id: annotation_count}
        img_per_class: {class_id: image_count}
        no_ann_count: number of images without annotations
    """
    ann_per_class = defaultdict(int)
    img_per_class = defaultdict(int)
    no_ann_count = 0
    id_to_name = {v: k for k, v in unified_map.items()}

    for img in images:
        boxes = img.get("annotations", {}).get("boxes", [])
        if not boxes:
            no_ann_count += 1
            continue
        classes_in_img = set()
        for box in boxes:
            cid = int(box[0])
            ann_per_class[cid] += 1
            classes_in_img.add(cid)
        for cid in classes_in_img:
            img_per_class[cid] += 1

    return dict(ann_per_class), dict(img_per_class), no_ann_count


def display_distribution(
    ann_per_class: dict[int, int],
    img_per_class: dict[int, int],
    no_ann_count: int,
    id_to_name: dict[int, str],
):
    print_header("Distribusi Dataset")
    headers = ["Class ID", "Class Name", "Annotations", "Images"]
    rows = []
    for cid in sorted(id_to_name.keys()):
        rows.append([
            cid,
            id_to_name[cid],
            ann_per_class.get(cid, 0),
            img_per_class.get(cid, 0),
        ])
    rows.append(["", "TOTAL", sum(ann_per_class.values()), ""])
    print_table(headers, rows, align=[">"," <", ">", ">"])
    if no_ann_count:
        print(
            f"\n  {style('Images tanpa annotation:', 33)} "
            f"{style(str(no_ann_count), 1, 33)}"
        )


# ── Balancing ────────────────────────────────────────────────────────────────

def get_class_images(
    images: list[dict], unified_map: dict[str, int]
) -> dict[int, list[int]]:
    """Map each class_id to list of image indices that contain that class."""
    class_imgs: dict[int, list[int]] = defaultdict(list)
    for idx, img in enumerate(images):
        boxes = img.get("annotations", {}).get("boxes", [])
        classes_in_img = set()
        for box in boxes:
            classes_in_img.add(int(box[0]))
        for cid in classes_in_img:
            class_imgs[cid].append(idx)
    return dict(class_imgs)


def balance_images(
    images: list[dict],
    unified_map: dict[str, int],
    mode: str,
    targets: dict[int, int] | None = None,
    include_no_ann: bool = False,
) -> list[dict]:
    """
    Sample images to achieve the desired class balance.
    mode: "none" | "equal" | "ratio" | "count"
    targets: for ratio/count mode, {class_id: target_annotation_count}
    """
    annotated = [img for img in images if img.get("annotations", {}).get("boxes")]
    no_ann = [img for img in images if not img.get("annotations", {}).get("boxes")]

    if mode == "none":
        result = list(annotated)
        if include_no_ann:
            result.extend(no_ann)
        return result

    ann_per_class = defaultdict(int)
    for img in annotated:
        for box in img["annotations"]["boxes"]:
            ann_per_class[int(box[0])] += 1

    if mode == "equal":
        if not ann_per_class:
            return list(no_ann) if include_no_ann else []
        min_count = min(ann_per_class.values())
        targets = {cid: min_count for cid in ann_per_class}

    class_imgs = defaultdict(list)
    for idx, img in enumerate(annotated):
        classes_in_img = set()
        for box in img["annotations"]["boxes"]:
            classes_in_img.add(int(box[0]))
        for cid in classes_in_img:
            class_imgs[cid].append(idx)

    selected_indices: set[int] = set()
    current_counts: dict[int, int] = defaultdict(int)

    for cid in sorted(targets.keys()):
        candidate_indices = class_imgs.get(cid, [])
        random.shuffle(candidate_indices)
        for idx in candidate_indices:
            if current_counts[cid] >= targets[cid]:
                break
            if idx in selected_indices:
                continue
            img = annotated[idx]
            add_counts: dict[int, int] = defaultdict(int)
            for box in img["annotations"]["boxes"]:
                add_counts[int(box[0])] += 1
            overshoot = False
            for c, cnt in add_counts.items():
                if c in targets and current_counts[c] + cnt > targets[c] * 1.2:
                    overshoot = True
                    break
            if not overshoot:
                selected_indices.add(idx)
                for c, cnt in add_counts.items():
                    current_counts[c] += cnt

    result = [annotated[i] for i in sorted(selected_indices)]
    if include_no_ann:
        result.extend(no_ann)
    return result


# ── Output ───────────────────────────────────────────────────────────────────

def write_merged(
    output_path: str,
    unified_map: dict[str, int],
    images: list[dict],
    source_names: list[str],
):
    id_to_name = {v: k for k, v in unified_map.items()}
    class_names_out = {str(k): v for k, v in sorted(id_to_name.items())}
    now = datetime.now()

    header = {
        "type": "dataset",
        "task": "detect",
        "name": " + ".join(source_names),
        "description": f"Merged from {len(source_names)} dataset(s)",
        "bytes": 0,
        "class_names": class_names_out,
        "version": "merged",
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for img in images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")

    return output_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_header("YOLO NDJSON Dataset Fusion")

    # 1. Scan files
    ndjson_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ndjson")))
    if not ndjson_files:
        print(style(f"\n  Tidak ada file .ndjson di {DATA_DIR}", 31, 1))
        sys.exit(1)

    # 2. Select files
    display_names = []
    parsed: list[tuple[str, dict, list[dict]]] = []
    for fp in ndjson_files:
        header, images = parse_ndjson(fp)
        parsed.append((fp, header, images))
        name = header.get("name", os.path.basename(fp))
        class_names = header.get("class_names", {})
        classes_str = ", ".join(f"{v}" for v in class_names.values()) if class_names else "-"
        display_names.append(f"{name}  ({len(images)} images, classes: {classes_str})")

    indices = prompt_multi_choice("Pilih file yang ingin di-merge:", display_names)
    selected = [parsed[i] for i in indices]
    print(f"\n  {style('Dipilih:', 2)} {style(str(len(selected)), 32, 1)} file")

    # 3. Build unified class map & remap
    unified_map, remap_tables = build_unified_class_map(selected)
    id_to_name = {v: k for k, v in unified_map.items()}

    all_images: list[dict] = []
    source_names: list[str] = []
    for filepath, header, images in selected:
        remap = remap_tables[filepath]
        for img in images:
            all_images.append(remap_image(img, remap))
        source_names.append(header.get("name", os.path.basename(filepath)))

    # 4. Count & display distribution
    ann_per_class, img_per_class, no_ann_count = count_distribution(all_images, unified_map)
    display_distribution(ann_per_class, img_per_class, no_ann_count, id_to_name)

    # 5. Handle images without annotations
    include_no_ann = False
    if no_ann_count > 0:
        choice = prompt_choice(
            f"Ada {no_ann_count} image tanpa annotation. Apa yang ingin dilakukan?",
            ["Include (sebagai negative sample)", "Skip (buang)"],
        )
        include_no_ann = choice == 0

    # 6. Balancing mode
    mode_choice = prompt_choice(
        "Pilih mode balancing:",
        [
            "No balance (gunakan semua data apa adanya)",
            "Equal balance (undersample ke jumlah class terkecil)",
            "Custom ratio (tentukan persentase per class)",
            "Custom count (tentukan jumlah annotation per class)",
        ],
    )
    mode_map = {0: "none", 1: "equal", 2: "ratio", 3: "count"}
    mode = mode_map[mode_choice]

    targets = None
    if mode in ("ratio", "count"):
        total_ann = sum(ann_per_class.values())
        targets = {}
        print(f"\n  {style('Masukkan target per class:', 1, 97)}")
        for cid in sorted(id_to_name.keys()):
            name = id_to_name[cid]
            current = ann_per_class.get(cid, 0)
            if mode == "ratio":
                while True:
                    raw = input(
                        style(
                            f"    {name} (saat ini {current} ann) - persentase target (%): ",
                            2,
                        )
                    ).strip()
                    try:
                        pct = float(raw)
                        if 0 < pct <= 100:
                            targets[cid] = int(total_ann * pct / 100)
                            print(
                                style(
                                    f"      -> target: ~{targets[cid]} annotations",
                                    32,
                                )
                            )
                            break
                    except ValueError:
                        pass
                    print(style("      Masukkan angka 1-100", 31))
            else:
                while True:
                    raw = input(
                        style(
                            f"    {name} (saat ini {current} ann) - jumlah target: ",
                            2,
                        )
                    ).strip()
                    try:
                        count = int(raw)
                        if count > 0:
                            targets[cid] = count
                            break
                    except ValueError:
                        pass
                    print(style("      Masukkan angka positif", 31))

    # 7. Apply balancing
    balanced = balance_images(all_images, unified_map, mode, targets, include_no_ann)

    # Show result preview
    bal_ann, bal_img, bal_no_ann = count_distribution(balanced, unified_map)
    print_header("Preview Hasil Setelah Balancing")
    display_distribution(bal_ann, bal_img, bal_no_ann, id_to_name)
    print(
        f"\n  {style('Total images dalam output:', 1)} "
        f"{style(str(len(balanced)), 96, 1)}"
    )

    # 8. Output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"merged_{timestamp}.ndjson"
    raw_name = input(
        style(
            f"\n  Nama file output (kosongkan untuk '{default_name}'): ",
            2,
        )
    ).strip()
    if not raw_name:
        raw_name = default_name
    if not raw_name.endswith(".ndjson"):
        raw_name += ".ndjson"

    output_path = os.path.join(OUTPUT_DIR, raw_name)

    # 9. Confirm
    print(
        f"\n  {style('Akan menulis', 2)} "
        f"{style(str(len(balanced)), 33, 1)} {style('images ke:', 2)} "
        f"{style(output_path, 36, 1)}"
    )
    confirm = input(style("  Lanjutkan? (y/n): ", 35)).strip().lower()
    if confirm not in ("y", "yes"):
        print(style("  Dibatalkan.", 33))
        sys.exit(0)

    # 10. Write
    write_merged(output_path, unified_map, balanced, source_names)

    print_header("Selesai!")
    print(f"  {style('Output:', 2)} {style(output_path, 32, 1)}")
    print(
        f"  {style('Total images:', 2)} "
        f"{style(str(len(balanced)), 97, 1)}"
    )
    total_boxes = sum(
        len(img.get("annotations", {}).get("boxes", []))
        for img in balanced
    )
    print(
        f"  {style('Total annotations:', 2)} "
        f"{style(str(total_boxes), 97, 1)}"
    )
    print()


if __name__ == "__main__":
    main()
