#!/usr/bin/env python3
"""
Rename class label strings in YOLO NDJSON dataset headers (type \"dataset\" -> class_names).
Run on copies under data/ or write to output/, then merge with merge.py.

With no INPUT arguments, runs an interactive wizard that lists data/*.ndjson and prompts
for map file, OLD=NEW rules, dry-run, in-place, and output directory.

Box class IDs are unchanged for 1:1 renames. If two class IDs would share the same new name,
the script exits with an error (merging IDs is not supported in v1).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict

from cli_utils import print_header, style
from paths import DATA_DIR, OUTPUT_DIR, ensure_data_and_output_dirs


def _load_map_file(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"--map file must be a JSON object, got {type(data).__name__}")
    out: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise SystemExit(f"--map keys and values must be strings, got {k!r} -> {v!r}")
        out[k] = v
    return out


def _parse_rename_args(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"--rename must be OLD=NEW, invalid: {raw!r}")
        old, new = raw.split("=", 1)
        if not old:
            raise SystemExit(f"--rename OLD must be non-empty: {raw!r}")
        out[old] = new
    return out


def _build_mapping(map_path: str | None, rename_pairs: list[str]) -> dict[str, str]:
    """
    Merge mapping: values from --map JSON first, then --rename entries
    (CLI wins when the same old name appears in both).
    """
    merged: dict[str, str] = {}
    if map_path:
        merged.update(_load_map_file(map_path))
    merged.update(_parse_rename_args(rename_pairs))
    return merged


def _apply_class_names_rename(
    class_names: dict, mapping: dict[str, str]
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """
    Returns (new_class_names_str_keys, list of (id_str, old_name, new_name) for changes).
    """
    if not isinstance(class_names, dict):
        raise ValueError("class_names must be a dict")
    new_cn: dict[str, str] = {}
    changes: list[tuple[str, str, str]] = []
    for id_str, name in class_names.items():
        sid = str(id_str)
        if not isinstance(name, str):
            raise ValueError(f"class_names[{sid!r}] must be a string, got {type(name).__name__}")
        new_name = mapping.get(name, name)
        new_cn[sid] = new_name
        if new_name != name:
            changes.append((sid, name, new_name))
    return new_cn, changes


def _detect_collision(class_names: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """Return list of (id_a, id_b, name) where two different ids map to the same name."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for sid, name in class_names.items():
        by_name[name].append(sid)
    collisions: list[tuple[str, str, str, str]] = []
    for name, ids in by_name.items():
        if len(ids) > 1:
            collisions.append((ids[0], ids[1], name, name))
    return collisions


def _default_out_path(input_path: str, output_dir: str) -> str:
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".ndjson":
        stem, ext = base, ""
    name = f"{stem}_renamed.ndjson"
    return os.path.join(output_dir, name)


def _process_file(
    input_path: str,
    mapping: dict[str, str],
    out_lines: list[str] | None,
    collect_changes: list[tuple[str, str, str]],
) -> tuple[int, int]:
    """
    Read input_path, apply renames to dataset lines, append JSON lines to out_lines if not None.
    Returns (raw_line_count, json_record_lines, dataset_records_with_renames).
    """
    lines_out: list[str] = []
    total_raw = 0
    total_records = 0
    datasets_updated = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            total_raw += 1
            stripped = line.strip()
            if not stripped:
                continue
            total_records += 1
            rec = json.loads(stripped)
            if rec.get("type") == "dataset":
                cn = rec.get("class_names") or {}
                new_cn, changes = _apply_class_names_rename(cn, mapping)
                cols = _detect_collision(new_cn)
                if cols:
                    id_a, id_b, n, _ = cols[0]
                    raise SystemExit(
                        f"{input_path}: collision after rename — class IDs {id_a} and {id_b} "
                        f"both use name {n!r}. Merge class IDs is not supported in v1."
                    )
                if changes:
                    datasets_updated += 1
                    for c in changes:
                        collect_changes.append(c)
                rec = {**rec, "class_names": new_cn}
                lines_out.append(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                lines_out.append(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_lines is not None:
        out_lines.extend(lines_out)

    return total_raw, total_records, datasets_updated


def _write_output(path: str, lines: list[str]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _prompt_multi_choice(prompt: str, options: list[str]) -> list[int]:
    """Comma-separated indices or 'a' for all; returns sorted 0-based indices."""
    print(f"\n{style(prompt, 1, 97)}")
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")
    print(f"  {style('[a]', 32, 1)} Select all")
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
                f"  Enter numbers 1-{len(options)} separated by commas, or 'a' for all",
                31,
            )
        )


def _interactive_wizard(cli_args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill inputs and optionally map/rename/output flags when no INPUT files on CLI.
    Interactive --rename rules are applied after CLI --rename (same OLD name: interactive wins).
    """
    print_header("NDJSON class rename (interactive)")
    ensure_data_and_output_dirs()

    ndjson_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.ndjson")))
    if not ndjson_paths:
        print(style(f"\n  No .ndjson files in {DATA_DIR}", 31, 1))
        sys.exit(1)

    display = [f"{os.path.basename(p)}  ({p})" for p in ndjson_paths]
    indices = _prompt_multi_choice("Select NDJSON file(s) to process:", display)
    inputs = [ndjson_paths[i] for i in indices]

    map_path = cli_args.map_path
    raw_map = input(
        f"\n{style('JSON map file path', 1, 97)} "
        f"{style('(optional, Enter to skip)', 2)}:\n{style('  > ', 35)}"
    ).strip()
    if raw_map:
        map_path = os.path.expanduser(raw_map)

    print(f"\n{style('Rename rules OLD=NEW', 1, 97)} — {style('one per line, empty line to finish', 2)}")
    extra_renames: list[str] = []
    while True:
        line = input(style("  > ", 35)).strip()
        if not line:
            break
        extra_renames.append(line)

    rename_all = list(cli_args.rename) + extra_renames

    if not map_path and not rename_all:
        print(style("\n  Need a JSON map file and/or at least one OLD=NEW rule.", 31))
        sys.exit(1)

    dry_ans = input(
        f"\n{style('Dry run only (no writes)?', 1, 97)} {style('[y/N]', 2)}\n{style('  > ', 35)}"
    ).strip().lower()
    dry_run = dry_ans in ("y", "yes")

    in_place = False
    force = False
    if not dry_run:
        ip_ans = input(
            f"\n{style('Overwrite files in-place?', 1, 97)} "
            f"{style('(requires typing YES)', 33)} {style('[y/N]', 2)}\n{style('  > ', 35)}"
        ).strip().lower()
        if ip_ans in ("y", "yes"):
            confirm = input(
                style("  Type YES to confirm in-place overwrite: ", 31)
            ).strip()
            if confirm == "YES":
                in_place = True
                force = True
            else:
                print(style("  In-place cancelled; will write to output directory.", 33))

    out_dir = cli_args.output_dir
    out_file = cli_args.output
    if not dry_run and not in_place:
        default_msg = OUTPUT_DIR
        od = input(
            f"\n{style('Output directory', 1, 97)} "
            f"{style(f'(Enter = {default_msg})', 2)}\n{style('  > ', 35)}"
        ).strip()
        out_dir = os.path.expanduser(od) if od else OUTPUT_DIR
        if len(inputs) != 1:
            out_file = None
        elif not out_file:
            single = input(
                f"\n{style('Single output file path', 1, 97)} "
                f"{style('(optional; Enter = <stem>_renamed.ndjson in output dir)', 2)}\n"
                f"{style('  > ', 35)}"
            ).strip()
            out_file = os.path.expanduser(single) if single else None

    ns = argparse.Namespace(
        inputs=inputs,
        map_path=map_path,
        rename=rename_all,
        output=out_file,
        output_dir=out_dir,
        dry_run=dry_run,
        in_place=in_place,
        force=force,
    )
    return ns


def _write_inplace(input_path: str, lines: list[str]) -> None:
    dirpath = os.path.dirname(os.path.abspath(input_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".rename_classes_", suffix=".ndjson", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, input_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_rename(args: argparse.Namespace) -> None:
    """Execute rename for all inputs in args (expects validated paths and mapping)."""
    mapping = _build_mapping(args.map_path, args.rename)

    ensure_data_and_output_dirs()

    for inp in args.inputs:
        if not os.path.isfile(inp):
            print(f"error: not a file: {inp}", file=sys.stderr)
            sys.exit(1)

    # Resolve output paths
    out_paths: dict[str, str] = {}
    if not args.dry_run and not args.in_place:
        out_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR
        if args.output:
            out_paths[args.inputs[0]] = args.output
        else:
            for inp in args.inputs:
                out_paths[inp] = _default_out_path(inp, out_dir)

    for inp in args.inputs:
        per_file_changes: list[tuple[str, str, str]] = []
        lines: list[str] = []
        try:
            _raw, total_recs, n_ds = _process_file(inp, mapping, lines, per_file_changes)
        except SystemExit as e:
            if str(e):
                print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        # Dedupe change lines for display (same id might appear if multiple dataset lines — unusual)
        seen: set[tuple[str, str, str]] = set()
        unique_changes: list[tuple[str, str, str]] = []
        for t in per_file_changes:
            if t not in seen:
                seen.add(t)
                unique_changes.append(t)

        if args.dry_run:
            print(
                f"{inp} ({total_recs} JSON lines, {n_ds} dataset header(s) with renames)"
            )
            for sid, old, new in unique_changes:
                print(f"  class {sid}: {old!r} -> {new!r}")
            if not unique_changes:
                print("  (no class_names values matched the mapping)")
            continue

        if args.in_place:
            _write_inplace(inp, lines)
            print(f"updated in-place: {inp}")
        else:
            outp = out_paths[inp]
            _write_output(outp, lines)
            print(f"wrote: {outp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename class name strings in NDJSON dataset headers (class_names only). "
        "Run with no INPUT arguments for interactive mode (scans data/*.ndjson)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        metavar="INPUT.ndjson",
        help="One or more NDJSON files (omit for interactive wizard)",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        metavar="FILE.json",
        help='JSON object {"OldName":"new_name",...} (exact old names)',
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename rule (repeatable). Overrides --map for the same OLD name.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE.ndjson",
        help="Output path (exactly one INPUT required)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help=f"Write <stem>_renamed.ndjson here (default if omitted: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames per file; do not write",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each INPUT (requires --force)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --in-place overwrite",
    )
    args = parser.parse_args()

    if not args.inputs:
        args = _interactive_wizard(args)
    else:
        if not args.map_path and not args.rename:
            parser.error("Provide at least one of --map or --rename")

    if args.in_place and not args.force:
        parser.error("--in-place requires --force")

    if args.in_place and (args.output or args.output_dir):
        parser.error("--in-place cannot be combined with --output or --output-dir")

    if args.output and len(args.inputs) != 1:
        parser.error("--output requires exactly one INPUT")

    _run_rename(args)


if __name__ == "__main__":
    main()
