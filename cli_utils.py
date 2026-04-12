"""Shared CLI helpers — coloured output, prompts, formatting."""

import os
import shutil
import sys


def _use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    return sys.stdout.isatty()


def style(text: str, *codes: int) -> str:
    if not _use_color() or not codes:
        return text
    seq = ";".join(str(c) for c in codes)
    return f"\033[{seq}m{text}\033[0m"


def print_header(title: str):
    width = 60
    bar = "=" * width
    bar_styled = style(bar, 36) if _use_color() else bar
    title_styled = style(f"  {title}", 1, 96) if _use_color() else f"  {title}"
    print(f"\n{bar_styled}")
    print(title_styled)
    print(bar_styled)


def prompt_choice(prompt: str, options: list[str]) -> int:
    """Display numbered options and return the 0-based index chosen by the user."""
    print(f"\n{style(prompt, 1, 97)}")
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")
    while True:
        raw = input(style("  > ", 35)).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(style(f"  Enter a number from 1 to {len(options)}", 31))


def fmt_size(num: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


def count_images_in_dir(dirpath: str) -> int:
    _suffixes = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not os.path.isdir(dirpath):
        return 0
    n = 0
    for name in os.listdir(dirpath):
        if any(name.lower().endswith(s) for s in _suffixes):
            n += 1
    return n


def zip_output_dir(out_dir: str) -> tuple[str | None, str | None]:
    """
    Create <out_dir>.zip; archive entries are files relative to out_dir.
    Returns (zip_path, error_message_or_None).
    """
    zip_path = out_dir + ".zip"
    try:
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        arc = shutil.make_archive(out_dir, "zip", root_dir=out_dir)
        return arc, None
    except OSError as e:
        return None, str(e)
