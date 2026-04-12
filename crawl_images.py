#!/usr/bin/env python3
"""Download images from the web by keyword — interactive CLI powered by icrawler."""

import logging
import os
import re
import sys
from datetime import datetime

from cli_utils import (
    count_images_in_dir,
    fmt_size,
    print_header,
    prompt_choice,
    style,
    zip_output_dir,
)
from paths import OUTPUT_DIR, ensure_data_and_output_dirs

_ENGINES = {
    "bing": "Bing  (recommended — stable, rarely blocked)",
    "google": "Google  (may be rate-limited)",
}

_SIZE_FILTERS = {
    "": "Any size",
    "large": "Large",
    "medium": "Medium",
    "small": "Small",
}

_TYPE_FILTERS = {
    "": "Any type",
    "photo": "Photo",
    "clipart": "Clipart",
    "lineart": "Line art",
    "animated": "Animated / GIF",
}


def _sanitize_folder_name(keyword: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", keyword.lower()).strip()
    slug = re.sub(r"[\s]+", "_", slug)
    return slug[:80] or "images"


def _prompt_positive_int(label: str, default: int) -> int:
    while True:
        raw = input(style(f"  {label} (default {default}): ", 2)).strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if v <= 0:
                print(style("  Enter a number > 0", 31))
                continue
            return v
        except ValueError:
            print(style("  Enter a whole number, e.g. 100", 31))


def _build_crawler(engine: str, out_dir: str):
    """Return an icrawler instance configured for *engine*."""
    if engine == "google":
        from icrawler.builtin import GoogleImageCrawler  # type: ignore[import-untyped]

        return GoogleImageCrawler(
            storage={"root_dir": out_dir},
            log_level=logging.WARNING,
        )

    from icrawler.builtin import BingImageCrawler  # type: ignore[import-untyped]

    return BingImageCrawler(
        storage={"root_dir": out_dir},
        log_level=logging.WARNING,
    )


def _build_filters(engine: str, size_key: str, type_key: str) -> dict | None:
    if not size_key and not type_key:
        return None

    if engine == "bing":
        filters: dict = {}
        if size_key:
            filters["size"] = size_key
        if type_key:
            filters["type"] = type_key
        return filters

    if engine == "google":
        filters = {}
        if size_key:
            filters["size"] = size_key
        if type_key:
            filters["type"] = type_key
        return filters

    return None


def _dir_total_size(dirpath: str) -> int:
    total = 0
    for name in os.listdir(dirpath):
        fp = os.path.join(dirpath, name)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total


def main():
    print_header("Crawl images from the web")
    ensure_data_and_output_dirs()

    # -- keyword ---------------------------------------------------------------
    print(f"\n{style('Search keyword', 1, 97)}")
    while True:
        keyword = input(style("  Keyword: ", 35)).strip()
        if keyword:
            break
        print(style("  Keyword cannot be empty.", 31))

    # -- engine ----------------------------------------------------------------
    engine_keys = list(_ENGINES.keys())
    engine_idx = prompt_choice("Search engine:", [_ENGINES[k] for k in engine_keys])
    engine = engine_keys[engine_idx]

    # -- max images ------------------------------------------------------------
    max_images = _prompt_positive_int("Max images to download", 100)

    # -- size filter -----------------------------------------------------------
    size_keys = list(_SIZE_FILTERS.keys())
    size_idx = prompt_choice("Image size filter:", [_SIZE_FILTERS[k] for k in size_keys])
    size_key = size_keys[size_idx]

    # -- type filter -----------------------------------------------------------
    type_keys = list(_TYPE_FILTERS.keys())
    type_idx = prompt_choice("Image type filter:", [_TYPE_FILTERS[k] for k in type_keys])
    type_key = type_keys[type_idx]

    # -- output folder ---------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_folder_name(keyword)
    default_subdir = f"crawl_{slug}_{ts}"

    raw_sub = input(
        style(f"\n  Output subfolder (empty = '{default_subdir}'): ", 2)
    ).strip()
    subdir = raw_sub if raw_sub else default_subdir
    subdir = subdir.replace(os.sep, "_").strip("/.")
    if not subdir:
        print(style("  Invalid folder name.", 31))
        sys.exit(1)

    out_dir = os.path.join(OUTPUT_DIR, subdir)

    existing = count_images_in_dir(out_dir)
    if existing > 0:
        print(
            style(
                f"\n  Folder already has {existing} image(s). "
                f"New downloads will be added alongside existing files.",
                33,
            )
        )
        ok = input(style("  Use this folder anyway? (y/n): ", 35)).strip().lower()
        if ok not in ("y", "yes"):
            print(style("  Cancelled.", 33))
            sys.exit(0)

    # -- summary ---------------------------------------------------------------
    size_label = _SIZE_FILTERS.get(size_key, "any")
    type_label = _TYPE_FILTERS.get(type_key, "any")
    engine_label = engine.capitalize()

    print(
        f"\n  {style('Summary', 1, 97)}\n"
        f"  {style('Keyword:', 2)}    {style(keyword, 36)}\n"
        f"  {style('Engine:', 2)}     {style(engine_label, 36)}\n"
        f"  {style('Max images:', 2)} {style(str(max_images), 36)}\n"
        f"  {style('Size:', 2)}       {style(size_label, 36)}\n"
        f"  {style('Type:', 2)}       {style(type_label, 36)}\n"
        f"  {style('Output:', 2)}     {style(out_dir, 36)}"
    )

    make_zip = (
        input(style("\n  Create a .zip archive after download? (y/n): ", 35))
        .strip()
        .lower()
        in ("y", "yes")
    )

    confirm = input(style("  Start download? (y/n): ", 35)).strip().lower()
    if confirm not in ("y", "yes"):
        print(style("  Cancelled.", 33))
        sys.exit(0)

    # -- download --------------------------------------------------------------
    print(f"\n  {style('Downloading...', 1, 93)} (this may take a while)\n")

    try:
        crawler = _build_crawler(engine, out_dir)
    except ImportError:
        print(style("  icrawler is not installed. Run:", 31, 1))
        print(style("    pip install icrawler", 33))
        sys.exit(1)

    filters = _build_filters(engine, size_key, type_key)
    crawl_kwargs: dict = {"keyword": keyword, "max_num": max_images}
    if filters:
        crawl_kwargs["filters"] = filters

    crawler.crawl(**crawl_kwargs)

    # -- results ---------------------------------------------------------------
    saved = count_images_in_dir(out_dir)
    total_bytes = _dir_total_size(out_dir) if saved else 0

    print_header("Done")
    print(f"  {style('Output:', 2)}       {style(out_dir, 32, 1)}")
    print(f"  {style('Images saved:', 2)} {style(str(saved), 97, 1)}")
    if total_bytes:
        print(f"  {style('Total size:', 2)}   {style(fmt_size(total_bytes), 97)}")

    if make_zip:
        if saved <= 0:
            print(style("  Skipped .zip (no images downloaded).", 33))
        else:
            print(f"  {style('Creating archive...', 2)}")
            zpath, zerr = zip_output_dir(out_dir)
            if zerr:
                print(style(f"  Could not create .zip: {zerr}", 31))
            elif zpath:
                zip_sz = os.path.getsize(zpath)
                print(f"  {style('Archive:', 2)}      {style(zpath, 32, 1)}  ({fmt_size(zip_sz)})")

    if saved == 0:
        print(
            style(
                "\n  No images were downloaded. Possible causes:\n"
                "    - No internet connection\n"
                "    - Search returned no results for this keyword\n"
                "    - Engine rate-limited this request (try Bing)\n",
                33,
            )
        )

    print()


if __name__ == "__main__":
    main()
