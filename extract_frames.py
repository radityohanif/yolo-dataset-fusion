#!/usr/bin/env python3
"""Extract image frames from video — interactive CLI; writes JPEGs under ./output/<subdir>/."""

import glob
import os
import sys
from datetime import datetime

import cv2

from cli_utils import (
    count_images_in_dir,
    fmt_size as _fmt_size,
    print_header,
    style,
    zip_output_dir,
)
from paths import DATA_DIR, OUTPUT_DIR, ensure_data_and_output_dirs

VIDEO_GLOBS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.m4v")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
JPEG_QUALITY = 92


def scan_video_paths() -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for pattern in VIDEO_GLOBS:
        for p in glob.glob(os.path.join(DATA_DIR, pattern)):
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap):
                seen.add(ap)
                paths.append(ap)
    paths.sort(key=lambda x: os.path.basename(x).lower())
    return paths


def probe_video(path: str) -> tuple[float | None, int | None, float | None]:
    """
    Returns (fps, frame_count, duration_sec).
    duration_sec may be None if metadata unreliable; fps may be 0 or None-like from OpenCV.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps is None or fps <= 0 or n <= 0:
        return fps or 0.0, n if n > 0 else None, None
    duration = n / fps
    if duration <= 0:
        return fps, n, None
    return float(fps), n, float(duration)



def prompt_float_minutes(label: str, empty_hint: str, max_min: float | None) -> float | None:
    """Return minutes as float, or None if the user leaves the input empty (use default bound)."""
    extra = f", max ~{max_min:.3f} min" if max_min is not None else ""
    while True:
        raw = input(
            style(f"  {label} (empty = {empty_hint}{extra}): ", 2)
        ).strip()
        if not raw:
            return None
        try:
            v = float(raw.replace(",", "."))
            if v < 0:
                print(style("  Enter a number >= 0", 31))
                continue
            if max_min is not None and v > max_min + 1e-6:
                print(style(f"  Value exceeds video duration (~{max_min:.3f} min)", 31))
                continue
            return v
        except ValueError:
            print(style("  Enter a decimal number (minutes), e.g. 0 or 1.5", 31))


def estimate_saved_frames(
    t0: float,
    t1: float,
    fps: float,
    interval_sec: float,
) -> int:
    span = max(0.0, t1 - t0)
    if span <= 0:
        return 0
    if interval_sec <= 0:
        if fps > 0:
            return int(span * fps) + 1
        return 0
    return int(span / interval_sec) + 1


def parse_video_selection(raw: str, n: int) -> list[int] | None:
    """
    Parse user input into 0-based indices.
    Accepts comma-separated 1-based indices (e.g. 1,2,5), duplicates removed in order,
    or '*' / 'all' (case-insensitive) for every video.
    """
    s = raw.strip().lower()
    if not s:
        return None
    if s in ("*", "all"):
        return list(range(n))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    out: list[int] = []
    seen: set[int] = set()
    for p in parts:
        if not p.isdigit():
            return None
        v = int(p)
        if not (1 <= v <= n):
            return None
        i = v - 1
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out if out else None


def count_images_under_tree(parent: str) -> int:
    """Count image files in parent and in its immediate subfolders."""
    n = count_images_in_dir(parent)
    if not os.path.isdir(parent):
        return n
    for name in os.listdir(parent):
        sub = os.path.join(parent, name)
        if os.path.isdir(sub):
            n += count_images_in_dir(sub)
    return n


def extract_frames(
    video_path: str,
    out_dir: str,
    t_start_sec: float,
    t_end_sec: float | None,
    interval_sec: float,
    frame_index_start: int = 1,
) -> tuple[int, str | None]:
    """
    Seeking via CAP_PROP_POS_MSEC is not always frame-accurate for all codecs.
    t_end_sec None means read until end of file (EOF).
    Returns (saved_count, error_message_or_None).
    Frame files are named frame_{index:06d}.jpg with index starting at frame_index_start.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Could not open video (unsupported codec or corrupt file)"

    start_ms = max(0.0, t_start_sec) * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    last_write_t = -1e18

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < t_start_sec - 0.05:
            continue
        if t_end_sec is not None and t > t_end_sec + 0.05:
            break

        if interval_sec <= 0:
            do_save = True
        else:
            do_save = (t - last_write_t) >= interval_sec - 1e-6

        if do_save:
            saved += 1
            idx = frame_index_start + saved - 1
            fname = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(
                fname,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            last_write_t = t

    cap.release()
    return saved, None



def main():
    print_header("Extract frames from video")
    ensure_data_and_output_dirs()

    paths = scan_video_paths()
    if not paths:
        print(style(f"\n  No video files in {DATA_DIR}", 31, 1))
        print(style("  Add .mp4, .mov, .mkv, .avi, .webm, or .m4v files to the data folder.", 2))
        sys.exit(1)

    meta: list[tuple[str, float | None, int | None, float | None]] = []
    options: list[str] = []
    for p in paths:
        fps, n, dur = probe_video(p)
        size_b = os.path.getsize(p)
        base = os.path.basename(p)
        if dur is not None and fps:
            line = f"{base}  ({_fmt_size(size_b)}, ~{dur:.1f}s, ~{fps:.2f} fps, {n} frames)"
        elif n:
            line = f"{base}  ({_fmt_size(size_b)}, ~{n} frames, duration/fps uncertain)"
        else:
            line = f"{base}  ({_fmt_size(size_b)}, limited metadata)"
        options.append(line)
        meta.append((p, fps, n, dur))

    print(f"\n{style('Select video(s) to extract:', 1, 97)}")
    print(
        style(
            "  Enter comma-separated numbers (e.g. 1,2,5), or type * or all for every video.",
            2,
        )
    )
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")

    while True:
        raw_sel = input(style("  > ", 35)).strip()
        sel = parse_video_selection(raw_sel, len(options))
        if sel is not None:
            break
        print(style(f"  Use 1–{len(options)}, comma-separated list, or * / all.", 31))

    selected_meta = [meta[i] for i in sel]
    n_sel = len(sel)

    for vp, _, _, _ in selected_meta:
        cap_check = cv2.VideoCapture(vp)
        if not cap_check.isOpened():
            print(
                style(
                    f"\n  Failed to open: {vp}\n"
                    "  Try another codec or convert to H.264/AAC.",
                    31,
                )
            )
            sys.exit(1)
        cap_check.release()

    layout_merged = True
    if n_sel > 1:
        print(
            f"\n{style('Output layout for multiple videos', 1, 97)}\n"
            f"  {style('[1]', 36, 1)} One folder — "
            "frame_000001.jpg … in order across all selected videos\n"
            f"  {style('[2]', 36, 1)} One parent folder — subfolder per video "
            "(each with its own frame_000001.jpg …)"
        )
        while True:
            raw_l = input(style("  Choose layout [1/2] (default 1): ", 35)).strip() or "1"
            if raw_l == "1":
                layout_merged = True
                break
            if raw_l == "2":
                layout_merged = False
                break
            print(style("  Enter 1 or 2", 31))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if n_sel == 1:
        stem0 = os.path.splitext(os.path.basename(selected_meta[0][0]))[0]
        default_subdir = f"frames_{stem0}_{ts}"
    elif layout_merged:
        default_subdir = f"frames_merged_{n_sel}v_{ts}"
    else:
        default_subdir = f"frames_batch_{n_sel}v_{ts}"

    raw_sub = input(
        style(
            f"\n  Subfolder under output (leave empty for '{default_subdir}'): ",
            2,
        )
    ).strip()
    subdir = raw_sub if raw_sub else default_subdir
    subdir = subdir.replace(os.sep, "_").strip("/.")
    if not subdir:
        print(style("  Invalid folder name.", 31))
        sys.exit(1)

    out_dir = os.path.join(OUTPUT_DIR, subdir)
    existing = (
        count_images_in_dir(out_dir)
        if n_sel == 1 or layout_merged
        else count_images_under_tree(out_dir)
    )
    if existing > 0:
        print(
            style(
                f"\n  Output path already has {existing} image(s). "
                "Continuing may overwrite frame_######.jpg in extraction order.",
                33,
            )
        )
        ok = input(style("  Use this folder anyway? (y/n): ", 35)).strip().lower()
        if ok not in ("y", "yes"):
            print(style("  Cancelled.", 33))
            sys.exit(0)

    known_durs = [m[3] for m in selected_meta if m[3] is not None]
    dur_min = (min(known_durs) / 60.0) if known_durs else None

    print(
        f"\n  {style('Time range (minutes)', 1, 97)} "
        f"{style('Same range for each video; end is clipped to each file length.', 2)}"
    )
    print(
        style("  CAP_PROP_POS_MSEC is not always frame-accurate for every codec.", 2)
    )
    start_min = prompt_float_minutes("Start at minute", "from beginning", dur_min)
    end_min = prompt_float_minutes("End at minute", "through end of each file", dur_min)

    t_start_sec = 0.0 if start_min is None else start_min * 60.0
    user_end_sec: float | None = None if end_min is None else end_min * 60.0

    def span_positive_for_video(dur: float | None) -> bool:
        ts = t_start_sec
        if dur is not None:
            ts = min(max(0.0, ts), dur)
        if user_end_sec is None:
            if dur is None:
                return True
            return ts < dur - 1e-9
        te = user_end_sec
        if dur is not None:
            te = min(te, dur)
        return ts < te - 1e-9

    if not any(span_positive_for_video(m[3]) for m in selected_meta):
        print(
            style(
                "  Invalid time range (start must be before end for at least one video).",
                31,
            )
        )
        sys.exit(1)

    if any(m[3] is None for m in selected_meta) and end_min is None:
        print(
            style(
                "  Some videos have unknown duration; those run until EOF.",
                33,
            )
        )

    while True:
        raw_iv = input(
            style(
                "  Save one frame every how many seconds? (0 = every frame in range): ",
                2,
            )
        ).strip()
        if not raw_iv:
            interval_sec = 0.0
            break
        try:
            interval_sec = float(raw_iv.replace(",", "."))
            if interval_sec < 0:
                print(style("  Enter a number >= 0", 31))
                continue
            break
        except ValueError:
            print(style("  Enter a number, e.g. 0 or 1.5", 31))

    est_parts: list[int] = []
    est_unknown = False
    for _, fp, _, dur in selected_meta:
        ts = t_start_sec
        if dur is not None:
            ts = min(max(0.0, ts), dur)
        if user_end_sec is None:
            if dur is None:
                est_unknown = True
                continue
            te_use = dur
        elif dur is None:
            te_use = user_end_sec
        else:
            te_use = min(user_end_sec, dur)
        eff_fp = fp if fp and fp > 0 else 0.0
        est_parts.append(estimate_saved_frames(ts, te_use, eff_fp, interval_sec))

    if est_unknown and not est_parts:
        est_msg = "estimate unavailable (duration unknown)"
    elif est_unknown:
        est_msg = f"~{sum(est_parts)}+ files (partial; some duration unknown)"
    else:
        s_est = sum(est_parts)
        est_msg = f"~{s_est} files" if s_est else "0 files"

    end_range_txt = (
        "through end of each file"
        if user_end_sec is None
        else f"≤ {user_end_sec:.2f}s (clipped per file)"
    )

    print(f"\n  {style('Summary', 1, 97)}")
    for k, (vp, _, _, _) in enumerate(selected_meta, 1):
        print(f"  {style(f'Video {k}:', 2)} {style(vp, 36)}")
    if n_sel > 1:
        layout_txt = "single merged folder" if layout_merged else "subfolder per video"
        print(f"  {style('Layout:', 2)} {layout_txt}")
    print(f"  {style('Output:', 2)} {style(out_dir, 36)}")
    print(
        f"  {style('Range:', 2)} start {t_start_sec:.2f}s — end {end_range_txt}\n"
        f"  {style('Save interval:', 2)} "
        f"{'every frame' if interval_sec <= 0 else f'{interval_sec} s'}\n"
        f"  {style('Estimated image count:', 2)} {est_msg}"
    )

    make_zip = (
        input(
            style(
                "  Also create a .zip of the output folder after extraction? (y/n): ",
                35,
            )
        )
        .strip()
        .lower()
        in ("y", "yes")
    )

    confirm = input(style("  Continue? (y/n): ", 35)).strip().lower()
    if confirm not in ("y", "yes"):
        print(style("  Cancelled.", 33))
        sys.exit(0)

    next_frame_idx = 1
    total_saved = 0
    for video_path, _, _, duration_sec in selected_meta:
        t_start_v = t_start_sec
        if duration_sec is not None:
            t_start_v = min(max(0.0, t_start_sec), duration_sec)

        if user_end_sec is None:
            t_end_arg = None
        else:
            t_end_arg = user_end_sec
            if duration_sec is not None:
                t_end_arg = min(t_end_arg, duration_sec)

        if t_end_arg is not None and t_start_v >= t_end_arg - 1e-9:
            print(style(f"\n  Skipping (empty range): {video_path}", 33))
            continue

        vid_out = out_dir
        if n_sel > 1 and not layout_merged:
            vst = os.path.splitext(os.path.basename(video_path))[0]
            vid_out = os.path.join(out_dir, vst.replace(os.sep, "_"))

        os.makedirs(vid_out, exist_ok=True)

        chain_indices = (n_sel == 1) or layout_merged
        fstart = next_frame_idx if chain_indices else 1

        saved, err = extract_frames(
            video_path,
            vid_out,
            t_start_v,
            t_end_arg,
            interval_sec,
            frame_index_start=fstart,
        )
        if err:
            print(style(f"\n  {err}", 31))
            sys.exit(1)
        total_saved += saved
        if chain_indices:
            next_frame_idx += saved

    print_header("Done")
    print(f"  {style('Output:', 2)} {style(out_dir, 32, 1)}")
    print(f"  {style('Total images saved:', 2)} {style(str(total_saved), 97, 1)}")
    if make_zip:
        if total_saved <= 0:
            print(style("  Skipped .zip (no frames were saved).", 33))
        else:
            zpath, zerr = zip_output_dir(out_dir)
            if zerr:
                print(style(f"  Could not create .zip: {zerr}", 31))
            elif zpath:
                print(f"  {style('Archive:', 2)} {style(zpath, 32, 1)}")
    print()


if __name__ == "__main__":
    main()
