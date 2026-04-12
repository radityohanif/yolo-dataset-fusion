#!/usr/bin/env python3
"""Ekstraksi frame (gambar) dari video — CLI interaktif, output ke ./output/<subdir>/."""

import glob
import os
import sys
from datetime import datetime

import cv2

from paths import DATA_DIR, OUTPUT_DIR, ensure_data_and_output_dirs

VIDEO_GLOBS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.m4v")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
JPEG_QUALITY = 92


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
    print(f"\n{style(prompt, 1, 97)}")
    for i, opt in enumerate(options, 1):
        idx_s = style(f"[{i}]", 36, 1)
        print(f"  {idx_s} {opt}")
    while True:
        raw = input(style("  > ", 35)).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(style(f"  Masukkan angka 1-{len(options)}", 31))


def _fmt_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


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


def count_images_in_dir(dirpath: str) -> int:
    if not os.path.isdir(dirpath):
        return 0
    n = 0
    for name in os.listdir(dirpath):
        low = name.lower()
        if any(low.endswith(s) for s in IMAGE_SUFFIXES):
            n += 1
    return n


def prompt_float_minutes(label: str, empty_hint: str, max_min: float | None) -> float | None:
    """Return menit (float) atau None jika pengguna mengosongkan (arti: batas default)."""
    extra = f", maks ~{max_min:.3f} menit" if max_min is not None else ""
    while True:
        raw = input(
            style(f"  {label} (kosong = {empty_hint}{extra}): ", 2)
        ).strip()
        if not raw:
            return None
        try:
            v = float(raw.replace(",", "."))
            if v < 0:
                print(style("  Masukkan angka >= 0", 31))
                continue
            if max_min is not None and v > max_min + 1e-6:
                print(style(f"  Nilai melebihi durasi video (~{max_min:.3f} menit)", 31))
                continue
            return v
        except ValueError:
            print(style("  Masukkan angka desimal (menit), contoh: 0 atau 1.5", 31))


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


def extract_frames(
    video_path: str,
    out_dir: str,
    t_start_sec: float,
    t_end_sec: float | None,
    interval_sec: float,
) -> tuple[int, str | None]:
    """
    Seeking via CAP_PROP_POS_MSEC is not always frame-accurate for all codecs.
    t_end_sec None = sampai video habis (EOF).
    Returns (saved_count, error_message_or_None).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Tidak bisa membuka video (codec tidak didukung atau file rusak)"

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
            fname = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(
                fname,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            last_write_t = t

    cap.release()
    return saved, None


def main():
    print_header("Ekstraksi frame dari video")
    ensure_data_and_output_dirs()

    paths = scan_video_paths()
    if not paths:
        print(style(f"\n  Tidak ada file video di {DATA_DIR}", 31, 1))
        print(style("  Letakkan file .mp4, .mov, .mkv, .avi, .webm, atau .m4v di folder data.", 2))
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
            line = f"{base}  ({_fmt_size(size_b)}, ~{n} frames, durasi/fps tak pasti)"
        else:
            line = f"{base}  ({_fmt_size(size_b)}, metadata terbatas)"
        options.append(line)
        meta.append((p, fps, n, dur))

    idx = prompt_choice("Pilih video yang ingin diekstrak:", options)
    video_path, fps, n_frames, duration_sec = meta[idx][0], meta[idx][1], meta[idx][2], meta[idx][3]

    cap_check = cv2.VideoCapture(video_path)
    if not cap_check.isOpened():
        print(style("\n  Gagal membuka file video. Coba codec lain atau konversi ke H.264/AAC.", 31))
        sys.exit(1)
    cap_check.release()

    stem = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_subdir = f"frames_{stem}_{ts}"
    raw_sub = input(
        style(
            f"\n  Nama folder di bawah output (kosongkan untuk '{default_subdir}'): ",
            2,
        )
    ).strip()
    subdir = raw_sub if raw_sub else default_subdir
    subdir = subdir.replace(os.sep, "_").strip("/.")
    if not subdir:
        print(style("  Nama folder tidak valid.", 31))
        sys.exit(1)

    out_dir = os.path.join(OUTPUT_DIR, subdir)
    existing = count_images_in_dir(out_dir)
    if existing > 0:
        print(
            style(
                f"\n  Folder sudah berisi {existing} gambar. "
                f"Melanjutkan akan menulis ulang frame_000001.jpg dst. sesuai urutan ekstraksi.",
                33,
            )
        )
        ok = input(style("  Tetap gunakan folder ini? (y/n): ", 35)).strip().lower()
        if ok not in ("y", "yes"):
            print(style("  Dibatalkan.", 33))
            sys.exit(0)

    dur_min = (duration_sec / 60.0) if duration_sec else None

    print(
        f"\n  {style('Rentang waktu (menit).', 1, 97)} "
        f"{style('CAP_PROP_POS_MSEC tidak selalu tepat per-frame untuk semua codec.', 2)}"
    )
    start_min = prompt_float_minutes("Mulai dari menit", "dari awal", dur_min)
    end_min = prompt_float_minutes("Sampai menit", "sampai akhir", dur_min)

    t_start = 0.0 if start_min is None else start_min * 60.0

    if end_min is None:
        if duration_sec is not None:
            t_end: float = duration_sec
        else:
            t_end = float("inf")
    else:
        t_end = end_min * 60.0
        if duration_sec is not None:
            t_end = min(t_end, duration_sec)

    if duration_sec is not None:
        t_start = min(max(0.0, t_start), duration_sec)

    if duration_sec is None and end_min is None:
        print(style("  Durasi video tidak terbaca; ekstraksi berjalan sampai akhir file.", 33))

    if t_end != float("inf") and t_start >= t_end - 1e-9:
        print(style("  Rentang waktu tidak valid (awal harus sebelum akhir).", 31))
        sys.exit(1)

    while True:
        raw_iv = input(
            style(
                "  Ambil 1 frame setiap berapa detik? (0 = setiap frame dalam rentang): ",
                2,
            )
        ).strip()
        if not raw_iv:
            interval_sec = 0.0
            break
        try:
            interval_sec = float(raw_iv.replace(",", "."))
            if interval_sec < 0:
                print(style("  Masukkan angka >= 0", 31))
                continue
            break
        except ValueError:
            print(style("  Masukkan angka, contoh: 0 atau 1.5", 31))

    eff_fps = fps if fps and fps > 0 else 0.0
    if t_end == float("inf"):
        est_msg = "perkiraan tidak tersedia (durasi tak diketahui)"
    else:
        est = estimate_saved_frames(t_start, t_end, eff_fps, interval_sec)
        est_msg = f"~{est} file" if est else "0 file"

    print(
        f"\n  {style('Ringkasan:', 1, 97)}\n"
        f"  {style('Video:', 2)} {style(video_path, 36)}\n"
        f"  {style('Output:', 2)} {style(out_dir, 36)}\n"
        f"  {style('Rentang:', 2)} {t_start:.2f}s – "
        f"{('∞' if t_end == float('inf') else f'{t_end:.2f}s')}\n"
        f"  {style('Interval simpan:', 2)} "
        f"{'setiap frame' if interval_sec <= 0 else f'{interval_sec} s'}\n"
        f"  {style('Perkiraan jumlah gambar:', 2)} {est_msg}"
    )

    confirm = input(style("  Lanjutkan? (y/n): ", 35)).strip().lower()
    if confirm not in ("y", "yes"):
        print(style("  Dibatalkan.", 33))
        sys.exit(0)

    t_end_arg = None if t_end == float("inf") else t_end
    saved, err = extract_frames(video_path, out_dir, t_start, t_end_arg, interval_sec)
    if err:
        print(style(f"\n  {err}", 31))
        sys.exit(1)

    print_header("Selesai!")
    print(f"  {style('Output:', 2)} {style(out_dir, 32, 1)}")
    print(f"  {style('Total gambar:', 2)} {style(str(saved), 97, 1)}")
    print()


if __name__ == "__main__":
    main()
