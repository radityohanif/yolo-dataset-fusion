# YOLO Dataset Fusion

Small CLI utilities for working with YOLO-style detection data:

1. **`merge.py`** — Merge multiple **NDJSON** datasets, align class IDs by **class name**, optionally **balance** annotations, and write one merged file.
2. **`extract_frames.py`** — Extract **JPEG frames** from videos under `data/` into `output/<subfolder>/` (time range + interval).

Typical NDJSON sources are [Ultralytics](https://docs.ultralytics.com/) platform exports or any exporter that uses the same record shape.

## Requirements

- **Python 3.10+** (syntax and typing used across the scripts).
- **`merge.py`** — standard library only; no `pip` packages required.
- **`extract_frames.py`** — needs **OpenCV** (`cv2`) for decoding video.

Install optional dependencies (OpenCV) from the repo root:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

| Path | Purpose |
|------|---------|
| `paths.py` | Shared `DATA_DIR` / `OUTPUT_DIR` and `ensure_data_and_output_dirs()` used by both CLIs. |
| `data/` | Inputs: `*.ndjson` for merge; videos (`*.mp4`, `*.mov`, `*.mkv`, `*.avi`, `*.webm`, `*.m4v`) for frame extraction. Large or private files are usually git-ignored. |
| `output/` | Merged NDJSON and extracted frame folders. |
| `merge.py` | Interactive NDJSON merge and class balancing. |
| `extract_frames.py` | Interactive video → JPEG frame export. |

## `data/` and `output/` folders

You **do not** need to create `data/` or `output/` manually. On startup, **`merge.py`** and **`extract_frames.py`** call `ensure_data_and_output_dirs()` from `paths.py`, which creates both directories next to the scripts if they are missing.

Run the tools from the **repository root** so Python can resolve the `paths` module (e.g. `python3 merge.py`, not from an unrelated working directory without adjusting `PYTHONPATH`).

## `merge.py` — merge NDJSON datasets

```bash
python3 merge.py
```

Flow:

1. Ensure `data/` and `output/` exist, then scan **`data/*.ndjson`**. Exits with a message if no files are found (after creating empty `data/` you can add exports there).
2. **Select** one or more datasets (comma-separated indices, or `a` for all).
3. **Remap** class IDs into one global map keyed by **class name** (same name → same ID everywhere).
4. Show **annotation and image counts per class**, and how many images have **no** boxes.
5. Choose to **include** unannotated images (as negatives) or **skip** them.
6. Choose a **balancing mode**:
   - **No balance** — use all annotated images (plus optional unannotated).
   - **Equal balance** — undersample toward the smallest class (greedy image selection).
   - **Custom ratio** — target shares of total annotations as percentages.
   - **Custom count** — target annotation counts per class.
7. **Preview** the distribution after balancing.
8. **Output filename** — default `merged_YYYYMMDD_HHMMSS.ndjson` under `output/` if you press Enter.
9. **Confirm** before writing.

The merged file begins with a new `dataset` header (`class_names` unified, `version`: `merged`), then all selected `image` lines with remapped IDs.

### NDJSON format (merge)

1. **Dataset** line: `"type": "dataset"`, with `class_names` as string keys → names (e.g. `"0": "person"`).
2. **Image** lines: `"type": "image"`, optional `split`, dimensions, and `annotations.boxes` as YOLO **normalized** boxes: `[class_id, x_center, y_center, width, height]` in \([0, 1]\).

Images may omit boxes; merge can still optionally keep them.

## `extract_frames.py` — video to frames

Requires `pip install -r requirements.txt` (OpenCV).

Put one or more supported videos in **`data/`**, then:

```bash
python3 extract_frames.py
```

Flow:

1. Ensure `data/` and `output/` exist, then list videos in `data/`. Exits with a message if none are found.
2. **Pick a video** from the list.
3. **Subfolder name** under `output/` (default suggests `frames_<stem>_<timestamp>`). Unsafe path characters are normalized.
4. If the target folder already has images, you are warned and can cancel.
5. **Time range** in minutes (start / end; empty uses full clip when duration is known). Seeking uses OpenCV timestamps and may be imperfect on some codecs.
6. **Interval** — seconds between saved frames (`0` = save every frame in the chosen range).
7. Summary and **confirm** before extraction.
8. Writes **`frame_000001.jpg`**, … with fixed JPEG quality (see `JPEG_QUALITY` in the script).

## Terminal colors

Both scripts respect:

- **`NO_COLOR`** — any non-empty value disables ANSI colors.
- **`FORCE_COLOR`** — any non-empty value forces colors even when stdout is not a TTY.

## License

Add a `LICENSE` file if you publish this repository; none is included in the default tree.
