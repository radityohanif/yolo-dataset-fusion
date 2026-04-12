"""Lokasi folder proyek; pastikan ada sebelum skrip membaca/menulis."""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
OUTPUT_DIR = os.path.join(_ROOT, "output")


def ensure_data_and_output_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
