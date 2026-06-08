#!/usr/bin/env python
"""
Compute full stats (mean/std/min/max + quantiles) for all features including
camera images/videos, and write to meta/stats.json.

Uses incremental RunningQuantileStats to avoid loading all frames into memory.
For video features, samples frames uniformly from each mp4 file via cv2.
For tabular features, reads parquet files directly.

Usage:
    conda run -n lerobot python scripts/compute_dataset_stats.py \
        --root /home/wook/lerobot/woozziam/260413_r1lite_lerobot_v3_clean
"""

import argparse
import json
import logging
from pathlib import Path

import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, RunningQuantileStats
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.utils import init_logging


# decode at reduced resolution via ffmpeg to speed up and save memory
DECODE_WIDTH   = 160   # width passed to ffmpeg scale filter
FPS_SAMPLE     = 2     # frames-per-second to extract from each video


def compute_video_stats(video_dir: Path) -> dict[str, np.ndarray]:
    """Sample frames from all mp4 files using ffmpeg (supports AV1 SW decode)."""
    mp4_files = sorted(video_dir.rglob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"No mp4 files found under {video_dir}")

    runner = None

    for vfile in tqdm(mp4_files, desc=f"  {video_dir.name}", leave=False):
        # Use ffmpeg to decode at reduced fps + scale, output raw RGB24
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(vfile),
            "-vf", f"fps={FPS_SAMPLE},scale={DECODE_WIDTH}:-1",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # read until the pipe is exhausted
        # we need to know frame size first — probe from the filter output
        # easier: read chunks and infer height once first bytes arrive
        frame_h = frame_w = None
        buf = b""
        first_chunk = True

        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk

            if first_chunk and len(buf) >= 100:
                # deduce h from total bytes / width / 3 — done after first full frame
                # We don't know h yet, so probe it with ffprobe
                res = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0", str(vfile)],
                    capture_output=True, text=True,
                )
                if res.returncode == 0:
                    orig_w, orig_h = map(int, res.stdout.strip().split(","))
                    frame_w = DECODE_WIDTH
                    frame_h = int(orig_h * DECODE_WIDTH / orig_w)
                    # make even
                    if frame_h % 2 != 0:
                        frame_h -= 1
                first_chunk = False

            if frame_h is not None:
                frame_bytes = frame_w * frame_h * 3
                while len(buf) >= frame_bytes:
                    raw_frame = buf[:frame_bytes]
                    buf = buf[frame_bytes:]
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(frame_h, frame_w, 3)
                    pixels = frame.reshape(-1, 3).astype(np.float32) / 255.0
                    if runner is None:
                        runner = RunningQuantileStats(quantile_list=DEFAULT_QUANTILES)
                    runner.update(pixels)

        proc.stdout.close()
        proc.wait()

    if runner is None:
        raise RuntimeError(f"No frames decoded from {video_dir}")

    raw = runner.get_statistics()  # each value: shape (3,)

    # reshape to (3, 1, 1) as expected by LeRobot
    result = {}
    for k, v in raw.items():
        result[k] = v if k == "count" else v.reshape(3, 1, 1)
    return result


def compute_tabular_stats(parquet_dir: Path, column: str) -> dict[str, np.ndarray]:
    """Read all parquet files in parquet_dir and return stats for `column`."""
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {parquet_dir}")

    runner = None

    for pfile in parquet_files:
        df = pd.read_parquet(pfile, columns=[column])
        arr = np.stack(df[column].tolist()).astype(np.float32)  # (T, dim)

        if runner is None:
            runner = RunningQuantileStats(quantile_list=DEFAULT_QUANTILES)
        runner.update(arr)

    if runner is None:
        raise RuntimeError(f"No data found in {parquet_dir} for column {column}")

    return runner.get_statistics()  # shape (dim,) per stat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Local root directory of the LeRobot v3 dataset")
    args = parser.parse_args()

    init_logging()
    root = Path(args.root)

    # --- load info.json to know which features exist ---
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]

    all_stats: dict[str, dict] = {}

    # --- video features ---
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    for vkey in video_keys:
        video_dir = root / "videos" / vkey
        logging.info(f"Computing image stats for {vkey} ...")
        all_stats[vkey] = compute_video_stats(video_dir)
        logging.info(f"  mean per channel: {all_stats[vkey]['mean'].squeeze()}")

    # --- tabular features ---
    data_dir = root / "data"
    tabular_keys = [k for k, v in features.items()
                    if v["dtype"] in ("float32", "float64") and k not in ("timestamp",)]
    # only state + action need normalization
    wanted = {"observation.state", "action"}
    tabular_keys = [k for k in tabular_keys if k in wanted]

    for tkey in tabular_keys:
        logging.info(f"Computing tabular stats for {tkey} ...")
        all_stats[tkey] = compute_tabular_stats(data_dir, tkey)
        mn = all_stats[tkey]["mean"]
        logging.info(f"  mean[:4]: {mn[:4]}")

    # --- write ---
    stats_path = root / "meta" / "stats.json"
    logging.info(f"Writing stats to {stats_path} ...")
    write_stats(all_stats, root)
    logging.info(f"Done. Keys: {list(all_stats.keys())}")


if __name__ == "__main__":
    main()
