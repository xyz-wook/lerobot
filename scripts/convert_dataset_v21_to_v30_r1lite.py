#!/usr/bin/env python
"""
r1lite v2.1 → v3.0 변환 스크립트 (flat merged 형태).

sub-key 컬럼들을 flat merged observation.state / action으로 변환하고
불필요한 컬럼을 제거하며 v3.0 파일 구조로 변환합니다.
비디오 파일 경로도 v3.0 형식으로 변환합니다.

Usage:
    python scripts/convert_dataset_v21_to_v30_r1lite.py \
        --root /home/wook/lerobot/woozziam/260413_r1lite_lerobot_v2.1 \
        --output /home/wook/lerobot/woozziam/260413_r1lite_lerobot_v3_clean

설정 변경:
    STATE_KEYS, ACTION_KEYS, CAMERA_KEYS 를 수정해 학습에 사용할 feature를 선택하세요.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


# ── r1lite 설정 (학습에 사용할 feature 선택) ────────────────────────────────────
# 순서 = merged 컬럼의 차원 순서

STATE_KEYS = [
    ("observation.state.left_arm",      ["left_arm.0", "left_arm.1", "left_arm.2",
                                         "left_arm.3", "left_arm.4", "left_arm.5"]),
    ("observation.state.right_arm",     ["right_arm.0", "right_arm.1", "right_arm.2",
                                         "right_arm.3", "right_arm.4", "right_arm.5"]),
    ("observation.state.left_gripper",  ["left_gripper"]),
    ("observation.state.right_gripper", ["right_gripper"]),
]

ACTION_KEYS = [
    ("action.left_arm",      ["left_arm.0", "left_arm.1", "left_arm.2",
                               "left_arm.3", "left_arm.4", "left_arm.5"]),
    ("action.right_arm",     ["right_arm.0", "right_arm.1", "right_arm.2",
                               "right_arm.3", "right_arm.4", "right_arm.5"]),
    ("action.left_gripper",  ["left_gripper"]),
    ("action.right_gripper", ["right_gripper"]),
]

CAMERA_KEYS = [
    "observation.images.head_rgb",
    # "observation.images.head_right_rgb",  # 4-카메라 모드면 주석 해제
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]


# ── 파생 상수 ────────────────────────────────────────────────────────────────────
META_COLS    = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
STATE_NAMES  = [n for _, names in STATE_KEYS for n in names]
ACTION_NAMES = [n for _, names in ACTION_KEYS for n in names]
STATE_DIM    = len(STATE_NAMES)
ACTION_DIM   = len(ACTION_NAMES)

V21 = "v2.1"
V30 = "v3.0"
CHUNK_SIZE         = 1000
DATA_FILE_SIZE_MB  = 100
VIDEO_FILE_SIZE_MB = 200
DATA_PATH_TMPL     = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TMPL    = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


# ── 공통 유틸 ────────────────────────────────────────────────────────────────────

def _next_idx(chunk: int, file: int) -> tuple[int, int]:
    """chunk/file 인덱스 증가."""
    file += 1
    if file >= CHUNK_SIZE:
        chunk += 1
        file = 0
    return chunk, file


def col_to_numpy(table: pa.Table, col_name: str) -> np.ndarray:
    """pyarrow list<double> 컬럼 → (N, dim) numpy 배열. scalar이면 (N, 1)."""
    rows = []
    for val in table.column(col_name).to_pylist():
        rows.append(np.array(val if isinstance(val, (list, tuple)) else [val], dtype=np.float64))
    return np.stack(rows)


def to_list_column(arr: np.ndarray) -> pa.ChunkedArray:
    """(N, dim) numpy → pyarrow list<double> 컬럼."""
    return pa.chunked_array([pa.array(arr.tolist(), type=pa.list_(pa.float64()))])


# ── 통계 ────────────────────────────────────────────────────────────────────────

def compute_episode_stats(table: pa.Table) -> dict:
    """에피소드 단위 per-dimension 통계 (observation.state, action)."""
    stats = {}
    for key in ["observation.state", "action"]:
        if key not in table.schema.names:
            continue
        arr = col_to_numpy(table, key)
        stats[key] = dict(
            mean=arr.mean(0), std=arr.std(0),
            min=arr.min(0),   max=arr.max(0),
            count=np.array([len(arr)]),
        )
    return stats


def load_v21_image_stats(root: Path) -> dict[int, dict[str, dict]]:
    """v2.1 episodes_stats.jsonl에서 CAMERA_KEYS에 해당하는 이미지 stats 로드.

    Returns: {ep_idx: {cam_key: {min, max, mean, std, count} as np.ndarray}}
    """
    jsonl_path = root / "meta" / "episodes_stats.jsonl"
    if not jsonl_path.exists():
        logging.warning("episodes_stats.jsonl 없음 — 이미지 stats 없이 진행")
        return {}

    result: dict[int, dict[str, dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            item   = json.loads(line)
            ep_idx = item["episode_index"]
            cam_stats: dict[str, dict] = {}
            for cam in CAMERA_KEYS:
                if cam in item["stats"]:
                    cam_stats[cam] = {k: np.array(v) for k, v in item["stats"][cam].items()}
            if cam_stats:
                result[ep_idx] = cam_stats

    logging.info(f"이미지 stats 로드: {len(result)}개 에피소드, 카메라: {CAMERA_KEYS}")
    return result


def aggregate_stats(all_stats: list[dict]) -> dict:
    """에피소드별 stats → 데이터셋 전체 stats (pooled mean/std).

    tabular: shape (dim,) / image: shape (3, 1, 1) 모두 처리.
    """
    keys = all_stats[0].keys()
    out = {}
    for key in keys:
        counts = np.array([s[key]["count"][0] for s in all_stats], dtype=float)
        total  = counts.sum()
        means  = np.stack([s[key]["mean"] for s in all_stats])
        # counts를 means의 차원 수에 맞게 reshape하여 브로드캐스팅
        c_view = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        g_mean = (means * c_view).sum(0) / total
        stds   = np.stack([s[key]["std"] for s in all_stats])
        diff2  = (means - g_mean) ** 2
        g_std  = np.sqrt(((stds**2 + diff2) * c_view).sum(0) / total)
        mins   = np.stack([s[key]["min"] for s in all_stats])
        maxs   = np.stack([s[key]["max"] for s in all_stats])
        out[key] = dict(
            mean=g_mean, std=g_std,
            min=mins.min(0), max=maxs.max(0),
            count=np.array([int(total)]),
        )
    return out


# ── 데이터 변환 ────────────────────────────────────────────────────────────────

def transform_table(table: pa.Table) -> pa.Table:
    """v2.1 sub-key 컬럼 → flat merged + meta 컬럼만 남긴 Table 반환."""
    state_arr  = np.concatenate([col_to_numpy(table, k) for k, _ in STATE_KEYS],  axis=1)
    action_arr = np.concatenate([col_to_numpy(table, k) for k, _ in ACTION_KEYS], axis=1)

    cols: dict = {}
    for col in META_COLS:
        if col in table.schema.names:
            cols[col] = table.column(col)
    cols["observation.state"] = to_list_column(state_arr)
    cols["action"]            = to_list_column(action_arr)

    return pa.table(cols)


def convert_data(root: Path, new_root: Path) -> tuple[list[dict], list[dict]]:
    """
    v2.1 에피소드 parquet → flat merged 변환 후 v3.0 chunk 파일로 저장.

    Returns:
        episodes_meta : 에피소드별 파일 위치 정보
        all_ep_stats  : 에피소드별 통계
    """
    ep_paths = sorted((root / "data").rglob("episode_*.parquet"))
    if not ep_paths:
        raise FileNotFoundError(f"parquet 파일 없음: {root / 'data'}")

    episodes_meta: list[dict] = []
    all_ep_stats:  list[dict] = []
    frame_offset = 0

    chunk_idx = file_idx = 0
    size_mb   = 0.0
    pending:  list[pa.Table] = []

    def flush(c: int, f: int) -> None:
        combined = pa.concat_tables(pending)
        out = new_root / DATA_PATH_TMPL.format(chunk_index=c, file_index=f)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined, out)

    for ep_path in tqdm.tqdm(ep_paths, desc="데이터 변환"):
        raw      = pq.read_table(ep_path)
        table    = transform_table(raw)
        ep_idx   = int(ep_path.stem.split("_")[-1])
        ep_mb    = ep_path.stat().st_size / 1024 / 1024
        n_frames = len(table)

        if size_mb + ep_mb >= DATA_FILE_SIZE_MB and pending:
            flush(chunk_idx, file_idx)
            chunk_idx, file_idx = _next_idx(chunk_idx, file_idx)
            size_mb = 0.0
            pending = []

        episodes_meta.append(dict(
            episode_index    = ep_idx,
            data_chunk_index = chunk_idx,
            data_file_index  = file_idx,
            from_index       = frame_offset,
            to_index         = frame_offset + n_frames,
        ))
        all_ep_stats.append(compute_episode_stats(table))
        pending.append(table)
        size_mb      += ep_mb
        frame_offset += n_frames

    if pending:
        flush(chunk_idx, file_idx)

    return episodes_meta, all_ep_stats


# ── 비디오 변환 ────────────────────────────────────────────────────────────────

def convert_videos(root: Path, new_root: Path, ep_lengths: dict, fps: int) -> dict:
    """
    v2.1 비디오 경로 (videos/chunk-XXX/{camera}/episode_N.mp4)
    → v3.0 경로 (videos/{camera}/chunk-XXX/file-N.mp4) 로 변환.

    ep_lengths: {ep_idx: n_frames} — 타임스탬프 계산에 사용

    Returns: {ep_idx: {camera_key: {chunk_index, file_index, from_timestamp, to_timestamp}}}
    """
    try:
        from lerobot.datasets.video_utils import concatenate_video_files
    except ImportError:
        logging.warning("lerobot 미설치 — 비디오 변환 건너뜀")
        return {}

    old_info   = json.loads((root / "meta" / "info.json").read_text())
    video_keys = [
        k for k, v in old_info["features"].items()
        if v["dtype"] == "video" and k in CAMERA_KEYS
    ]
    video_meta: dict = {}

    for camera in video_keys:
        # v2.1 경로: videos/chunk-000/{camera}/episode_000000.mp4
        ep_paths = sorted((root / "videos").glob(f"*/{camera}/episode_*.mp4"))
        if not ep_paths:
            logging.warning(f"비디오 없음: {camera}")
            continue

        # 크기 기반으로 그룹핑
        groups: list[tuple[int, int, list[Path]]] = []
        chunk_idx = file_idx = 0
        size_mb   = 0.0
        current:  list[Path] = []

        for vpath in ep_paths:
            vmb = vpath.stat().st_size / 1024 / 1024
            if size_mb + vmb >= VIDEO_FILE_SIZE_MB and current:
                groups.append((chunk_idx, file_idx, current))
                chunk_idx, file_idx = _next_idx(chunk_idx, file_idx)
                size_mb, current = 0.0, []
            current.append(vpath)
            size_mb += vmb

        if current:
            groups.append((chunk_idx, file_idx, current))

        for g_chunk, g_file, g_paths in tqdm.tqdm(groups, desc=f"비디오: {camera}"):
            out = new_root / VIDEO_PATH_TMPL.format(
                video_key=camera, chunk_index=g_chunk, file_index=g_file
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            concatenate_video_files(g_paths, out)

            # 파일 내 누적 타임스탬프 계산
            t_offset = 0.0
            for vpath in g_paths:
                ep_idx  = int(vpath.stem.split("_")[-1])
                n       = ep_lengths.get(ep_idx, 0)
                dur     = n / fps
                video_meta.setdefault(ep_idx, {})[camera] = {
                    "chunk_index":    g_chunk,
                    "file_index":     g_file,
                    "from_timestamp": t_offset,
                    "to_timestamp":   t_offset + dur,
                }
                t_offset += dur

    return video_meta


# ── 메타 파일 작성 ──────────────────────────────────────────────────────────────

def write_info(root: Path, new_root: Path, total_frames: int) -> None:
    """meta/info.json 작성 (flat feature 정의)."""
    old = json.loads((root / "meta" / "info.json").read_text())

    features: dict = {}

    # 카메라 feature (video 메타 그대로 복사, fps 제거)
    for cam in CAMERA_KEYS:
        if cam in old["features"]:
            feat = dict(old["features"][cam])
            feat.pop("fps", None)
            features[cam] = feat

    # flat merged state / action
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [STATE_DIM],
        "names": STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [ACTION_DIM],
        "names": ACTION_NAMES,
    }

    # 메타 컬럼
    for col in META_COLS:
        if col in old["features"]:
            feat = dict(old["features"][col])
            feat.pop("fps", None)
            features[col] = feat

    info = {
        "codebase_version":       V30,
        "robot_type":             old.get("robot_type", "r1lite"),
        "total_episodes":         old["total_episodes"],
        "total_frames":           total_frames,
        "total_tasks":            old["total_tasks"],
        "chunks_size":            CHUNK_SIZE,
        "data_files_size_in_mb":  DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
        "fps":                    int(old["fps"]),
        "splits":                 old["splits"],
        "data_path":              DATA_PATH_TMPL,
        "video_path":             VIDEO_PATH_TMPL,
        "features":               features,
    }

    p = new_root / "meta" / "info.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2))
    logging.info(f"info.json 작성: {p}")


def write_stats(new_root: Path, stats: dict) -> None:
    """meta/stats.json 작성."""
    def ser(v):
        return v.tolist() if isinstance(v, np.ndarray) else float(v)

    out = {k: {sk: ser(sv) for sk, sv in s.items()} for k, s in stats.items()}
    p = new_root / "meta" / "stats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    logging.info(f"stats.json 작성: {p}")


def write_tasks(root: Path, new_root: Path) -> None:
    """meta/tasks.parquet 작성.

    LeRobot load_tasks()는 task 문자열이 index인 DataFrame을 기대함.
    schema: index(name="task")=task문자열, 컬럼=task_index(int64)
    """
    try:
        import jsonlines
        with jsonlines.open(root / "meta" / "tasks.jsonl") as r:
            tasks = list(r)
    except Exception:
        logging.warning("tasks.jsonl 없음 — 빈 tasks.parquet 생성")
        tasks = []

    df = pd.DataFrame(
        {"task_index": [t["task_index"] for t in tasks]},
        index=pd.Index([t["task"] for t in tasks], name="task"),
    )
    p = new_root / "meta" / "tasks.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    logging.info(f"tasks.parquet 작성: {p}")


def write_episodes(
    root: Path,
    new_root: Path,
    episodes_meta: list[dict],
    all_ep_stats: list[dict],
    video_meta: dict,
) -> None:
    """meta/episodes/chunk-000/file-000.parquet 작성 (v3.0 스키마)."""
    try:
        import jsonlines
        with jsonlines.open(root / "meta" / "episodes.jsonl") as r:
            legacy = {e["episode_index"]: e for e in r}
    except Exception:
        logging.warning("episodes.jsonl 없음")
        legacy = {}

    rows = []
    for em, ep_stats in zip(episodes_meta, all_ep_stats):
        ep_idx   = em["episode_index"]
        leg      = legacy.get(ep_idx, {})
        n_frames = em["to_index"] - em["from_index"]

        row: dict = {
            "episode_index":      ep_idx,
            "data/chunk_index":   em["data_chunk_index"],
            "data/file_index":    em["data_file_index"],
            "dataset_from_index": em["from_index"],
            "dataset_to_index":   em["to_index"],
            "tasks":              leg.get("tasks", []),
            "length":             n_frames,
        }

        for cam in CAMERA_KEYS:
            vm = video_meta.get(ep_idx, {}).get(cam, {})
            row[f"videos/{cam}/chunk_index"]    = vm.get("chunk_index",    0)
            row[f"videos/{cam}/file_index"]     = vm.get("file_index",     0)
            row[f"videos/{cam}/from_timestamp"] = vm.get("from_timestamp", 0.0)
            row[f"videos/{cam}/to_timestamp"]   = vm.get("to_timestamp",   n_frames / 31.0)

        # per-episode stats
        for feat_key, s in ep_stats.items():
            for stat_name in ("min", "max", "mean", "std", "count"):
                col = f"stats/{feat_key}/{stat_name}"
                row[col] = s[stat_name].tolist()

        # episodes 파일 자신의 위치 (항상 chunk-000 / file-000)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"]  = 0

        rows.append(row)

    df = pd.DataFrame(rows)
    p  = new_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logging.info(f"episodes parquet 작성: {p}")


# ── 메인 ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="r1lite v2.1 → v3.0 변환 (flat merged)")
    parser.add_argument("--root",   required=True, type=Path, help="v2.1 데이터셋 루트")
    parser.add_argument("--output", required=True, type=Path, help="v3.0 출력 경로")
    args = parser.parse_args()

    root, new_root = Path(args.root), Path(args.output)

    # 입력 검증
    if not root.exists():
        raise FileNotFoundError(f"입력 경로 없음: {root}")
    old_info = json.loads((root / "meta" / "info.json").read_text())
    if old_info.get("codebase_version") != V21:
        raise ValueError(f"v2.1 데이터셋이 아닙니다: {old_info.get('codebase_version')}")

    if new_root.exists():
        shutil.rmtree(new_root)
        logging.info(f"기존 출력 경로 제거: {new_root}")

    print(f"\n=== r1lite v2.1 → v3.0 변환 ===")
    print(f"입력:  {root}")
    print(f"출력:  {new_root}")
    print(f"observation.state  {STATE_DIM}-dim: {STATE_NAMES}")
    print(f"action             {ACTION_DIM}-dim: {ACTION_NAMES}")
    print(f"cameras: {CAMERA_KEYS}\n")

    fps = int(json.loads((root / "meta" / "info.json").read_text())["fps"])

    print("[1/4] 데이터 변환 중...")
    episodes_meta, all_ep_stats = convert_data(root, new_root)

    # v2.1 episodes_stats.jsonl에서 이미지 stats 읽어서 각 에피소드 stats에 merge
    image_stats = load_v21_image_stats(root)
    if image_stats:
        for i, em in enumerate(episodes_meta):
            ep_idx = em["episode_index"]
            if ep_idx in image_stats:
                all_ep_stats[i].update(image_stats[ep_idx])

    print("[2/4] 비디오 변환 중...")
    ep_lengths = {em["episode_index"]: em["to_index"] - em["from_index"] for em in episodes_meta}
    video_meta = convert_videos(root, new_root, ep_lengths, fps)

    print("[3/4] 통계 집계 중...")
    dataset_stats = aggregate_stats(all_ep_stats)

    print("[4/4] 메타 파일 작성 중...")
    total_frames = sum(em["to_index"] - em["from_index"] for em in episodes_meta)
    write_info(root, new_root, total_frames)
    write_stats(new_root, dataset_stats)
    write_tasks(root, new_root)
    write_episodes(root, new_root, episodes_meta, all_ep_stats, video_meta)

    print(f"\n=== 완료 ===")
    print(f"  에피소드: {len(episodes_meta)}개  |  프레임: {total_frames:,}개")
    print(f"  observation.state: {STATE_DIM}-dim")
    print(f"  action:            {ACTION_DIM}-dim")
    print(f"  출력: {new_root}")


if __name__ == "__main__":
    main()
