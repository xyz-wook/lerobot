#!/usr/bin/env python
"""
데이터셋의 서브키들을 concatenate해서 observation.state와 action 컬럼을 추가하는 스크립트.
기존 parquet 파일을 in-place로 수정(백업 권장).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/wook/lerobot/woozziam/260413_r1lite_lerobot")

STATE_KEYS = [
    "observation.state.left_arm",      # (6,)
    "observation.state.right_arm",     # (6,)
    "observation.state.left_gripper",  # (1,)
    "observation.state.right_gripper", # (1,)
]
# total state dim = 6+6+1+1 = 14

ACTION_KEYS = [
    "action.left_gripper",   # (1,)  — POLICY ordering: lg first
    "action.right_gripper",  # (1,)
    "action.left_arm",       # (6,)
    "action.right_arm",      # (6,)
]
# total action dim = 1+1+6+6 = 14

STATE_DIM = 14
ACTION_DIM = 14
# ─────────────────────────────────────────────────────────────────────────────


def extract_col_as_array(table: pa.Table, col_name: str) -> np.ndarray:
    """
    파케이 테이블에서 컬럼을 추출해 (N, dim) numpy 배열로 반환.
    컬럼이 list<float> 타입이면 그대로, scalar float이면 (N,1)로 reshape.
    """
    col = table.column(col_name)
    arr = col.to_pylist()

    rows = []
    for val in arr:
        if isinstance(val, (list, tuple)):
            rows.append(np.array(val, dtype=np.float64))
        else:
            # scalar (gripper 등 shape=() 케이스)
            rows.append(np.array([val], dtype=np.float64))
    return np.stack(rows, axis=0)  # (N, dim)


def concat_keys(table: pa.Table, keys: list[str]) -> np.ndarray:
    """여러 컬럼을 concat해서 (N, total_dim) 배열 반환."""
    parts = []
    for key in keys:
        if key not in table.schema.names:
            raise KeyError(f"Column '{key}' not found in parquet. Available: {table.schema.names[:10]}...")
        parts.append(extract_col_as_array(table, key))
    return np.concatenate(parts, axis=1)


def ndarray_to_list_column(arr: np.ndarray) -> pa.ChunkedArray:
    """(N, dim) numpy 배열을 pyarrow list<double> 컬럼으로 변환."""
    list_data = arr.tolist()
    return pa.chunked_array([pa.array(list_data, type=pa.list_(pa.float64()))])


def process_parquet_file(parquet_path: Path) -> None:
    print(f"  Processing: {parquet_path.name}")
    table = pq.read_table(parquet_path)

    # observation.state 컬럼 생성
    state_arr = concat_keys(table, STATE_KEYS)
    assert state_arr.shape[1] == STATE_DIM, f"Expected {STATE_DIM} dims, got {state_arr.shape[1]}"
    state_col = ndarray_to_list_column(state_arr)

    # action 컬럼 생성
    action_arr = concat_keys(table, ACTION_KEYS)
    assert action_arr.shape[1] == ACTION_DIM, f"Expected {ACTION_DIM} dims, got {action_arr.shape[1]}"
    action_col = ndarray_to_list_column(action_arr)

    # 기존 컬럼에 추가 (이미 있으면 덮어쓰기)
    if "observation.state" in table.schema.names:
        idx = table.schema.get_field_index("observation.state")
        table = table.remove_column(idx)
    if "action" in table.schema.names:
        idx = table.schema.get_field_index("action")
        table = table.remove_column(idx)

    table = table.append_column("observation.state", state_col)
    table = table.append_column("action", action_col)

    # 덮어쓰기
    pq.write_table(table, parquet_path)
    print(f"    → Added observation.state ({STATE_DIM}d) and action ({ACTION_DIM}d)")


def update_meta_info(dataset_root: Path) -> None:
    """info.json에 observation.state, action feature 추가."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"  [WARN] info.json not found at {info_path}, skipping meta update.")
        return

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})

    features["observation.state"] = {
        "dtype": "float64",
        "shape": [STATE_DIM],
        "names": (
            [f"obs_state_{i}" for i in range(STATE_DIM)]
        ),
    }

    features["action"] = {
        "dtype": "float64",
        "shape": [ACTION_DIM],
        "names": (
            [f"action_{i}" for i in range(ACTION_DIM)]
        ),
    }

    info["features"] = features

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Updated meta/info.json")


def main():
    print("=== LeRobot Dataset Key Merger ===")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"STATE_DIM: {STATE_DIM}  (keys: {len(STATE_KEYS)})")
    print(f"ACTION_DIM: {ACTION_DIM}  (keys: {len(ACTION_KEYS)})")
    print()

    # 백업 경로
    backup_root = DATASET_ROOT.parent / (DATASET_ROOT.name + "_backup")
    if not backup_root.exists():
        print(f"[1/3] Backing up dataset to {backup_root} ...")
        shutil.copytree(DATASET_ROOT, backup_root)
        print("  Backup done.\n")
    else:
        print(f"[1/3] Backup already exists at {backup_root}, skipping.\n")

    # 파케이 파일 처리
    data_dir = DATASET_ROOT / "data"
    if not data_dir.exists():
        # 일부 버전은 train/ 하위에 있음
        data_dir = DATASET_ROOT
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    print(f"[2/3] Processing {len(parquet_files)} parquet file(s)...")
    for pf in parquet_files:
        process_parquet_file(pf)

    print()
    print("[3/3] Updating meta/info.json ...")
    update_meta_info(DATASET_ROOT)

    print()
    print("=== Done! ===")
    print(f"observation.state: {STATE_DIM} dims")
    print(f"  " + " + ".join([
        "left_arm(6)", "right_arm(6)", "left_gripper(1)", "right_gripper(1)",
    ]))
    print(f"action: {ACTION_DIM} dims")
    print(f"  " + " + ".join([
        "left_gripper(1)", "right_gripper(1)", "left_arm(6)", "right_arm(6)",
    ]))
    print()
    print("이제 아래 커맨드로 학습을 시작하세요:")
    print("""  lerobot-train \\
    --dataset.repo_id=woozziam/260413_r1lite_lerobot \\
    --dataset.root=/home/wook/lerobot/woozziam/260413_r1lite_lerobot \\
    --policy.type=diffusion \\
    --output_dir=outputs/train/260420_r1lite_diffusion \\
    --job_name=260420_r1lite_diffusion \\
    --policy.device=cuda \\
    --wandb.enable=true \\
    --policy.repo_id=woozziam/260420_r1lite_diffusion \\
    --dataset.image_transforms.enable=true""")


if __name__ == "__main__":
    main()
