"""
Open-loop evaluation for R1Lite diffusion policy.

Loads a dataset episode, feeds observations through the policy (same pipeline as
the async server), plots predicted actions vs ground-truth. Use this to determine
whether the issue is training quality or a runtime/observation-ordering problem.

Usage:
    conda run -n lerobot python eval_open_loop.py --episode 0 --device cuda
    conda run -n lerobot python eval_open_loop.py --episode 0 --device cpu --no_images
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

CHECKPOINT = "/home/wook/lerobot/outputs/train/260420_r1lite_diffusion/checkpoints/200000/pretrained_model"
DATASET_ROOT = "/home/wook/lerobot/woozziam/260413_r1lite_lerobot"
DATASET_ID = "woozziam/260413_r1lite_lerobot"

ACTION_LABELS = [
    "left_gripper",
    "right_gripper",
    "chassis_vx", "chassis_vy", "chassis_vz", "chassis_v3", "chassis_v4", "chassis_v5",
    "torso_v0", "torso_v1", "torso_v2", "torso_v3", "torso_v4", "torso_v5",
    "left_arm_j0", "left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4", "left_arm_j5",
    "right_arm_j0", "right_arm_j1", "right_arm_j2", "right_arm_j3", "right_arm_j4", "right_arm_j5",
]


def inject_combined_action_stats(postprocessor):
    """Same fix as policy_server._inject_combined_action_stats."""
    from lerobot.configs import FeatureType
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep

    for step in postprocessor.steps:
        if not isinstance(step, UnnormalizerProcessorStep):
            continue
        if "action" in step._tensor_stats:
            print("Combined action stats already present.")
            return
        sub_keys = [k for k, v in step.features.items() if v.type == FeatureType.ACTION and k != "action"]
        if not sub_keys:
            return
        missing = [k for k in sub_keys if k not in step._tensor_stats]
        if missing:
            print(f"WARNING: missing sub-feature stats: {missing}")
            return
        combined = {}
        for stat_name, stat_val in step._tensor_stats[sub_keys[0]].items():
            if stat_name == "count":
                combined[stat_name] = stat_val
            else:
                combined[stat_name] = torch.cat([step._tensor_stats[k][stat_name] for k in sub_keys])

        # Must update BOTH _tensor_stats AND self.stats so the key survives device switches.
        # self.to() rebuilds _tensor_stats from self.stats; writing only to _tensor_stats
        # causes the key to disappear when the tensor moves to GPU.
        step._tensor_stats["action"] = combined
        step.stats["action"] = {
            k: v.cpu().numpy() if v.numel() > 1 else float(v.item())
            for k, v in combined.items()
        }
        print(f"Injected combined action stats from {sub_keys}, shape={combined['min'].shape}")


def main(args):
    device = args.device

    # ── Load policy + processors ──────────────────────────────────────────────
    print(f"Loading policy from {CHECKPOINT} ...")
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    policy = get_policy_class("diffusion").from_pretrained(CHECKPOINT)
    policy.eval()
    policy.to(device)
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    inject_combined_action_stats(postprocessor)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset episode {args.episode} ...")
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        DATASET_ID,
        root=DATASET_ROOT,
        episodes=[args.episode],
    )

    # Figure out which global frame indices belong to this episode
    ep_meta = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == args.episode
    )
    n_frames = len(ep_meta)
    if args.max_frames and args.max_frames < n_frames:
        n_frames = args.max_frames
        print(f"Episode {args.episode}: {len(ep_meta)} frames total, evaluating first {n_frames}")
    else:
        print(f"Episode {args.episode}: {n_frames} frames")

    # ── Run open-loop inference ───────────────────────────────────────────────
    pred_actions = []   # list of np.ndarray [26]
    gt_actions = []     # list of np.ndarray [26]

    n_obs_steps = policy.config.n_obs_steps

    t_decode = t_preproc = t_infer = t_post = 0.0

    pbar = tqdm(range(n_frames), desc="Open-loop eval", unit="frame", dynamic_ncols=True)
    for i in pbar:
        t0 = time.perf_counter()
        item = dataset[i]  # dict with shape [] per tensor, [H,W,C] for images
        t_decode += time.perf_counter() - t0

        # Ground-truth action in POLICY ORDER (matches postprocessor output ordering):
        # left_gripper(1), right_gripper(1), chassis.vel(6), torso.vel(6), left_arm(6), right_arm(6)
        # NOTE: item["action"] uses DATASET ordering (left_arm first) — do NOT use it for GT.
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return np.atleast_1d(x.float().numpy())
            return np.atleast_1d(float(x)).astype(np.float32)

        if "action.left_gripper" in item:
            gt_vec = np.concatenate([
                _to_np(item["action.left_gripper"]),
                _to_np(item["action.right_gripper"]),
                _to_np(item.get("action.chassis.velocities", torch.zeros(6))),
                _to_np(item.get("action.torso.velocities", torch.zeros(6))),
                _to_np(item["action.left_arm"]),
                _to_np(item["action.right_arm"]),
            ])
            gt_actions.append(gt_vec)
        elif "action" in item:
            gt_actions.append(item["action"].float().numpy())

        # Build observation dict matching what the server sends to the preprocessor:
        #   observation.state [state_dim] and observation.images.* [C,H,W]
        obs = {}

        # State: use combined observation.state [64] directly from dataset
        if "observation.state" in item:
            obs["observation.state"] = item["observation.state"].float()
        else:
            # Fallback: concatenate sub-features in dataset feature order
            sub_state_keys = [
                k for k in item if k.startswith("observation.state.")
                and isinstance(item[k], torch.Tensor)
            ]
            if sub_state_keys:
                obs["observation.state"] = torch.cat([item[k].float() for k in sorted(sub_state_keys)])

        for img_key, img_feat in policy.config.image_features.items():
            if args.no_images:
                # Zero image: policy still needs OBS_IMAGES; predictions will reflect state only
                c, h, w = img_feat.shape
                obs[img_key] = torch.zeros(c, h, w)
            elif img_key in item:
                img = item[img_key]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                # Dataset returns (H,W,C), policy expects (C,H,W)
                if img.ndim == 3 and img.shape[-1] in (1, 3):
                    img = img.permute(2, 0, 1)
                obs[img_key] = img

        # Add batch dim to all tensors
        obs_batched = {k: v.unsqueeze(0) for k, v in obs.items()}

        t1 = time.perf_counter()
        # Run through preprocessor (normalizes images with MEAN_STD, state no-op)
        obs_processed = preprocessor(obs_batched)
        obs_processed.pop(ACTION, None)

        # Stack images into OBS_IMAGES for diffusion policy (always required)
        if policy.config.image_features:
            obs_processed = policy._resize_images_in_batch(
                obs_processed, list(policy.config.image_features)
            )
            obs_processed[OBS_IMAGES] = torch.stack(
                [obs_processed[k] for k in policy.config.image_features], dim=-4
            )

        # Populate temporal queues
        policy._queues = populate_queues(policy._queues, obs_processed)
        t_preproc += time.perf_counter() - t1

        # Only predict after queue is full
        if i >= n_obs_steps - 1:
            t2 = time.perf_counter()
            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(obs_processed)
            t_infer += time.perf_counter() - t2

            t3 = time.perf_counter()
            # Model outputs raw physical values in dataset ordering: [la(0:6), ra(6:12), lg(12), rg(13), cv(14:20), tv(20:26)]
            # Reorder to policy ordering: [lg, rg, cv, tv, la, ra]
            DATASET_TO_POLICY = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            first_action = action_chunk[0, 0, DATASET_TO_POLICY].cpu().numpy()
            pred_actions.append(first_action)
            t_post += time.perf_counter() - t3

        # Update progress bar postfix with timing breakdown
        if i > 0 and i % 10 == 0:
            pbar.set_postfix({
                "decode": f"{t_decode/max(i,1)*1000:.0f}ms",
                "infer": f"{t_infer/max(len(pred_actions),1)*1000:.0f}ms",
            }, refresh=False)

    pred = np.array(pred_actions)   # (T, 26)
    gt = np.array(gt_actions[n_obs_steps - 1:])  # align with predictions

    T = min(len(pred), len(gt))
    pred, gt = pred[:T], gt[:T]

    print(f"\nTiming summary over {n_frames} frames:")
    print(f"  Video decode:  {t_decode*1000/n_frames:.1f} ms/frame")
    print(f"  Preprocess:    {t_preproc*1000/n_frames:.1f} ms/frame")
    print(f"  Inference:     {t_infer*1000/max(len(pred_actions),1):.1f} ms/inference")
    print(f"  Postprocess:   {t_post*1000/max(len(pred_actions),1):.1f} ms/inference")
    print(f"\nPrediction range (min/max): {pred.min():.3f} / {pred.max():.3f}")
    print(f"GT range         (min/max): {gt.min():.3f} / {gt.max():.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Show arm joints + grippers (most informative)
    plot_indices = {
        "Left Gripper": [0],
        "Right Gripper": [1],
        "Left Arm Joints": list(range(14, 20)),
        "Right Arm Joints": list(range(20, 26)),
    }

    fig, axes = plt.subplots(len(plot_indices), 1, figsize=(14, 3 * len(plot_indices)))
    fig.suptitle(
        f"Open-Loop Eval — Episode {args.episode}\n"
        f"{'(no images, state-only)' if args.no_images else '(with images)'}",
        fontsize=12,
    )

    t = np.arange(T)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (title, idxs) in zip(axes, plot_indices.items()):
        for ci, idx in enumerate(idxs):
            label = ACTION_LABELS[idx] if idx < len(ACTION_LABELS) else f"dim{idx}"
            ax.plot(t, gt[:, idx], color=colors[ci], linestyle="-", alpha=0.5,
                    label=f"GT {label}")
            ax.plot(t, pred[:, idx], color=colors[ci], linestyle="--", linewidth=1.5,
                    label=f"Pred {label}")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.legend(fontsize=7, ncol=min(len(idxs) * 2, 6))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(f"open_loop_ep{args.episode}.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path.absolute()}")

    # ── Print per-joint RMSE ──────────────────────────────────────────────────
    print("\nPer-joint RMSE (pred vs GT):")
    rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=0))
    groups = [
        ("left_gripper", [0]),
        ("right_gripper", [1]),
        ("chassis_vel", list(range(2, 8))),
        ("torso_vel", list(range(8, 14))),
        ("left_arm", list(range(14, 20))),
        ("right_arm", list(range(20, 26))),
    ]
    for name, idxs in groups:
        print(f"  {name:20s}: {rmse[idxs].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0, help="Episode index to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_images", action="store_true",
                        help="Use zero images (state-only). Fast but predictions rely on state alone.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit to first N frames (e.g. --max_frames 200 for quick check)")
    args = parser.parse_args()
    main(args)
