import os
import argparse
import subprocess
import numpy as np
import gzip
import json


def read_episodes(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    return data["episodes"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-data-path", type=str, default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz")
    parser.add_argument("--navila-model-path", type=str, default="/home/zhaojing/mnt/legged_nav/NaVILA/NaVILA-llama3-8B-8f-scanqa-rxr")
    parser.add_argument("--task", type=str, default="go2_matterport_vision")
    parser.add_argument("--low_level_policy_dir", type=str, default="2024-09-25_23-22-02")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="Exclusive end episode index for sharding across GPUs/machines. -1 = run to the end.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip episodes whose measurement json already exists (default on, for restartable full runs).")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--vlm-port", type=int, default=54321,
                        help="Port of the VLM server to forward to navila_eval (per-shard distinct port for multi-GPU).")
    args = parser.parse_args()

    # Define the arguments for evaluation
    eval_args = [f"--task={args.task}", "--num_envs=1",
                f"--load_run={args.low_level_policy_dir}",
                "--headless", "--enable_cameras",
                #  "--visualize_path"
                ]

    # Per-task proprio history (verified empirically against shipped ckpts):
    #   go2_matterport_vision (obs=450), go2_matterport_base (obs=450)  -> history=9
    #   h1_matterport_vision (obs=256), h1_matterport_base              -> history=1 (default)
    # Mismatch fires a load_state_dict shape error on ckpt load.
    if args.task.startswith("go2_"):
        eval_args.append("--history_length=9")

    episodes = read_episodes(args.r2r_data_path)

    end_idx = len(episodes) if args.end_idx < 0 else min(args.end_idx, len(episodes))
    # navila_eval writes measurements here; file name = episode_id - 1
    measurement_dir = os.path.join(
        f"eval_results/{args.task}_loco_{args.low_level_policy_dir}", "measurements"
    )

    # NOTE: tried single-process batch (--episode_idx_list) but Isaac Sim
    # `gym.make` hangs the second time it's called in the same process (known
    # Omniverse limitation: stage swap leaks). Reverted to per-ep subprocess.
    # Per-ep navila_eval still picks up any in-file optimizations on each spawn.
    n_done = n_run = 0
    for i in range(args.start_idx, end_idx):
        episode = episodes[i]
        result_json = os.path.join(measurement_dir, f"{int(episode['episode_id']) - 1}.json")
        if args.resume and os.path.exists(result_json):
            n_done += 1
            continue

        per_ep_args = eval_args + [f"--episode_idx={i}", f"--vlm_port={args.vlm_port}"]
        subprocess.run(['python', 'scripts/navila_eval.py'] + per_ep_args)
        n_run += 1

    print(f"\n[run_benchmark] done. ran {n_run}, skipped {n_done} already-complete, "
          f"range [{args.start_idx},{end_idx}).", flush=True)
