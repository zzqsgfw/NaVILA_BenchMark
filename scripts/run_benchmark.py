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
    parser.add_argument("--lidar_constrain", action="store_true",
                        help="(B-A) forward to navila_eval: lidar-feasibility decode clip")
    parser.add_argument("--safety_factor", type=float, default=0.7)
    parser.add_argument("--loop_breaker_N", type=int, default=0,
                        help="(B-B) forward to navila_eval: 0 disables")
    parser.add_argument("--drift_thresh", type=float, default=0.15)
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to eval_results dir to keep ablations separate from baseline.")
    parser.add_argument("--multi_env_n", type=int, default=1,
                        help="If >1, dispatch via navila_eval_multi.py with this num_envs per call. "
                             "Scene groups larger than N are sliced into multiple N-batches.")
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
        f"eval_results/{args.task}_loco_{args.low_level_policy_dir}{args.output_suffix}", "measurements"
    )

    # Within-scene episode batching: one subprocess per UNIQUE scene_id in the
    # shard range, driving all to-do episodes for that scene via
    # --episode_idx_list. This amortizes the ~30-60s Isaac Sim cold-start +
    # USD load over all eps in the scene (avg ~24 eps/scene over the full
    # 1077-ep benchmark, 11 unique scenes). Cross-scene cannot share an Isaac
    # process (matterport USD can't be hot-swapped + Isaac's gym.make hangs
    # on second call in the same Python process), hence one subprocess per
    # scene rather than one per ep.
    def _scene_id(ep):
        return os.path.splitext(os.path.basename(ep["scene_id"]))[0]

    # Build ordered groups: dict preserving insertion order from sequential
    # iteration over [start_idx, end_idx). Preserves contiguous-scene locality
    # already present in the dataset.
    groups = {}  # scene_id -> list of ep_idx
    n_done = 0
    for i in range(args.start_idx, end_idx):
        episode = episodes[i]
        result_json = os.path.join(measurement_dir, f"{int(episode['episode_id']) - 1}.json")
        if args.resume and os.path.exists(result_json):
            n_done += 1
            continue
        sid = _scene_id(episode)
        groups.setdefault(sid, []).append(i)

    n_run = 0
    n_groups_run = 0
    # navila_eval_multi.py supports num_envs=1 (in-process VLM + hot-swap),
    # which is what we want even for N=1 to avoid per-scene VLM reload. The
    # old single-env navila_eval.py path uses TCP VLM server only.
    use_multi = args.multi_env_n >= 1
    target_script = 'scripts/navila_eval_multi.py' if use_multi else 'scripts/navila_eval.py'

    if use_multi:
        # SINGLE subprocess for the whole shard — navila_eval_multi.py groups
        # by scene + chunks of N internally + hot-swaps matterport USD across
        # scenes (no Isaac/VLM reload per chunk). Massive saving vs per-chunk
        # spawn (each spawn = ~3min VLM in-process load).
        all_eps = [i for sid, ep_list in groups.items() for i in ep_list]
        if not all_eps:
            print("[run_benchmark] nothing to run.", flush=True)
        else:
            per_args = eval_args + [
                f"--episode_idx_list={','.join(str(x) for x in all_eps)}",
                f"--vlm_port={args.vlm_port}",
                f"--output_suffix={args.output_suffix}",
                f"--safety_factor={args.safety_factor}",
                f"--loop_breaker_N={args.loop_breaker_N}",
                f"--drift_thresh={args.drift_thresh}",
                f"--num_envs={args.multi_env_n}",
            ]
            if args.lidar_constrain:
                per_args.append("--lidar_constrain")
            if args.resume:
                per_args.append("--skip_if_done")
            print(f"[run_benchmark] multi dispatch: {len(all_eps)} eps across "
                  f"{len(groups)} scenes (one persistent subprocess)", flush=True)
            subprocess.run(['python', target_script] + per_args)
            n_run = len(all_eps)
            n_groups_run = len(groups)
    else:
        for sid, ep_list in groups.items():
            idx_list = ",".join(str(x) for x in ep_list)
            per_scene_args = eval_args + [
                f"--episode_idx_list={idx_list}",
                f"--vlm_port={args.vlm_port}",
                f"--output_suffix={args.output_suffix}",
                f"--safety_factor={args.safety_factor}",
                f"--loop_breaker_N={args.loop_breaker_N}",
                f"--drift_thresh={args.drift_thresh}",
            ]
            if args.lidar_constrain:
                per_scene_args.append("--lidar_constrain")
            if args.resume:
                per_scene_args.append("--skip_if_done")
            print(f"[run_benchmark] scene={sid}: dispatching via {target_script}",
                  flush=True)
            subprocess.run(['python', target_script] + per_scene_args)
            n_run += len(ep_list)
            n_groups_run += 1
        n_groups_run += 1

    print(f"\n[run_benchmark] done. ran {n_run} episodes across {n_groups_run} scene-groups, "
          f"skipped {n_done} already-complete, range [{args.start_idx},{end_idx}).", flush=True)
