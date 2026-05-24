"""Multi-env NaVILA eval: num_envs=N in a single Isaac process + in-process
batched VLM. Each env runs a different episode; VLM gets one big batch=N call
per VLM tick instead of N serial socket round-trips.

Run via:
  python scripts/navila_eval_multi.py --task=go2_matterport_base \
      --num_envs=4 --history_length=9 --load_run=2024-12-10_21-44-44 \
      --episode_idx_list=10,11,12,13 \
      --headless --enable_cameras

Notes:
  * All episodes in --episode_idx_list MUST share the same scene_id (loader
    builds ONE matterport USD per env at env_origin offset).
  * env_spacing is forced to 100 m so per-env cameras can't see across.
  * Robot/disk init positions are kept absolute (episode coords); IsaacLab
    automatically adds env_origin to every env's prim, so each env's robot
    spawns inside its own matterport copy at the episode's true start_pos.
  * VLM is loaded in-process; no socket. Each VLM tick batches up to N
    parallel image_lists.
"""
import argparse, gzip, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

# CLI
import cli_args  # NaVILA-Bench's local
parser = argparse.ArgumentParser(description="multi-env NaVILA eval")
parser.add_argument("--task", type=str, default="go2_matterport_base")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--history_length", default=9, type=int)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_cnn", action="store_true", default=None)
parser.add_argument("--use_rnn", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--episode_idx_list", type=str, required=True,
                    help="comma-separated ep indices (MUST share scene_id)")
parser.add_argument("--skip_if_done", action="store_true", default=False)
parser.add_argument("--vlm_port", type=int, default=0,
                    help="Ignored — multi loads VLM in-process. Accepted for run_benchmark compatibility.")
parser.add_argument("--output_suffix", type=str, default="",
                    help="Suffix on eval_results dir to keep ablations separate from baseline.")
parser.add_argument("--lidar_constrain", action="store_true",
                    help="(B-A) per-env clip move_forward time-to-go by lidar fwd clearance")
parser.add_argument("--safety_factor", type=float, default=0.7)
parser.add_argument("--loop_breaker_N", type=int, default=0,
                    help="(B-B) per-env loop breaker; 0 disables")
parser.add_argument("--drift_thresh", type=float, default=0.15)
parser.add_argument("--vlm_model_path", type=str,
                    default="/wuji-vefps-D/wuji-rl/zzq_ws/navila_ws/checkpoints/navila-llama3-8b-8f")
parser.add_argument("--max_steps_per_ep", type=int, default=2500)
parser.add_argument("--env_spacing", type=float, default=100.0)
cli_args.add_rsl_rl_args(parser)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(__file__))
from _meshcache import enable_local_mesh_cache
enable_local_mesh_cache()

# boto3 is stubbed (_stubs/boto3/__init__.py) — short-circuits accelerate's
# sagemaker import chain that otherwise drags in Isaac's broken extscache
# botocore. The stub only needs to satisfy `import boto3`; nothing in the
# inference path actually calls boto3.
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, process_image,
                            tokenizer_image_token, get_model_name_from_path)
from llava.model.builder import load_pretrained_model

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401  registers tasks
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.io import load_yaml
from isaaclab.utils import update_class_from_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.vlnce.config import *  # noqa: F401,F403
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper
from omni.isaac.vlnce.utils.eval_utils import get_vel_command, read_episodes

# VLM imports done above (after sys.path repair), see comment.


def quat2yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)


def _build_vlm():
    print(f"[multi] loading VLM int8 from {args.vlm_model_path}", flush=True)
    mp = args.vlm_model_path
    mn = get_model_name_from_path(mp)
    tok, model, ip, _ = load_pretrained_model(mp, mn, None, load_8bit=True, device="cuda:0")
    model.config.image_processor = ip
    print(f"[multi] VLM loaded; alloc={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return tok, model, ip


def _vlm_batch_tcp(image_lists, instructions, host, port):
    """Send batched VLM request to persistent vlm_server.py over TCP.
    Server handles batched generate (FIXED path). Returns list[str].
    Avoids per-scene VLM in-process reload (~20s/scene saving).
    """
    import socket as _sock, base64 as _b64
    from io import BytesIO as _BIO
    # PIL -> base64
    images_list_b64 = []
    for imgs in image_lists:
        env_b64 = []
        for im in imgs:
            buf = _BIO()
            im.save(buf, format="JPEG", quality=85)
            env_b64.append(_b64.b64encode(buf.getvalue()).decode())
        images_list_b64.append(env_b64)
    payload = json.dumps({"images_list": images_list_b64, "queries": instructions}).encode()
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.settimeout(30.0)  # surface hangs as exception
    s.connect((host, port))
    s.sendall(len(payload).to_bytes(8, "big"))
    s.sendall(payload)
    sz = int.from_bytes(s.recv(8), "big")
    data = b""
    while len(data) < sz:
        pkt = s.recv(65536)
        if not pkt:
            break
        data += pkt
    s.close()
    resp = json.loads(data.decode())
    if not isinstance(resp, list):
        raise RuntimeError(f"batched server returned non-list: {type(resp)}: {str(resp)[:200]}")
    return resp


def _vlm_batch_call(tok, model, image_lists, instructions, conv_mode="llama_3",
                    num_video_frames=8, max_new_tokens=32):
    """Batched VLM call (N envs in one generate). Returns list[str] of length N.

    KNOWN BUG in this VILA build's prepare_inputs_labels_for_multimodal:
    when batch rows have mixed lengths and we use standard pad_token_id +
    attention_mask=0 for pads, the re-pad/position_id alignment for row>=1
    breaks → row>=1 outputs garbage like '.'. Verified via /tmp/diag_batched_vlm.py
    (tests B / B' fail, tests D / E pass).

    Workaround: pad SHORT rows to the LONGEST row with BOS tokens (any non-pad
    real token works), and set attention_mask to all 1 (no masking). The model
    treats the BOS-prefix as benign extra context and ignores it because the
    image tokens + instruction tail still dominate the next-token prediction.
    """
    N = len(image_lists)
    image_token = "<image>\n"
    stop_str = conv_templates[conv_mode].sep2 or conv_templates[conv_mode].sep

    # Build prompts and tokenize each independently
    tok_lists = []
    for env_id in range(N):
        conv = conv_templates[conv_mode].copy()
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_token * (num_video_frames-1)}, and current observation <image>\n. '
            f'Your assigned task is: "{instructions[env_id]}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right "
            f"by a specific degree, moving forward a certain distance, or stop if the task is completed."
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        tok_lists.append(ids)

    # Pad short rows at FRONT with BOS tokens (not pad_token_id!) — verified
    # safe in test D. Attention mask = all 1.
    Lmax = max(t.shape[0] for t in tok_lists)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    input_ids = torch.full((N, Lmax), bos_id, dtype=tok_lists[0].dtype)
    for i, t in enumerate(tok_lists):
        input_ids[i, Lmax - t.shape[0]:] = t  # right-aligned real tokens, BOS-prefix
    attn = torch.ones((N, Lmax), dtype=torch.long)
    input_ids = input_ids.to("cuda:0")
    attn = attn.to("cuda:0")

    # Process per-env images
    per_env_tensors = []
    for env_id in range(N):
        imgs = image_lists[env_id]
        if len(imgs) < 8:
            pad_template = imgs[-1] if imgs else Image.new("RGB", (224, 224), (0, 0, 0))
            imgs = [Image.new("RGB", pad_template.size, (0, 0, 0))] * (8 - len(imgs)) + imgs
        ns = len(imgs)
        idxs = [int(i * (ns - 1) / 7) for i in range(7)] + [ns - 1]
        sampled = [imgs[i] for i in idxs]
        processed = [process_image(im.resize((224, 224), Image.BILINEAR), model.config, None) for im in sampled]
        per_env_tensors.append(torch.stack(processed, dim=0).to("cuda:0", dtype=torch.float16))

    sc = KeywordsStoppingCriteria([stop_str], tok, input_ids)
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids, attention_mask=attn, images=per_env_tensors,
            do_sample=False, temperature=0, top_p=None, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
            stopping_criteria=[sc],
        )
    # generate returns ONLY new tokens; per-row truncation at first period
    # handles batched-stop tail garbage
    raws = tok.batch_decode(out_ids, skip_special_tokens=True)
    out = []
    for r in raws:
        r = r.strip()
        p = r.find('.')
        if p >= 0:
            r = r[:p + 1]
        out.append(r)
    return out


def _snapshot_and_write_async(i, ep, meas_dir, env_w, n_vlm, n_pf, broke,
                              pool, futures):
    """Snapshot env state for env i NOW (synchronously, off GPU), then queue
    JSON serialize + disk write on a worker thread so the sim loop never
    blocks on filesystem.

    The snapshot must happen on the main thread because env_w torch tensors
    will be mutated by the next sim step.
    """
    try:
        p_world = env_w.unwrapped.scene["robot"].data.root_pos_w[i].detach().cpu().numpy()
        origins = env_w.unwrapped.scene.env_origins[i].detach().cpu().numpy()
        p_local = p_world - origins
        goal = ep["reference_path"][-1]
        d2g = float(np.linalg.norm(np.array(goal[:2]) - p_local[:2]))
    except Exception:
        d2g = -1.0
    meas = {
        "path_length": -1.0,
        "distance_to_goal": d2g,
        "success": 1.0 if (0 <= d2g < 3.0) else 0.0,
        "spl": 0.0,
        "oracle_navigation_error": -1.0,
        "oracle_success": 0.0,
        "collision": 0.0,
        "termination_reason": "none",
        "obstacle_clearance": {"min": -1.0, "p5": -1.0, "mean": -1.0, "frac_lt_0p3": 0.0},
        "n_vlm_calls": int(n_vlm),
        "n_parse_fallthrough": int(n_pf),
        "parse_fail_rate": (n_pf / n_vlm) if n_vlm else 0.0,
        "broke_by": broke,
        "_multi_env_provisional": True,
    }
    eid = int(ep["episode_id"]) - 1
    path = f"{meas_dir}/{eid}.json"

    def _do_write():
        # JSON encode + write on the worker thread (these CAN block).
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meas, f, indent=4)
        os.replace(tmp, path)
        print(f"[multi] wrote {path}  vlm={n_vlm} broke={broke}  d2g={d2g:.2f}", flush=True)

    futures.append(pool.submit(_do_write))


def main():
    ep_indices_full = [int(x) for x in args.episode_idx_list.split(",") if x.strip()]
    N = args.num_envs

    all_eps = json.load(gzip.open(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")))["episodes"]
    episodes_full = [all_eps[i] for i in ep_indices_full]

    # Group eps by scene_id, then chunk each scene's eps into batches of N.
    # The Isaac process stays alive ACROSS scenes — we hot-swap the matterport
    # USD via terrain.swap_matterport() instead of re-spawning Isaac per scene.
    from collections import defaultdict, OrderedDict
    by_scene = OrderedDict()
    for ep in episodes_full:
        sid = os.path.splitext(os.path.basename(ep["scene_id"]))[0]
        by_scene.setdefault(sid, []).append(ep)
    # Chunk each scene's eps into [N]-sized batches (last chunk may be smaller).
    scene_chunks = []  # list of (scene_id, list[ep] of length<=N)
    for sid, eps in by_scene.items():
        for i in range(0, len(eps), N):
            scene_chunks.append((sid, eps[i:i + N]))
    if not scene_chunks:
        print("[multi] no eps to run", flush=True); return
    first_scene_id = scene_chunks[0][0]
    print(f"[multi] {len(scene_chunks)} scene-chunks across {len(by_scene)} scenes "
          f"(N={N})", flush=True)

    # The Isaac env we build uses the FIRST scene; later chunks swap in place.
    scene_id = first_scene_id
    # The chunk-local 'episodes' starts as the first chunk's eps; chunks with
    # fewer than N eps will pad by repeating the last ep (won't be written).
    episodes = scene_chunks[0][1]
    if len(episodes) < N:
        episodes = list(episodes) + [episodes[-1]] * (N - len(episodes))
    ep_indices = [ep_indices_full[i] for i in range(min(len(ep_indices_full), N))]

    # Resume skip — short-circuit if all eps already done
    result_dir = os.path.join(os.path.dirname(__file__), "..",
                              f"eval_results/{args.task}_loco_{args.load_run}{args.output_suffix}")
    meas_dir = os.path.join(result_dir, "measurements")
    os.makedirs(meas_dir, exist_ok=True)
    if args.skip_if_done:
        remaining = [(i, e) for i, e in zip(ep_indices, episodes)
                     if not os.path.exists(f"{meas_dir}/{int(e['episode_id'])-1}.json")]
        if not remaining:
            print(f"[multi] all {N} eps already done. exit.", flush=True)
            return
        if len(remaining) != N:
            print(f"[multi] WARNING: {N - len(remaining)} eps already done; cannot drop "
                  f"because num_envs is fixed at build time. Running anyway and over-writing.",
                  flush=True)

    # Build env_cfg with num_envs=N
    env_cfg = parse_env_cfg(args.task, num_envs=N)
    # Set scene-wide knobs
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.terrain.obj_filepath = os.path.join(
        ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    # Use FIRST episode's start as the cfg-level robot init (IsaacLab adds env_origin
    # per-env automatically, so all envs spawn at (init_state.pos + env_origin)).
    # We mutate per-env to the EACH ep's start_position AFTER env build via
    # write_root_pose_to_sim, so the cfg-level pos only matters for env_0.
    ep0 = episodes[0]
    sp = ep0["start_position"]
    if "go2" in args.task:
        z_off = 0.4
    elif "h1" in args.task:
        z_off = 1.0
    else:
        z_off = 0.5
    env_cfg.scene.robot.init_state.rot = tuple(ep0["start_rotation"])
    env_cfg.scene.robot.init_state.pos = (sp[0], sp[1], sp[2] + z_off)
    env_cfg.scene.disk_1.init_state.pos = (sp[0], sp[1], sp[2] + 2.5)
    g = ep0["reference_path"][-1]
    env_cfg.scene.disk_2.init_state.pos = (g[0], g[1], g[2] + 2.5)
    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    # Load locomotion ckpt
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args, play=True)
    log_root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, args.load_run)
    log_agent_cfg = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    update_class_from_dict(agent_cfg, log_agent_cfg)
    resume_path = get_checkpoint_path(log_root_path, args.load_run, agent_cfg.load_checkpoint)
    print(f"[multi] policy ckpt: {resume_path}", flush=True)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if args.history_length > 0:
        env_w = RslRlVecEnvHistoryWrapper(env, history_length=args.history_length)
    else:
        env_w = RslRlVecEnvWrapper(env)

    import inspect as _inspect
    import rsl_rl as _rsl_rl
    from rsl_rl.algorithms.ppo import PPO as _PPO
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, getattr(_rsl_rl, "__version__", "3.0.1"))
        _kw = set(_inspect.signature(_PPO.__init__).parameters.keys()) | {"class_name"}
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["algorithm"] = {k: v for k, v in agent_cfg_dict["algorithm"].items() if k in _kw}
    except ImportError:
        agent_cfg_dict = agent_cfg.to_dict()

    runner = OnPolicyRunner(env_w, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env_w.unwrapped.device)

    # VLM: persistent server (TCP) if --vlm_port>0, else in-process load (10GB/scene)
    if args.vlm_port > 0:
        print(f"[multi] using persistent VLM server at localhost:{args.vlm_port}", flush=True)
        tok, vlm_model, image_processor = None, None, None
    else:
        tok, vlm_model, image_processor = _build_vlm()

    # Set per-env robot start poses via direct sim write.
    # default_root_state has shape [N, 13]: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    device = env_w.unwrapped.device
    robot = env_w.unwrapped.scene["robot"]
    for env_id, ep in enumerate(episodes):
        sp = ep["start_position"]
        rot = ep["start_rotation"]  # [w,x,y,z]
        robot.data.default_root_state[env_id, 0:3] = torch.tensor(
            [sp[0], sp[1], sp[2] + z_off], device=device,
            dtype=robot.data.default_root_state.dtype)
        robot.data.default_root_state[env_id, 3:7] = torch.tensor(
            rot, device=device, dtype=robot.data.default_root_state.dtype)
        robot.data.default_root_state[env_id, 7:13] = 0.0
    # disk markers are AssetBaseCfg XformPrims — no .data, skip.

    # Per-env state shared across scene-chunks
    sim_dt = env_w.unwrapped.cfg.sim.dt * env_w.unwrapped.cfg.decimation
    steps_per_image = int(0.5 / sim_dt)
    max_steps = args.max_steps_per_ep
    # Background write pool: shared across all chunks (writes from earlier chunks
    # can complete in parallel with later chunks' sim).
    _write_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="meas-write")
    _write_futures = []

    def _run_chunk(chunk_episodes, real_n):
        """Run inner sim loop for one chunk of up-to-N eps. Writes per-ep meas
        async. real_n = number of REAL eps in this chunk (others are padding).
        """
        # Re-init per-chunk state
        image_observations = [[] for _ in range(N)]
        target_steps = [0] * N
        env_steps_to_go = [0] * N
        vlm_vels = [[0.0, 0.0, 0.0] for _ in range(N)]
        cur_steps = [0] * N
        n_vlm_calls = [0] * N
        n_parse_fall = [0] * N
        broke_by = ["running"] * N
        done_envs = [False] * N
        # Padding envs are immediately "done" so we don't write or step them
        for j in range(real_n, N):
            done_envs[j] = True
            broke_by[j] = "padding"
        last_raws = ["(none)" for _ in range(N)]
        instructions = [e["instruction"]["instruction_text"] for e in chunk_episodes]

        # Set per-env robot start poses for this chunk
        for env_id, ep in enumerate(chunk_episodes):
            sp = ep["start_position"]; rot = ep["start_rotation"]
            robot.data.default_root_state[env_id, 0:3] = torch.tensor(
                [sp[0], sp[1], sp[2] + z_off], device=device,
                dtype=robot.data.default_root_state.dtype)
            robot.data.default_root_state[env_id, 3:7] = torch.tensor(
                rot, device=device, dtype=robot.data.default_root_state.dtype)
            robot.data.default_root_state[env_id, 7:13] = 0.0

        obs, infos = env_w.reset()
        cam0 = infos["observations"]["camera_obs"]
        for i in range(N):
            frame = cam0[i, :, :, :3].cpu().numpy().astype(np.uint8)
            image_observations[i].append(Image.fromarray(frame))

        t_chunk_start = time.perf_counter()
        while app.is_running() and not all(done_envs):
            need_vlm = [i for i in range(N)
                        if (not done_envs[i]) and cur_steps[i] >= target_steps[i]]
            if need_vlm:
                t_v = time.perf_counter()
                if args.vlm_port > 0:
                    raws = _vlm_batch_tcp(
                        [image_observations[i] for i in need_vlm],
                        [instructions[i] for i in need_vlm],
                        host="localhost", port=args.vlm_port,
                    )
                else:
                    raws = _vlm_batch_call(
                        tok, vlm_model,
                        [image_observations[i] for i in need_vlm],
                        [instructions[i] for i in need_vlm],
                    )
                lat = time.perf_counter() - t_v
                for j, env_id in enumerate(need_vlm):
                    last_raws[env_id] = raws[j]
                    vel, t_to_go, parse_info = get_vel_command(raws[j])

                    if args.loop_breaker_N >= 2:
                        if 'loop_breakers' not in locals():
                            from _inline_intervention_patch import LoopBreaker
                            loop_breakers = [LoopBreaker(N=args.loop_breaker_N,
                                                         drift_thresh_m=args.drift_thresh) for _ in range(N)]
                        _rp = env_w.unwrapped.scene["robot"].data.root_pos_w[env_id].detach().cpu().numpy()
                        trig, ov_vel, ov_ttg = loop_breakers[env_id].step(raws[j], (_rp[0], _rp[1]))
                        if trig:
                            vel, t_to_go = ov_vel, ov_ttg
                            parse_info = dict(parse_info, loop_breaker_fired=True)

                    if args.lidar_constrain:
                        if 'lidar_clip_fn' not in locals():
                            from _inline_intervention_patch import lidar_constrain_command
                            lidar_clip_fn = lidar_constrain_command
                        try:
                            vel, t_to_go, parse_info, clip_meta = lidar_clip_fn(
                                raws[j], vel, t_to_go, parse_info, env_w,
                                safety_factor=args.safety_factor,
                            )
                            parse_info = dict(parse_info, clip=clip_meta)
                        except Exception:
                            pass

                    vlm_vels[env_id] = list(vel)
                    env_steps_to_go[env_id] = int(t_to_go / sim_dt)
                    target_steps[env_id] = cur_steps[env_id] + env_steps_to_go[env_id]
                    n_vlm_calls[env_id] += 1
                    if parse_info["fallthrough"]:
                        n_parse_fall[env_id] += 1
                    # VLM said "stop" — terminate this env now (matches single-env behaviour)
                    if parse_info.get("action") == "stop":
                        broke_by[env_id] = "done"
                        done_envs[env_id] = True
                        if env_id < real_n:
                            _snapshot_and_write_async(env_id, chunk_episodes[env_id], meas_dir,
                                                      env_w, n_vlm_calls[env_id], n_parse_fall[env_id],
                                                      broke_by[env_id], _write_pool, _write_futures)
                print(f"[multi] step={max(cur_steps):4d} VLM batch={len(need_vlm)} "
                      f"latency={lat:.2f}s raw0={raws[0][:50]!r}", flush=True)

            vlm_cmd = torch.zeros((N, 3), device=device, dtype=torch.float32)
            for i in range(N):
                if not done_envs[i]:
                    vlm_cmd[i] = torch.tensor(vlm_vels[i], device=device, dtype=torch.float32)
            env_w.proprio_obs_buf[:, -1, 6:9] = vlm_cmd
            if isinstance(obs, torch.Tensor):
                obs[:, 6:9] = vlm_cmd
            else:
                obs["policy"][:, 6:9] = vlm_cmd

            with torch.inference_mode():
                low_action = policy(obs)
            obs, _, done_t, infos = env_w.step(low_action)

            cam = infos["observations"]["camera_obs"]
            for i in range(N):
                if done_envs[i]:
                    continue
                cur_steps[i] += 1
                if cur_steps[i] % steps_per_image == 0:
                    frame = cam[i, :, :, :3].cpu().numpy().astype(np.uint8)
                    image_observations[i].append(Image.fromarray(frame))
                d_i = bool(done_t[i].item()) if hasattr(done_t, "__getitem__") else bool(done_t)
                just_finished = False
                if d_i:
                    broke_by[i] = "done"; done_envs[i] = True; just_finished = True
                elif cur_steps[i] > max_steps:
                    broke_by[i] = "maxsteps"; done_envs[i] = True; just_finished = True
                if just_finished and i < real_n:  # don't write padding envs
                    _snapshot_and_write_async(i, chunk_episodes[i], meas_dir,
                                              env_w, n_vlm_calls[i], n_parse_fall[i],
                                              broke_by[i], _write_pool, _write_futures)

        t_chunk_end = time.perf_counter()
        print(f"[multi] chunk done in {t_chunk_end - t_chunk_start:.1f}s "
              f"(real_n={real_n})", flush=True)

    # ---- OUTER LOOP: iterate scene chunks, hot-swap matterport between scenes ----
    cur_scene = first_scene_id
    t_total_start = time.perf_counter()
    for chunk_idx, (sid, chunk_eps) in enumerate(scene_chunks):
        real_n = len(chunk_eps)
        # Pad to N envs if needed (last chunk of a scene may have <N)
        if real_n < N:
            chunk_eps = list(chunk_eps) + [chunk_eps[-1]] * (N - real_n)
        # Hot-swap matterport if scene changed
        if sid != cur_scene:
            t_sw = time.perf_counter()
            new_path = os.path.join(ASSETS_DIR, f"matterport_usd/{sid}/{sid}.usd")
            env_w.unwrapped.scene.terrain.swap_matterport(new_path)
            cur_scene = sid
            print(f"[multi] swap to {sid} in {time.perf_counter()-t_sw:.1f}s", flush=True)
        print(f"[multi] chunk {chunk_idx+1}/{len(scene_chunks)} scene={sid} real_n={real_n}", flush=True)
        _run_chunk(chunk_eps, real_n)

    # Drain all writes
    for f in _write_futures:
        try:
            f.result(timeout=30)
        except Exception as _e:
            print(f"[multi] WARN write fail: {_e}", flush=True)
    _write_pool.shutdown(wait=True)
    print(f"[multi] ALL DONE in {time.perf_counter() - t_total_start:.1f}s", flush=True)


if __name__ == "__main__":
    import traceback as _tb, sys as _sys, os as _o
    try:
        main()
        print("[multi] DONE.", flush=True)
    except SystemExit:
        raise
    except BaseException as _e:
        print(f"[multi] FATAL: {type(_e).__name__}: {_e}", flush=True)
        _tb.print_exc(file=_sys.stdout)
        _sys.stdout.flush()
        _sys.stderr.flush()
        _o._exit(1)
    _o._exit(0)
