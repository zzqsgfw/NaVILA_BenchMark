# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import os
import json
import math
import torch
import numpy as np
import imageio
from PIL import Image
import time
import base64
import io
import socket
import json

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# isaaclab argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--visualize_path", action="store_true", default=False, help="Visualize the path in the simulator.")

# navila argparse arguments
# (--device is now provided by AppLauncher.add_app_launcher_args in IL HEAD)
parser.add_argument("--vlm_host", type=str, default="localhost")
parser.add_argument("--vlm_port", type=int, default=54321)


# r2r argparse arguments
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--episode_idx_list", type=str, default=None,
                    help="Comma-separated list of episode indices. If set, overrides --episode_idx "
                         "and runs them all in a single process (avoids the ~30-60s per-ep Isaac "
                         "app launch + USD load + python startup overhead).")
parser.add_argument("--skip_if_done", action="store_true", default=False,
                    help="Skip episodes whose measurement json already exists (for resume).")
parser.add_argument("--precook", action="store_true", default=False,
                    help="Only build the env (cooks + caches the matterport collision mesh) then hard-exit. One-time per scene.")
parser.add_argument("--lidar_constrain", action="store_true",
                    help="(B-A) clip forward time-to-go by lidar fwd clearance")
parser.add_argument("--safety_factor", type=float, default=0.7,
                    help="(B-A) multiplier on lidar clearance for the clip")
parser.add_argument("--loop_breaker_N", type=int, default=0,
                    help="(B-B) N>=2 enables loop-breaker; 0 disables")
parser.add_argument("--drift_thresh", type=float, default=0.15,
                    help="(B-B) xy drift threshold for the loop detector (m)")
parser.add_argument("--output_suffix", type=str, default="",
                    help="Suffix appended to eval_results/{task}_loco_{run} dir, e.g. '_lidar_sf07'.")

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))
from _meshcache import enable_local_mesh_cache
enable_local_mesh_cache()

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.io import load_yaml
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import update_class_from_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
import isaaclab.sim as sim_utils

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command, 
    read_episodes, 
    add_instruction_on_img,
    InstructionData, 
)
from omni.isaac.vlnce.utils.measures import PathLength, DistanceToGoal, Success, SPL, OracleNavigationError, OracleSuccess, MeasureManager


def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0**2 + q1**2 - q2**2 - q3**2)

    return roll, pitch, yaw


def define_markers() -> VisualizationMarkers:
    """Define path markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pathMarkers",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def reset_start_pos_rot(env_cfg, args_cli, episode):
    scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
    env_cfg.scene.terrain.obj_filepath = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    
    start_pos, start_rot, goal_pos = episode["start_position"], episode["start_rotation"], episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot

    if "go2" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.4)
    elif "h1" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+1.0)
    else:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.5)

    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    env_cfg.scene.disk_1.init_state.pos = ([start_pos[0], start_pos[1], start_pos[2] + 2.5])
    env_cfg.scene.disk_2.init_state.pos = ([goal_pos[0], goal_pos[1], goal_pos[2] + 2.5])

    return env_cfg


def add_measurement(env, episode):
    measure_manager = MeasureManager()
    measure_names = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    for measure_name in measure_names:
        measure = eval(measure_name)(env, episode, measure_manager)
        measure_manager.register_measure(measure)
    
    env.measure_manager = measure_manager
    return


def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query):
    if len(image_list) == 0:
        print("Did not receive any images.")
        return None
    elif len(image_list) < 8:
        print("Not enough images received, padding.")
        image_list = image_list.copy()
        # append image value=0, in front of the existing images, image size equal to the last one
        for _ in range(8 - len(image_list)):
            image_list.insert(0, Image.new('RGB', image_list[-1].size, (0, 0, 0)))
    else:
        image_list = image_list.copy()
        
    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])

    # save sampled images
    # time_stamp = time.strftime("%Y%m%d-%H%M%S")
    # if not os.path.exists("test_images"):
    #     os.makedirs("test_images")
    # for i, img in enumerate(sampled_images):
    #     # convert to PIL Image
    #     img = Image.fromarray(img)
    #     img.save(os.path.join("test_images", f"{time_stamp}_image_{i}.jpg"))

    # Convert images to base64 for transmission. Downsize to 224x224 first —
    # the VLM (SigLIP/CLIP) resizes to 224 anyway. Sending 512x512 wastes ~80%
    # of JPEG encode + transfer cost on every VLM call (8 frames × ~40 calls
    # per episode = ~2-5 s saved per ep).
    encoded_images = []
    for image in sampled_images:
        # Ensure PIL Image for JPEG encoding
        if isinstance(image, np.ndarray):
            array_image = image
            if array_image.dtype != np.uint8:
                # Convert to uint8. If values are 0-1, scale; otherwise clip to 0-255
                if array_image.max() <= 1.0:
                    array_image = (array_image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    array_image = array_image.clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array_image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            # Fallback: try to construct a PIL image from whatever object is provided
            pil_image = Image.fromarray(np.array(image, dtype=np.uint8))

        # Pre-resize to 224 (VLM target) — major JPEG/transfer-time saver.
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.BILINEAR)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # Prepare request data
    request_data = {
        'images': encoded_images,
        'query': query
    }

    # Send to VLM server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))
        
        # Send data
        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, 'big'))
        s.sendall(data_bytes)
        
        # Receive response
        size_data = s.recv(8)
        size = int.from_bytes(size_data, 'big')
        
        response_data = b''
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet
            
        response = json.loads(response_data.decode())
        return response


def _setup_isaac_for_scene(first_episode):
    """One-time per scene: parse cfg, build env, load policy.

    Returns (env_inner, ppo_runner, policy). ``env_inner`` is the rsl-rl-wrapped
    env BEFORE it is wrapped with VLNEnvWrapper (VLNEnvWrapper is per-episode
    because it stores episode-specific goal info / measure manager).
    """
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    # reset the position and rotation of the robot for the FIRST episode of this scene
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, first_episode)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, play=True
    )

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # update agent config with the one from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    # specify directory for logging experiments
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # wrap around environment for rsl-rl
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # Strip cfg keys that newer rsl-rl versions do not accept (only for the new
    # stack; the old IL 2.2.1 / rsl-rl 2.3.3 stack on volc3 lacks the shim and
    # the original NaVILA-Bench code path is used as-is).
    import inspect as _inspect
    import rsl_rl as _rsl_rl
    from rsl_rl.algorithms.ppo import PPO as _PPO
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, getattr(_rsl_rl, "__version__", "3.0.1"))
        _ppo_kw = set(_inspect.signature(_PPO.__init__).parameters.keys()) | {"class_name"}
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["algorithm"] = {k: v for k, v in agent_cfg_dict["algorithm"].items() if k in _ppo_kw}
    except ImportError:
        # old stack: cfg passed through unchanged, matches original NaVILA-Bench usage
        agent_cfg_dict = agent_cfg.to_dict()

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    return env, ppo_runner, policy


def _mutate_pose_for_episode(env_inner, episode):
    """In-place mutate robot + disk markers' default_root_state to the new episode's
    start/goal. Caller must subsequently call env.reset() so reset_root_state_uniform
    picks up the new defaults (and clears velocities + counters).

    Only valid when staying within the SAME scene (USD terrain unchanged).
    """
    device = env_inner.unwrapped.device

    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]  # [w, x, y, z]
    goal_pos = episode["reference_path"][-1]

    if "go2" in args_cli.task:
        z_off = 0.4
    elif "h1" in args_cli.task:
        z_off = 1.0
    else:
        z_off = 0.5

    robot = env_inner.unwrapped.scene["robot"]
    robot.data.default_root_state[0, 0:3] = torch.tensor(
        [start_pos[0], start_pos[1], start_pos[2] + z_off], device=device, dtype=robot.data.default_root_state.dtype
    )
    robot.data.default_root_state[0, 3:7] = torch.tensor(
        [start_rot[0], start_rot[1], start_rot[2], start_rot[3]], device=device, dtype=robot.data.default_root_state.dtype
    )
    robot.data.default_root_state[0, 7:13] = 0.0

    # disk_1 / disk_2 are AssetBaseCfg XformPrims (visual markers, no physics)
    # — they don't have .data.default_root_state. They're only used for the
    # video overlay. Visually wrong for ep 2+ in scene-batch, but doesn't
    # affect benchmark correctness (success / d2g use episode["reference_path"]
    # directly). Skipping mutation is OK.


def _run_episode_inner(env_inner, ppo_runner, policy, episode, ep_idx, is_first):
    """Run a single episode using an already-built (env_inner, policy).

    For is_first=False, mutate default_root_state to the new episode's start
    before reset; the wrapper's reset_root_state_uniform then teleports the
    robot + markers via the new defaults.

    Does NOT call env.close() — the env outlives the episode (lifetime = scene).
    """
    # Wrap the inner env with a fresh VLNEnvWrapper for THIS episode (carries
    # measure manager + goal info bound to this episode).
    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess", "CollisionRate", "TerminationReason", "ObstacleClearance"]
    env = VLNEnvWrapper(env_inner, policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)

    if not is_first:
        # Same scene, different episode: mutate scene-default root states so
        # the upcoming reset teleports robot + disk markers to the new pose.
        _mutate_pose_for_episode(env_inner, episode)

    # step with zeros actions to get the initial frame
    obs, infos = env.reset()

    # set view pos and target (per-episode because robot pose changes per ep)
    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
    cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
    # set the camera view
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    if args_cli.precook:
        # env build + reset has triggered & completed the matterport collision
        # cook, which (with useLocalMeshCache on) is now written to the on-disk
        # mesh cache. Nothing else needed; hard-exit to skip Isaac finalize hang.
        import os as _o
        print("[precook] cook done + cached; hard exit", flush=True)
        _o._exit(0)

    # NaViLA training gets image observations each 0.5s, visualize every 0.1s
    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    # init_frame = cv2.rotate(init_frame, cv2.ROTATE_90_CLOCKWISE)
    instruction = InstructionData(**episode["instruction"])
    image_observations = []
    image_observations.append(Image.fromarray(init_frame))

    add_instruction_on_img(init_frame, instruction.instruction_text)
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    # vis_frame = cv2.rotate(vis_frame, cv2.ROTATE_90_CLOCKWISE)
    add_instruction_on_img(vis_frame, "")
    rgb_obses = [np.concatenate([init_frame, vis_frame], axis=1)]

    num_steps = 0
    target_steps = 0
    same_pos_count = 0
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    max_episode_steps = 100 * 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    # ---- comprehensive per-episode trace (record once, analyse offline forever) ----
    trace_events = []      # one record per VLM call
    traj = []              # per-step [step, x, y, z, yaw, d2g]
    proprio = []           # per-step full robot state (quat,base vel,joint pos/vel,contacts)
    n_vlm_calls = 0
    n_parse_fall = 0
    broke_by = "running"
    parse_info = {"action": "none", "mag_matched": False, "fallthrough": False}

    # Problem B interventions (decode-time, embodiment-aware).
    # Import once per episode (cheap), only used when flags enabled.
    if args_cli.loop_breaker_N >= 2 or args_cli.lidar_constrain:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from _inline_intervention_patch import LoopBreaker, lidar_constrain_command
        loop_breaker = LoopBreaker(N=args_cli.loop_breaker_N or 999,
                                   drift_thresh_m=args_cli.drift_thresh)
    else:
        loop_breaker = None

    # Profiling counters. Print as one line at episode end via [prof]. Cheap
    # — only adds 6 perf_counter calls per iter + 1 final print.
    _PROF = os.environ.get("NAVILA_PROF") == "1"
    if _PROF:
        _t_step = 0.0   # env.step()
        _t_vlm = 0.0    # sample_images_and_send_to_vlm
        _t_cap = 0.0    # camera frame capture (image_observations + viz)
        _t_traj = 0.0   # per-step traj sample (post-step)
        _t_stuck = 0.0  # same-pos check
        _t_loop_start = time.perf_counter()

    # visualizer = define_markers()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if num_steps == target_steps:
                _t0 = time.perf_counter() if _PROF else 0
                stream_output = sample_images_and_send_to_vlm(image_observations, args_cli.vlm_host, args_cli.vlm_port, instruction.instruction_text)
                if _PROF: _t_vlm += time.perf_counter() - _t0
                vlm_vel_commands, time_to_go, parse_info = get_vel_command(stream_output)

                # (B-B) loop-breaker: override stuck VLM output before clip
                if args_cli.loop_breaker_N >= 2 and loop_breaker is not None:
                    _rp_now = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                    trig, ov_vel, ov_ttg = loop_breaker.step(stream_output, (_rp_now[0], _rp_now[1]))
                    if trig:
                        vlm_vel_commands, time_to_go = ov_vel, ov_ttg
                        parse_info = dict(parse_info, loop_breaker_fired=True)

                # (B-A) lidar-constrained clip for forward commands
                if args_cli.lidar_constrain:
                    vlm_vel_commands, time_to_go, parse_info, clip_meta = lidar_constrain_command(
                        stream_output, vlm_vel_commands, time_to_go, parse_info, env,
                        safety_factor=args_cli.safety_factor,
                    )
                    parse_info = dict(parse_info, clip=clip_meta)

                env_steps_to_go = int(time_to_go / (
                    env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
                ))
                target_steps = num_steps + env_steps_to_go
                print(f"VLM output: {stream_output}\nVel Command: {vlm_vel_commands}, Env Steps to go: {env_steps_to_go}\n")

                # record this VLM call into the trace
                n_vlm_calls += 1
                if parse_info["fallthrough"]:
                    n_parse_fall += 1
                _rp = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                _rq = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                _, _, _yaw = quat2eulers(_rq[0], _rq[1], _rq[2], _rq[3])
                _meas = infos.get("measurements", {}) if isinstance(infos, dict) else {}
                trace_events.append({
                    "step": int(num_steps),
                    "raw": stream_output,
                    "vel": [float(v) for v in vlm_vel_commands],
                    "time_to_go": float(time_to_go),
                    "env_steps_to_go": int(env_steps_to_go),
                    "parse": parse_info,
                    "x": float(_rp[0]), "y": float(_rp[1]), "z": float(_rp[2]),
                    "yaw": float(_yaw),
                    "d2g": float(_meas.get("distance_to_goal", -1.0)),
                    "n_frames_avail": len(image_observations),  # frames the VLM saw this call
                })

        _t0 = time.perf_counter() if _PROF else 0
        obs, _, done, infos = env.step(torch.tensor(vlm_vel_commands, device = obs.device))
        if _PROF: _t_step += time.perf_counter() - _t0

        # per-step trajectory sample (cheap; lets any future metric be recomputed offline)
        # Sample at most every 5 steps to cut GPU→CPU sync overhead (~14 syncs/step).
        # vlm-call events are still recorded separately via trace_events.
        # `done` may be a scalar Python bool (from VLNEnvWrapper) or a tensor.
        _t0 = time.perf_counter() if _PROF else 0
        _done_flag = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        if (int(num_steps) % 5 == 0) or _done_flag:
            _p = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            _q = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            _, _, _y = quat2eulers(_q[0], _q[1], _q[2], _q[3])
            _m = infos.get("measurements", {}) if isinstance(infos, dict) else {}
            traj.append([int(num_steps), float(_p[0]), float(_p[1]), float(_p[2]),
                         float(_y), float(_m.get("distance_to_goal", -1.0))])
        # NOTE: per-step proprio dump removed — was ~14 GPU→CPU syncs/step × 1400 steps
        # = 20k syncs/ep. None of our current metrics use it; can be re-enabled by
        # setting NAVILA_DUMP_PROPRIO=1.
        if os.environ.get("NAVILA_DUMP_PROPRIO") == "1":
            try:
                _rd = env.unwrapped.scene["robot"].data
                _q = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                _row = np.concatenate([
                    np.array([int(num_steps)], dtype=np.float32),
                    _q.astype(np.float32),
                    _rd.root_lin_vel_b[0].detach().cpu().numpy().astype(np.float32),
                    _rd.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32),
                    _rd.joint_pos[0].detach().cpu().numpy().astype(np.float32),
                    _rd.joint_vel[0].detach().cpu().numpy().astype(np.float32),
                ])
                try:
                    _cf = env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w[0]
                    _row = np.concatenate([_row, _cf.norm(dim=-1).detach().cpu().numpy().astype(np.float32)])
                except Exception:
                    pass
                proprio.append(_row)
            except Exception:
                pass

        if _PROF: _t_traj += time.perf_counter() - _t0

        if done or env.is_stop_called or num_steps > max_episode_steps:
            broke_by = "done" if done else ("stop" if env.is_stop_called else "maxsteps")
            break

        _t0 = time.perf_counter() if _PROF else 0
        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_vel = np.linalg.norm(env.unwrapped.scene["robot"].data.root_vel_w[0].detach().cpu().numpy())
        if np.linalg.norm(cur_pos - prev_pos) < 0.01 and robot_vel < 0.01:
            same_pos_count += 1
        else:
            same_pos_count = 0
        prev_pos = cur_pos
        if _PROF: _t_stuck += time.perf_counter() - _t0

        # Break out of the loop if the robot has stayed in the same location for 500 steps
        if same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            broke_by = "stuck"
            break

        _t0 = time.perf_counter() if _PROF else 0
        if num_steps % steps_per_image == 0:
            curr_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            image_observations.append(Image.fromarray(curr_frame))
            curr_frame_copy = curr_frame.copy()
            add_instruction_on_img(curr_frame_copy, instruction.instruction_text)
            
        if num_steps % steps_per_viz_image == 0:
            curr_vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            add_instruction_on_img(curr_vis_frame, stream_output)
            rgb_obses.append(np.concatenate([curr_frame_copy, curr_vis_frame], axis=1))
        if _PROF: _t_cap += time.perf_counter() - _t0

        num_steps += 1
        if env_steps_to_go == 0:
            env.set_stop_called(True)

        # if args_cli.visualize_path:
        #     visualizer.visualize(reference_path_isaac)

    if _PROF:
        _wall = time.perf_counter() - _t_loop_start
        _other = _wall - _t_step - _t_vlm - _t_cap - _t_traj - _t_stuck
        print(f"[prof] ep_idx={ep_idx} steps={num_steps} vlm_calls={n_vlm_calls} "
              f"wall={_wall:.1f}s | step={_t_step:.1f}s ({100*_t_step/_wall:.0f}%) "
              f"vlm={_t_vlm:.1f}s ({100*_t_vlm/_wall:.0f}%) "
              f"cap={_t_cap:.1f}s ({100*_t_cap/_wall:.0f}%) "
              f"traj={_t_traj:.1f}s ({100*_t_traj/_wall:.0f}%) "
              f"stuck={_t_stuck:.1f}s ({100*_t_stuck/_wall:.0f}%) "
              f"other={_other:.1f}s ({100*_other/_wall:.0f}%)", flush=True)

    measurements = infos["measurements"]
    # parse-failure stats into the lightweight summary (class-(c) signal + IPSR)
    measurements["n_vlm_calls"] = int(n_vlm_calls)
    measurements["n_parse_fallthrough"] = int(n_parse_fall)
    measurements["parse_fail_rate"] = (n_parse_fall / n_vlm_calls) if n_vlm_calls else 0.0
    measurements["broke_by"] = broke_by

    result_dir = f"eval_results/{args_cli.task}_loco_{args_cli.load_run}{args_cli.output_suffix}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    measurement_dir = os.path.join(result_dir, "measurements")
    if not os.path.exists(measurement_dir):
        os.makedirs(measurement_dir)
    eid = int(episode['episode_id']) - 1
    with open(f"{measurement_dir}/{eid}.json", "w") as f:
        json.dump(measurements, f, indent=4)

    # ---- comprehensive trace: everything needed for ANY future offline analysis
    # (failure decomposition, drift, parse attribution, arbiter design) so the
    # expensive Isaac run never has to be repeated for a new analysis need ----
    trace_dir = os.path.join(result_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    try:
        clr_series = list(env.measure_manager.measures["obstacle_clearance"]._step_min)
    except Exception:
        clr_series = []
    trace = {
        "episode_id": episode["episode_id"],
        "episode_idx": ep_idx,
        "scene_id": episode.get("scene_id"),
        "instruction": episode["instruction"]["instruction_text"],
        "start_position": episode.get("start_position"),
        "goals": episode.get("goals"),
        "gt_locations": episode.get("gt_locations"),
        "task": args_cli.task,
        "low_level_run": args_cli.load_run,
        "summary": measurements,
        "termination_reason": measurements.get("termination_reason"),
        "broke_by": broke_by,
        "num_steps": int(num_steps),
        "vlm_events": trace_events,
        "clearance_step_series": clr_series,
        # exact VLM input is: this server prompt template, with {n-1} history +
        # 1 current frame sampled from the recorded frame sequence (frames.npz),
        # and {instruction}. n_frames_avail per event = frames the server saw.
        "vlm_prompt_template": (
            "Imagine you are a robot programmed for navigation tasks. You have been "
            "given a video of historical observations <image>*({num_video_frames}-1), "
            "and current observation <image>. Your assigned task is: \"{instruction}\" "
            "Analyze this series of images to decide your next action, which could be "
            "turning left or right by a specific degree, moving forward a certain "
            "distance, or stop if the task is completed."
        ),
        "vlm_num_video_frames": 8,
        "proprio_cols": (["step", "qw", "qx", "qy", "qz",
                          "vlin_x", "vlin_y", "vlin_z", "vang_x", "vang_y", "vang_z"]
                         + [f"jpos_{i}" for i in range(12)]
                         + [f"jvel_{i}" for i in range(12)]
                         + ["contact_body_norms..."]),
    }
    with open(os.path.join(trace_dir, f"{eid}.json"), "w") as f:
        json.dump(trace, f)
    if len(traj) > 0:
        _npz = dict(
            traj=np.asarray(traj, dtype=np.float32),  # [N,6]: step,x,y,z,yaw,d2g
            traj_cols=np.array(["step", "x", "y", "z", "yaw", "d2g"]),
            clearance=np.asarray(clr_series, dtype=np.float32),
        )
        if len(proprio) > 0:
            # rows can differ in length only if contact body count varies; pad-safe
            _w = max(len(r) for r in proprio)
            _pr = np.full((len(proprio), _w), np.nan, dtype=np.float32)
            for _i, _r in enumerate(proprio):
                _pr[_i, :len(_r)] = _r
            _npz["proprio"] = _pr
        # NaN-padded npz can be a few MB on shared disk; default uncompressed
        # savez is ~3-5x faster on NFS for medium tensors. Toggle compression
        # back on with NAVILA_TRACE_COMPRESS=1.
        if os.environ.get("NAVILA_TRACE_COMPRESS") == "1":
            np.savez_compressed(os.path.join(trace_dir, f"{eid}.npz"), **_npz)
        else:
            np.savez(os.path.join(trace_dir, f"{eid}.npz"), **_npz)
    # exact frames fed to the VLM over the episode (JPEG, ~30KB ea) so every
    # VLM input can be reconstructed offline without re-running Isaac.
    # Heavy: ~5-15s/ep on shared disk. Defaults to OFF to keep the benchmark
    # dispatch fast; opt-in with NAVILA_DUMP_FRAMES=1 if the npz is needed.
    if os.environ.get("NAVILA_DUMP_FRAMES") == "1":
        try:
            import io
            _buf = []
            for _im in image_observations:
                _b = io.BytesIO(); _im.save(_b, format="JPEG", quality=85)
                _buf.append(np.frombuffer(_b.getvalue(), dtype=np.uint8))
            np.savez(
                os.path.join(trace_dir, f"{eid}_frames.npz"),
                frames=np.array(_buf, dtype=object),
            )
        except Exception as _e:
            print(f"[trace] frame dump skipped: {_e}")


    # MP4 video write — heavy H264 encode of ~280 1024×512 frames (~3-5s/ep).
    # Defaults to OFF for the benchmark dispatch; opt-in with NAVILA_WRITE_VIDEO=1.
    if os.environ.get("NAVILA_WRITE_VIDEO") == "1":
        video_dir = os.path.join(result_dir, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        writer = imageio.get_writer(f"{video_dir}/output_{int(episode['episode_id'])-1}.mp4", fps=10)
        for frame in rgb_obses:
            frame = frame.astype(np.uint8)
            writer.append_data(frame)

        writer.close()

    # NOTE: do NOT call env.close() here — env_inner is reused across episodes
    # within the same scene. Cleanup happens via simulation_app.close() at
    # process exit (one process per scene; see main()).


def main():
    """Drive one or many episodes in a single Isaac Sim process.

    --episode_idx_list "i,j,k,..." lets one navila_eval process run a batch of
    episodes from the SAME SCENE, paying the Isaac Sim cold-start + USD load
    cost only ONCE (vs. once per episode). Cross-scene batching is NOT
    supported in-process (matterport USD cannot be hot-swapped, and isaac's
    gym.make hangs on second invocation in the same Python process) — the
    caller (run_benchmark.py) must dispatch one subprocess per scene.

    With --skip_if_done, episodes whose measurement json already exists are
    skipped (resume).
    """
    import traceback as _tb
    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)

    if args_cli.episode_idx_list:
        ep_indices = [int(x) for x in args_cli.episode_idx_list.split(",") if x.strip()]
    else:
        ep_indices = [args_cli.episode_idx]

    # Resolve result_dir for resume-skip (matches the path used inside _run_episode_inner).
    result_dir = os.path.join(os.path.dirname(__file__), "..",
                              f"eval_results/{args_cli.task}_loco_{args_cli.load_run}{args_cli.output_suffix}")
    measurement_dir = os.path.join(result_dir, "measurements")

    # Filter to-run set (after skip_if_done).
    todo = []
    for i, ep_idx in enumerate(ep_indices):
        if args_cli.skip_if_done:
            eid = int(all_episodes[ep_idx]['episode_id']) - 1
            if os.path.exists(f"{measurement_dir}/{eid}.json"):
                print(f"[batch] ({i+1}/{len(ep_indices)}) ep_idx={ep_idx} eid={eid}: SKIP (done)", flush=True)
                continue
        todo.append(ep_idx)

    if not todo:
        print("[batch] nothing to run (all skipped).", flush=True)
        return

    # Sanity-check: all eps in todo must share scene_id (run_benchmark.py groups
    # by scene before dispatching; in-process scene swap is not supported).
    def _sid(ep):
        return os.path.splitext(os.path.basename(ep["scene_id"]))[0]
    first_sid = _sid(all_episodes[todo[0]])
    for ep_idx in todo[1:]:
        s = _sid(all_episodes[ep_idx])
        if s != first_sid:
            print(f"[batch] ABORT: mixed scene_ids in one process "
                  f"(first={first_sid}, ep_idx={ep_idx}->{s}). "
                  f"Dispatch must group by scene.", flush=True)
            return

    print(f"[batch] scene={first_sid}: driving {len(todo)} episodes in this process "
          f"(of {len(ep_indices)} requested)", flush=True)

    # One-time scene setup (gym.make + policy load) using FIRST episode's pose.
    env_inner, ppo_runner, policy = _setup_isaac_for_scene(all_episodes[todo[0]])

    for i, ep_idx in enumerate(todo):
        is_first = (i == 0)
        print(f"[batch] ({i+1}/{len(todo)}) ep_idx={ep_idx}: START", flush=True)
        try:
            _run_episode_inner(env_inner, ppo_runner, policy,
                               all_episodes[ep_idx], ep_idx, is_first=is_first)
        except Exception:
            print(f"[batch] ep_idx={ep_idx}: FAILED", flush=True)
            _tb.print_exc()
            # Continue to next ep — env_inner is still alive; the wrapper
            # construction was the only per-ep mutable state.
    print(f"[batch] all done.", flush=True)
    # env_inner is NOT closed — simulation_app.close() at process exit will
    # handle Isaac shutdown.


if __name__ == "__main__":
    try:
        main()
    finally:
        import os as _os
        _os._exit(0)
