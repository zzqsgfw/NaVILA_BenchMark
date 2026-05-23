"""Diagnostic: run go2_matterport_vision with zero vel_command for 200 steps,
report robot orientation and joint pos drift to detect ckpt/env mismatch."""
import argparse
import gymnasium as gym
import torch
import sys

# CLI is shared with isaaclab.app.AppLauncher
import cli_args
parser = argparse.ArgumentParser(description="Zero-cmd diagnostic.")
parser.add_argument("--task", type=str, default="go2_matterport_vision")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--history_length", type=int, default=9)
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=2000)
parser.add_argument("--use_cnn", action="store_true", default=None)
parser.add_argument("--arm_fixed", action="store_true", default=False)
parser.add_argument("--use_rnn", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)  # adds --load_run, --device, etc.
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from _meshcache import enable_local_mesh_cache
enable_local_mesh_cache()

import math
import inspect as _inspect
import rsl_rl as _rsl_rl
from rsl_rl.algorithms.ppo import PPO as _PPO
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
try:
    # only exists on newer IsaacLab (rsl-rl 3.x stack); absent on IL 2.2.1 / rsl-rl 2.3.3
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    _HAS_RSL_SHIM = True
except ImportError:
    _HAS_RSL_SHIM = False
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.vlnce.config import *  # noqa: F401, F403  registers tasks
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper
import gzip, json, os

# Pick episode
episodes = json.load(gzip.open(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")))["episodes"]
episode = episodes[args_cli.episode_idx]

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
env_cfg.scene_id = episode["scene_id"].split("/")[1]
env_cfg.scene.robot.init_state.pos = tuple(episode["start_position"])
env_cfg.scene.robot.init_state.rot = tuple(episode["start_rotation"])
env_cfg.episode_length_s = 1000.0
env_cfg.expert_path = episode["reference_path"]
udf = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
env_cfg.scene.terrain.obj_filepath = udf

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)

# Load policy
agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, agent_cfg.load_checkpoint)
if _HAS_RSL_SHIM:
    # new stack (rsl-rl 3.x): migrate deprecated cfg + strip kwargs PPO won't accept
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, getattr(_rsl_rl, "__version__", "3.0.1"))
    _ppo_kw = set(_inspect.signature(_PPO.__init__).parameters.keys()) | {"class_name"}
    acfg = agent_cfg.to_dict()
    acfg["algorithm"] = {k: v for k, v in acfg["algorithm"].items() if k in _ppo_kw}
else:
    # old stack (rsl-rl 2.3.3): original NaVILA-Bench usage, cfg passed as-is
    acfg = agent_cfg.to_dict()
runner = OnPolicyRunner(env, acfg, log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

robot = env.unwrapped.scene["robot"]
am = env.unwrapped.action_manager
print("\n========== DIAG ==========")
print("joint_names :", robot.joint_names)
print("default_jp  :", robot.data.default_joint_pos[0].cpu().tolist())
# Inspect action term offset / scale
for term_name, term in am._terms.items():
    print(f"action term '{term_name}': scale={getattr(term, '_scale', None)}, offset_shape={getattr(getattr(term, '_offset', None), 'shape', None)}")
    if hasattr(term, '_offset') and isinstance(term._offset, torch.Tensor):
        print(f"  offset[0]={term._offset[0].cpu().tolist()}")
# Inspect actuator
for act_name, act in robot.actuators.items():
    print(f"actuator '{act_name}': stiffness={act.stiffness[0].cpu().tolist() if hasattr(act.stiffness,'shape') else act.stiffness}")
    print(f"  damping={act.damping[0].cpu().tolist() if hasattr(act.damping,'shape') else act.damping}")
    print(f"  effort_limit={act.effort_limit[0].cpu().tolist() if hasattr(act.effort_limit,'shape') else act.effort_limit}")

obs_td, _ = env.reset()
print("initial root_quat (w,x,y,z):", robot.data.root_quat_w[0].cpu().tolist())
print("initial root_pos          :", robot.data.root_pos_w[0].cpu().tolist())
print("initial joint_pos         :", robot.data.joint_pos[0].cpu().tolist())

zero_cmd = torch.zeros(1, 3, device=env.unwrapped.device)
env.update_command(zero_cmd[0])

_scn = env.unwrapped.scene
print("=== SENSOR INTROSPECTION ===")
print("scene.sensors keys:", list(_scn.sensors.keys()))
for step in range(200):
    action = policy(obs_td)
    obs_td, _, _, _ = env.step(action)
    env.update_command(zero_cmd[0])
    if step in (30, 60, 120):
        for _sn in ("lidar_sensor", "height_scanner"):
            if _sn not in _scn.sensors:
                print(f"  [{_sn}] ABSENT"); continue
            _s = _scn.sensors[_sn]
            _h = _s.data.ray_hits_w
            _p = _s.data.pos_w
            _v = _h[0] - _p[0].unsqueeze(0)
            _d = _v.norm(dim=-1)
            _fin = torch.isfinite(_d)
            _real = _fin & (_d > 0.05)
            print(f"  step{step} [{_sn}] hits{tuple(_h.shape)} pos{tuple(_p.shape)} "
                  f"pos0={[round(x,2) for x in _p[0].tolist()]} "
                  f"d: finite={int(_fin.sum())}/{_d.numel()} real(>0.05)={int(_real.sum())} "
                  f"min_all={float(_d.min()):.4f} "
                  f"min_real={float(_d[_real].min()) if int(_real.sum())>0 else -1:.4f} "
                  f"max={float(_d[_fin].max()) if int(_fin.sum())>0 else -1:.3f}")
            if _sn == "lidar_sensor":
                # candidate fixed clearance: floor = lowest hit; keep body band
                _hw = _h[0][_fin]
                if _hw.shape[0] > 0:
                    _fz = float(_hw[:, 2].min())
                    _band = (_hw[:, 2] > _fz + 0.10) & (_hw[:, 2] < _fz + 1.5)
                    _vv = (_hw[_band] - _p[0].unsqueeze(0))
                    _hd = _vv[:, :2].norm(dim=-1)
                    _hd = _hd[_hd > 0.05]
                    print(f"    -> floor_z={_fz:.2f} band_hits={int(_band.sum())} "
                          f"CLEARANCE_min={float(_hd.min()) if _hd.numel()>0 else -1:.3f} "
                          f"mean={float(_hd.mean()) if _hd.numel()>0 else -1:.3f}")
    if step % 50 == 0 or step == 199:
        q = robot.data.root_quat_w[0]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        sinp = max(-1.0, min(1.0, float(2*(qw*qy - qz*qx))))
        pitch = math.asin(sinp)
        z = float(robot.data.root_pos_w[0, 2])
        raw = am.action[0].cpu().tolist()
        proc = am.get_term('joint_pos').processed_actions[0].cpu().tolist() if hasattr(am, 'get_term') else None
        jp = robot.data.joint_pos[0].cpu().tolist()
        print(f"step={step:3d} roll={math.degrees(roll):+.2f}° pitch={math.degrees(pitch):+.2f}° z={z:.3f}m")
        print(f"   raw_action[0]: {[f'{x:+.2f}' for x in raw]}")
        print(f"   joint_pos[0] : {[f'{x:+.2f}' for x in jp]}")
        if proc is not None:
            print(f"   target_pos[0]: {[f'{x:+.2f}' for x in proc]}")

print("=========================\n")
simulation_app.close()
