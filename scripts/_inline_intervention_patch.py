"""Reference inline intervention patches for navila_eval.py.

DO NOT IMPORT — this is a *reference* file documenting the minimal edits
needed to add `--lidar_constrain` and `--loop_breaker_N` flags to the live
Isaac Lab evaluation loop. Apply the edits manually to navila_eval.py once
the offline ablations validate the hyper-parameters.

The two helpers below (`lidar_forward_clearance`, `LoopBreaker`) are
sim-side; they read from the env that navila_eval.py already constructs.
"""
import math
import numpy as np
import torch


# -----------------------------------------------------------------------------
# (A) lidar-feasibility constrained decoding
# -----------------------------------------------------------------------------
def lidar_forward_clearance(env, cone_deg: float = 30.0, sensor_name: str = "lidar_sensor") -> float:
    """Return the minimum lidar range inside a forward cone of `cone_deg` half-angle.

    Reads the same `lidar_sensor` raycaster ObstacleClearance uses.  Returns
    `+inf` if the sensor / data is missing — caller must treat as 'no clip'.
    """
    try:
        sensor = env.unwrapped.scene.sensors[sensor_name]
        hits = sensor.data.ray_hits_w[0]          # [num_rays, 3] world
        origin = sensor.data.pos_w[0]             # [3] world
        root_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    except Exception:
        return float("inf")

    finite = torch.isfinite(hits).all(dim=-1)
    if finite.sum() == 0:
        return float("inf")
    hits = hits[finite]

    # body forward direction in world frame (yaw from quat w,x,y,z)
    w, x, y, z = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
    yaw = math.atan2(2 * (x * y + w * z).item(),
                     (w * w + x * x - y * y - z * z).item())
    fwd = torch.tensor([math.cos(yaw), math.sin(yaw)], device=hits.device)

    # filter to the body band (drop floor + ceiling), as ObstacleClearance does
    floor_z = float(hits[:, 2].min())
    band = (hits[:, 2] > floor_z + 0.10) & (hits[:, 2] < floor_z + 1.5)
    if band.sum() == 0:
        return float("inf")
    hits = hits[band]

    rel = hits[:, :2] - origin[:2].unsqueeze(0)
    dist = rel.norm(dim=-1)
    dist = torch.where(dist > 0.05, dist, torch.full_like(dist, float("inf")))

    cos = (rel @ fwd) / dist.clamp(min=1e-6)
    cone = cos > math.cos(math.radians(cone_deg))
    if cone.sum() == 0:
        return float("inf")
    return float(dist[cone].min())


def lidar_constrain_command(raw_text: str, vlm_vel_commands, time_to_go,
                            parse_info, env, safety_factor: float = 0.7):
    """Clip a move_forward time_to_go if the forward cone is tight.

    Returns (vel, time_to_go, parse_info, clip_meta).  Only mutates
    move_forward; turn / stop pass through.  `vlm_vel_commands` is the
    [vx, vy, wz] vector get_vel_command returned.
    """
    if parse_info.get("action") != "move_forward":
        return vlm_vel_commands, time_to_go, parse_info, {"clipped": False}
    clr = lidar_forward_clearance(env)
    vx = vlm_vel_commands[0]  # 0.5 m/s in current parser
    orig_dist_m = vx * time_to_go
    budget_m = safety_factor * clr
    if budget_m >= orig_dist_m:
        return vlm_vel_commands, time_to_go, parse_info, {
            "clipped": False, "clr_fwd": clr, "orig_dist_m": orig_dist_m,
        }
    new_dist_m = max(0.0, budget_m)
    new_time_to_go = new_dist_m / max(vx, 1e-6)
    return vlm_vel_commands, new_time_to_go, parse_info, {
        "clipped": True, "clr_fwd": clr, "orig_dist_m": orig_dist_m,
        "new_dist_m": new_dist_m, "safety_factor": safety_factor,
    }


# -----------------------------------------------------------------------------
# (B) grounded loop-breaker
# -----------------------------------------------------------------------------
class LoopBreaker:
    """Detect (raw==prev AND ||xy-prev_xy||<drift_thresh) for N consecutive calls.

    When triggered, override `vlm_vel_commands` to a sampled alternative
    (default: 90-deg turn in the opposite direction of the stuck action).
    """

    def __init__(self, N: int = 3, drift_thresh_m: float = 0.15):
        self.N = N
        self.drift = drift_thresh_m
        self._prev_raw = None
        self._prev_xy = None
        self._repeat = 1

    def step(self, raw_text: str, xy):
        """Returns (trigger, override_vel, override_time_to_go).

        trigger=True means the VLM output should be discarded for this call.
        """
        if self._prev_raw is None:
            self._prev_raw, self._prev_xy = raw_text, xy
            return False, None, None
        dxy = math.hypot(xy[0] - self._prev_xy[0], xy[1] - self._prev_xy[1])
        if raw_text == self._prev_raw and dxy < self.drift:
            self._repeat += 1
        else:
            self._repeat = 1
        self._prev_raw, self._prev_xy = raw_text, xy

        if self._repeat < self.N:
            return False, None, None
        self._repeat = 1  # reset after firing

        # override: pick the opposite-class action. cheap rule -- if stuck
        # turning, do a long forward; if stuck moving, do a 90deg turn.
        rl = raw_text.lower()
        if "turn left" in rl:
            return True, [0.0, 0.0, -math.pi / 6.0], 1.5    # 45 deg right
        if "turn right" in rl:
            return True, [0.0, 0.0,  math.pi / 6.0], 1.5    # 45 deg left
        if "move" in rl:
            return True, [0.0, 0.0,  math.pi / 6.0], 1.5    # break with 45 deg turn
        return True, [0.0, 0.0,  math.pi / 6.0], 1.5


# -----------------------------------------------------------------------------
# Suggested edits to scripts/navila_eval.py
# -----------------------------------------------------------------------------
"""
1) After the existing argparse block:

    parser.add_argument("--lidar_constrain", action="store_true",
                        help="(A) clip forward time-to-go by lidar fwd clearance")
    parser.add_argument("--safety_factor", type=float, default=0.7,
                        help="(A) multiplier on lidar clearance for the clip")
    parser.add_argument("--loop_breaker_N", type=int, default=0,
                        help="(B) N>=2 enables loop-breaker; 0 disables")
    parser.add_argument("--drift_thresh", type=float, default=0.15,
                        help="(B) xy drift threshold for the loop detector (m)")

2) Just before the `while simulation_app.is_running():` loop, instantiate B:

    from _inline_intervention_patch import LoopBreaker, lidar_constrain_command
    loop_breaker = LoopBreaker(N=args_cli.loop_breaker_N or 999,
                               drift_thresh_m=args_cli.drift_thresh)

3) Inside the loop, AFTER `vlm_vel_commands, time_to_go, parse_info = get_vel_command(stream_output)`
   (line ~370 in current navila_eval.py), BEFORE building env_steps_to_go:

    # (B) loop-breaker: override stuck VLM output
    if args_cli.loop_breaker_N >= 2:
        _rp_now = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        trig, ov_vel, ov_ttg = loop_breaker.step(stream_output, (_rp_now[0], _rp_now[1]))
        if trig:
            vlm_vel_commands, time_to_go = ov_vel, ov_ttg
            parse_info = dict(parse_info, loop_breaker_fired=True)

    # (A) lidar-constrained clip for forward commands
    if args_cli.lidar_constrain:
        vlm_vel_commands, time_to_go, parse_info, clip_meta = lidar_constrain_command(
            stream_output, vlm_vel_commands, time_to_go, parse_info, env,
            safety_factor=args_cli.safety_factor,
        )
        parse_info = dict(parse_info, clip=clip_meta)

   THEN proceed with the existing `env_steps_to_go = int(time_to_go / ...)` line.

4) The existing trace event recording already serialises `parse_info`, so the
   intervention metadata is captured automatically — no schema change needed.
"""
