from typing import Any, Dict, List, Optional, Tuple, Union
from numpy import ndarray

import numpy as np
from scipy.spatial import KDTree


def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


class Measure:
    """Represents a measure that provides measurement on top of environment
    and task.

    :data _metric: metric for the :ref:`Measure`, this has to be updated with
        each :ref:`step() <env.Env.step()>` call on :ref:`env.Env`.

    This can be used for tracking statistics when running experiments. The
    user of this class needs to implement the :ref:`reset_metric()` and
    :ref:`update_metric()` method

    """

    _metric: Any
    uuid: str

    def __init__(self, env, episode, **kwargs: Any) -> None:
        self._env = env
        self._episode = episode
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        """Reset the metric for :ref:`Measure`"""
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Update :ref:`_metric`, this method is called from :ref:`env.Env`
        on each :ref:`step() <env.Env.step()>`
        """
        raise NotImplementedError

    def get_metric(self):
        r"""..

        :return: the current metric for :ref:`Measure`.
        """
        return self._metric
    
    def get_robot_position(self):
        robot_pos_w = self._env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        return robot_pos_w
    

class MeasureManager:
    """A manager class for handling different measures and dependencies."""
    def __init__(self):
        self.measures = {}

    def register_measure(self, measure):
        """Register a new measure."""
        self.measures[measure._get_uuid()] = measure

    def get_measure(self, measure_uuid):
        """Get a measure by its uuid."""
        return self.measures.get(measure_uuid)

    def check_measure_dependencies(self, measure_uuid, dependencies):
        """
        Check if all required dependencies for the measure are initialized.
        :param measure_uuid: The UUID of the measure being checked.
        :param dependencies: List of dependent measure UUIDs.
        """
        for dependency_uuid in dependencies:
            if dependency_uuid not in self.measures:
                raise Exception(f"Dependency {dependency_uuid} is missing for measure {measure_uuid}.")
            
    def reset_measures(self, *args: Any, **kwargs: Any):
        """Reset all measures."""
        for measure in self.measures.values():
            measure.reset_metric(*args, **kwargs)
    
    def update_measures(self, *args: Any, **kwargs: Any):
        """Update all measures."""
        for measure in self.measures.values():
            measure.update_metric(*args, **kwargs)

    def get_measurements(self):
        """Get metrics for all measures."""
        return {measure._get_uuid(): measure.get_metric() for measure in self.measures.values()}

    

class PathLength(Measure):
    """Path Length (PL)
    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    def __init__(self, env, episode, measure_manager, **kwargs: Any):
        super().__init__(env, episode, **kwargs)
        self.measure_manager = measure_manager

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self.get_robot_position()
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self.get_robot_position()
        self._metric += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, env, episode, *args: Any, **kwargs: Any
    ):
        super().__init__(env, episode, **kwargs)

        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._gt_waypoints: Optional[
            List[Tuple[float, float, float]]
        ] = episode["gt_locations"]
        self._kdtree = KDTree(self._gt_waypoints)
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = None
        self.update_metric(*args, **kwargs)  # type: ignore

    def distance_to_goal(self, current_position):
        
        # Find the closest waypoint to the current position
        closest_distance, closest_waypoint_idx = self._kdtree.query(current_position)
        
        # Initialize the total distance with the distance from the robot to the closest waypoint
        total_distance = closest_distance
        
        # Add the distance between waypoints from the closest waypoint to the goal
        for i in range(closest_waypoint_idx, len(self._gt_waypoints) - 1):
            total_distance += euclidean_distance(self._gt_waypoints[i], self._gt_waypoints[i + 1])
    
        return total_distance

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self.get_robot_position()

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            distance_to_target = self.distance_to_goal(current_position)

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    cls_uuid: str = "spl"

    def __init__(self, env, episode, measure_manager: MeasureManager, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):

        self._previous_position = self.get_robot_position()
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = self.measure_manager.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            measure_manager=self.measure_manager,
            *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, *args: Any, **kwargs: Any
    ):
        ep_success = self.measure_manager.measures[Success.cls_uuid].get_metric()

        current_position = self.get_robot_position()
        self._agent_episode_distance += euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(self, env, episode, measure_manager, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self._success_distance = episode["goals"][0]["radius"]
        self.measure_manager = measure_manager

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self.update_metric(*args, **kwargs)  # type: ignore
        setattr(self._env, "is_stop_called", False)

    def update_metric(self, *args: Any, **kwargs: Any):
        distance_to_target = self.measure_manager.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(self._env, "is_stop_called")
            and self._env.is_stop_called  # type: ignore
            and distance_to_target < self._success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def __init__(self, env, episode, measure_manager, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self.measure_manager.check_measure_dependencies(
            self.cls_uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        distance_to_target = self.measure_manager.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, env, episode, measure_manager: MeasureManager, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager
        self._success_distance = episode["goals"][0]["radius"]

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self.measure_manager.check_measure_dependencies(
            self.cls_uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        d = self.measure_manager.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self._success_distance)


class CollisionRate(Measure):
    """Collision Rate (CR).

    Per-episode binary flag: 1.0 if the robot's torso (``base`` body) ever
    registered a contact force above ``threshold`` during the episode, else 0.0.
    Aggregated over episodes this is the collision rate. Mirrors the
    ``base_contact`` termination logic (threshold 1.0) so CR is consistent with
    why the episode ends. Also tracks the number of colliding steps for
    intensity analysis.
    """

    cls_uuid: str = "collision"

    def __init__(self, env, episode, measure_manager, threshold: float = 1.0, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager
        self._threshold = threshold
        self._base_body_ids = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _contact_sensor(self):
        return self._env.unwrapped.scene.sensors["contact_forces"]

    def reset_metric(self, *args: Any, **kwargs: Any):
        cs = self._contact_sensor()
        # Resolve the torso/base body index once. Quadruped (Go2) calls it "base";
        # humanoid (H1) has no "base" — try torso_link / pelvis instead.
        ids = None
        for name in ("base", "torso_link", "pelvis"):
            try:
                ids, _ = cs.find_bodies(name)
                if ids:
                    break
            except ValueError:
                continue
        if not ids:
            raise ValueError(
                f"CollisionRate: no recognised base body in {cs.body_names}; "
                "extend the candidate list in measures.py:CollisionRate.reset_metric")
        self._base_body_ids = ids
        self._collided = False
        self._n_collision_steps = 0
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        cs = self._contact_sensor()
        # net_forces_w_history: [num_envs, history, num_bodies, 3]
        net = cs.data.net_forces_w_history
        force = net[:, :, self._base_body_ids].norm(dim=-1)  # [N, hist, n_base]
        max_force = force.max(dim=1)[0].max(dim=-1)[0]  # per-env scalar
        if float(max_force[0]) > self._threshold:
            self._collided = True
            self._n_collision_steps += 1
        self._metric = 1.0 if self._collided else 0.0


class TerminationReason(Measure):
    """Latches WHY the episode ended, read from the env TerminationManager.

    Values: "collision" (base_contact), "fell_over" (bad_orientation),
    "timeout" (time_out), or "none" (ended by stop / stuck / max steps with no
    sim termination). Combine with the ``success`` metric in post-analysis to
    get the success / collision / fell / timeout breakdown for the CR-SPL study.
    """

    cls_uuid: str = "termination_reason"

    # checked in priority order; first fired term is latched
    _TERM_MAP = [
        ("base_contact", "collision"),
        ("bad_orientation", "fell_over"),
        ("time_out", "timeout"),
    ]

    def __init__(self, env, episode, measure_manager, *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = "none"

    def update_metric(self, *args: Any, **kwargs: Any):
        if self._metric != "none":
            return  # already latched
        tm = self._env.unwrapped.termination_manager
        active = set(tm.active_terms)
        for term_name, reason in self._TERM_MAP:
            if term_name in active and bool(tm.get_term(term_name)[0]):
                self._metric = reason
                return


class ObstacleClearance(Measure):
    """Continuous obstacle-clearance safety metric.

    Binary CR (torso contact) is too coarse: a quadruped brushing a doorframe
    with a leg, or wedging against a wall without a >1.0 base hit, registers
    nothing. This measure reads the body-mounted ``lidar_sensor`` raycaster
    (the same one feeding the policy) and, every step, computes the minimum
    HORIZONTAL distance to an obstacle at body height.

    Floor exclusion is floor-adaptive: floor_z = lowest finite lidar hit this
    step; only hits in the band [floor_z+0.10, floor_z+1.5] m count (the
    sensor sits only ~0.2-0.35 m above the floor, so a fixed +/-dz band around
    the sensor leaks floor hits straight down and pins the min to ~0 — that
    was the original bug, verified via diag_zero_cmd lidar introspection).

    Reported (dict): episode min of per-step clearance, p5 of per-step
    clearance (robust "boxed-in" indicator; raw min is noisy, pinned by a few
    grazing rays), mean, and frac of steps below 0.3 m (near-miss dwell). p5
    is the principled criterion for selecting / building the crash-test split.

    Degrades safely: sensor absent or a step with no valid band hit is
    skipped; an episode with no valid reading reports -1.0.
    """

    cls_uuid: str = "obstacle_clearance"

    _NEAR = 0.3            # m, near-miss threshold
    _FLOOR_MARGIN = 0.10   # m above the adaptive floor to start the body band
    _BODY_TOP = 1.5        # m above the adaptive floor to end the body band
    _MIN_RAY = 0.05        # m, drop degenerate/self rays

    def __init__(self, env, episode, measure_manager, sensor_name: str = "lidar_sensor",
                 *args: Any, **kwargs: Any):
        super().__init__(env, episode)
        self.measure_manager = measure_manager
        self._sensor_name = sensor_name

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._step_min = []     # per-step min clearance
        self._n_close = 0
        self._metric = {"min": -1.0, "p5": -1.0, "mean": -1.0, "frac_lt_0p3": 0.0}

    def update_metric(self, *args: Any, **kwargs: Any):
        import torch
        try:
            sensors = self._env.unwrapped.scene.sensors
            if self._sensor_name not in sensors:
                return
            sensor = sensors[self._sensor_name]
            hits = sensor.data.ray_hits_w[0]          # [num_rays, 3] world
            origin = sensor.data.pos_w[0]             # [3] world
            finite = torch.isfinite(hits).all(dim=-1)
            if finite.sum() == 0:
                return
            hits = hits[finite]
            # floor-adaptive vertical band (exclude floor straight-down + ceiling)
            floor_z = float(hits[:, 2].min())
            band = (hits[:, 2] > floor_z + self._FLOOR_MARGIN) & (
                hits[:, 2] < floor_z + self._BODY_TOP)
            if band.sum() == 0:
                return
            horiz = (hits[band][:, :2] - origin[:2].unsqueeze(0)).norm(dim=-1)
            horiz = horiz[horiz > self._MIN_RAY]
            if horiz.numel() == 0:
                return
            c = float(horiz.min())
        except Exception:
            return  # never crash an episode over a metric

        self._step_min.append(c)
        if c < self._NEAR:
            self._n_close += 1
        n = len(self._step_min)
        s = sorted(self._step_min)
        self._metric = {
            "min": s[0],
            "p5": s[max(0, int(0.05 * n) - 1)],
            "mean": sum(self._step_min) / n,
            "frac_lt_0p3": self._n_close / n,
        }


def add_measurement(env, episode, measure_names=["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess", "CollisionRate", "TerminationReason", "ObstacleClearance"]):
    measure_manager = MeasureManager()
    for measure_name in measure_names:
        measure = eval(measure_name)(env, episode, measure_manager)
        measure_manager.register_measure(measure)
    
    return measure_manager