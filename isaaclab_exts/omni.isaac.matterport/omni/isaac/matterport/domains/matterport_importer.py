# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins

# python
import os
from typing import TYPE_CHECKING

# omni
import carb
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils

# isaac-lab
import isaaclab.sim as sim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.matterport.config import MatterportImporterCfg

# omniverse
from isaacsim.core.utils import extensions

extensions.enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter as converter
from pxr import Usd


class MatterportConverter:
    def __init__(self, input_obj: str, context: converter.impl.AssetConverterContext) -> None:
        self._input_obj = input_obj
        self._context = context

        # setup converter
        self.task_manager = converter.extension.AssetImporterExtension()
        return

    async def convert_asset_to_usd(self) -> None:
        # get usd file path and create directory
        base_path, _ = os.path.splitext(self._input_obj)
        # set task
        task = self.task_manager.create_converter_task(
            self._input_obj, base_path + ".usd", asset_converter_context=self._context
        )
        success = await task.wait_until_finished()

        # print error
        if not success:
            detailed_status_code = task.get_status()
            detailed_status_error_string = task.get_error_message()
            carb.log_error(
                f"Failed to convert {self._input_obj} to {base_path + '.usd'} "
                f"with status {detailed_status_code} and error {detailed_status_error_string}"
            )
        return


class MatterportImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: MatterportImporterCfg

    def __init__(self, cfg: MatterportImporterCfg) -> None:
        """
        :param
        """
        # store inputs
        self.cfg = cfg
        self.device = SimulationContext.instance().device

        # buffers (mirroring TerrainImporter.__init__ since we bypass super().__init__)
        self.terrain_prim_paths = list()
        self.env_origins = None
        self.terrain_origins = None
        self._terrain_flat_patches = dict()

        # import the world
        if not self.cfg.terrain_type == "matterport":
            raise ValueError(
                "MatterportImporter can only import 'matterport' data. Given terrain type "
                f"'{self.cfg.terrain_type}'is not supported."
            )

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            self.load_world()
        else:
            carb.log_info("[INFO]: Loading in extension mode requires calling 'load_world_async'")

        if isinstance(self.cfg.num_envs, int):
            self.configure_env_origins()

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Converter
        self.converter: MatterportConverter = MatterportConverter(self.cfg.obj_filepath, self.cfg.asset_converter)
        return

    async def load_world_async(self) -> None:
        """Function called when clicking load button"""
        # create world
        await self.load_matterport()
        # update stage for any remaining process.
        await stage_utils.update_stage_async()
        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")
        return

    def load_world(self) -> None:
        """Function called when clicking load button"""
        # create world
        self.load_matterport_sync()
        # update stage for any remaining process.
        stage_utils.update_stage()
        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")
        return

    async def load_matterport(self) -> None:
        _, ext = os.path.splitext(self.cfg.obj_filepath)
        # if obj mesh --> convert to usd
        if ext == ".obj":
            await self.converter.convert_asset_to_usd()
        # add mesh to stage
        self.load_matterport_sync()

    def load_matterport_sync(self) -> None:
        base_path, _ = os.path.splitext(self.cfg.obj_filepath)
        assert os.path.exists(base_path + ".usd"), (
            "Matterport load sync can only handle '.usd' files not obj files. "
            "Please use the async function to convert the obj file to usd first (accessed over the extension in the GUI)"
        )

        # Multi-env: create one matterport prim per env at the env's spatial
        # origin. With env_spacing large (e.g. 100) the envs are physically far
        # apart in world space so cameras can't see across, and each env's
        # robot can be placed at its episode's start_pos relative to the env's
        # own matterport copy.
        num_envs = int(getattr(self.cfg, "num_envs", 1) or 1)
        # env_origins is filled in by configure_env_origins() during base init.
        # If we run before that, fall back to a single-env layout.
        env_origins = getattr(self, "env_origins", None)
        if env_origins is None or num_envs <= 1:
            self._xform_prim = prim_utils.create_prim(
                prim_path=self.cfg.prim_path + "/Matterport",
                translation=(0.0, 0.0, 0.0),
                usd_path=base_path + ".usd",
            )
            self._env_matterport_prims = [self._xform_prim]
        else:
            self._env_matterport_prims = []
            origins_cpu = env_origins.detach().cpu().numpy().tolist()
            for env_id in range(num_envs):
                ox, oy, oz = origins_cpu[env_id]
                prim_path = f"/World/envs/env_{env_id}/Matterport"
                p = prim_utils.create_prim(
                    prim_path=prim_path,
                    translation=(float(ox), float(oy), float(oz)),
                    usd_path=base_path + ".usd",
                )
                self._env_matterport_prims.append(p)
            self._xform_prim = self._env_matterport_prims[0]

        # No PhysX collider on the matterport mesh: the building USDs have zero
        # pre-authored collision and PhysX cold-cooking the exact triangle mesh
        # is effectively non-terminating on these scenes (>16h, still in cooker).
        # Visual mesh is sufficient for the RayCaster lidar (RTX, no PhysX).
        # Ground is provided by GroundPlane (set groundplane=True in cfg);
        # base_contact termination still fires on falls.

        # create physics material
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = self.cfg.physics_material
        # spawn the material
        physics_material_cfg.func(f"{self.cfg.prim_path}/physicsMaterial", self.cfg.physics_material)
        for _p in self._env_matterport_prims:
            sim_utils.bind_physics_material(_p.GetPrimPath(), f"{self.cfg.prim_path}/physicsMaterial")

        # add colliders and physics material
        if self.cfg.groundplane:
            ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material)
            ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
            ground_plane.visible = False
        return

    def swap_matterport(self, new_obj_filepath: str) -> None:
        """Hot-swap the matterport USD without restarting Isaac Sim.

        Mutates each per-env matterport prim's USD reference to the new file.
        InteractiveScene cloning means each env has its OWN prim instance, so
        we iterate all per-env prim paths and update each's USD reference.

        NOTE: caller must reset the env after this to flush physics state.
        """
        from pxr import Sdf
        import isaacsim.core.utils.stage as _stage_u
        stage = _stage_u.get_current_stage()
        base_path, _ = os.path.splitext(new_obj_filepath)
        new_usd = base_path + ".usd"
        assert os.path.exists(new_usd), f"swap: USD missing {new_usd}"
        self.cfg.obj_filepath = new_obj_filepath
        num_envs = int(getattr(self.cfg, "num_envs", 1) or 1)
        # Stage probe says only /World/matterport/Matterport exists (no per-env
        # clones). Try both layouts in order; the one that's actually present wins.
        candidate_paths = [self.cfg.prim_path + "/Matterport"]
        if num_envs > 1:
            for env_id in range(num_envs):
                candidate_paths.append(f"/World/envs/env_{env_id}/Matterport")
        n_swapped = 0
        for pp in candidate_paths:
            prim = stage.GetPrimAtPath(pp)
            if not prim.IsValid():
                print(f"[matterport.swap] prim missing: {pp}", flush=True)
                continue
            refs = prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference(new_usd)
            n_swapped += 1
        sim = SimulationContext.instance()
        if sim is not None and sim.app is not None:
            for _ in range(3):
                sim.app.update()
        print(f"[matterport.swap] {n_swapped}/{len(candidate_paths)} envs -> {os.path.basename(base_path)}", flush=True)
