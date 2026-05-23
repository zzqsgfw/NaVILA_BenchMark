"""Enable PhysX local mesh cooking cache.

Verified settings (omni.physx bindings, Isaac Sim 5.1):
  /persistent/physics/useLocalMeshCache   (bool)
  /persistent/physics/localMeshCacheSizeMB (int)

With this on, the first time a triangle mesh is cooked it is written to an
on-disk cache keyed by mesh hash; every later load of the SAME mesh (any
later episode / any process) is a cache hit -> no re-cook. There are only
11 unique matterport scenes, so after a one-time pre-cook the exact
triangle-mesh collider loads instantly forever (no interpenetration
explosion, no per-episode cook).

Call enable_local_mesh_cache() AFTER the Kit app is launched
(simulation_app = app_launcher.app) and BEFORE the env / sim is created
(gym.make), since cooking happens at sim start.
"""


def enable_local_mesh_cache(size_mb: int = 8192):
    import os
    import carb
    s = carb.settings.get_settings()
    if os.environ.get("NAVILA_NO_MESHCACHE") == "1":
        s.set_bool("/persistent/physics/useLocalMeshCache", False)
        print("[meshcache] DISABLED (NAVILA_NO_MESHCACHE=1) - pure clean cook", flush=True)
        return
    s.set_bool("/persistent/physics/useLocalMeshCache", True)
    s.set_int("/persistent/physics/localMeshCacheSizeMB", int(size_mb))
    print(f"[meshcache] useLocalMeshCache=True size={size_mb}MB", flush=True)
