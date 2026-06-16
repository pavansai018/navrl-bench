from dynamic_obstacle_avoidance.assets.m3 import M3_CFG
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from .base_env_cfg import LocalPlannerBaseEnvCfg, LocalPlannerSceneCfg


@configclass
class LocalPlannerM3SceneCfg(LocalPlannerSceneCfg):
    robot: ArticulationCfg = M3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


@configclass
class LocalPlannerM3EnvCfg(LocalPlannerBaseEnvCfg):
    debug_draw_nav2: bool = False
    debug_draw_lidar: bool = False
    debug_draw_map: bool = False
    debug_draw_path: bool = False
    debug_draw_dynamic_obstacles: bool = False
    debug_draw_rl_local_path: bool = True

    scene: LocalPlannerM3SceneCfg = LocalPlannerM3SceneCfg(
        num_envs=512,
        env_spacing=22.0,
        replicate_physics=True,
    )