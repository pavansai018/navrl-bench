from dynamic_obstacle_avoidance.assets.m3 import M3_CFG
from .base_env_cfg import DynamicObstacleAvoidanceEnvCfg, DynamicObstacleAvoidanceSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg


@configclass
class DynamicObstacleAvoidanceM3SceneCfg(DynamicObstacleAvoidanceSceneCfg):
    """
    ROSMASTER M3 Scene config.
    """
    robot: ArticulationCfg = M3_CFG.replace(
        prim_path='{ENV_REGEX_NS}/Robot'
    )

@configclass
class DynamicObstacleAvoidanceM3EnvCfg(DynamicObstacleAvoidanceEnvCfg):
        """ROSMASTER M3 dynamic obstacle avoidance environment."""
        debug_draw_nav2: bool = False
        debug_draw_lidar: bool = False
        debug_draw_map: bool = False
        debug_draw_path: bool = False
        scene: DynamicObstacleAvoidanceM3SceneCfg = DynamicObstacleAvoidanceM3SceneCfg(
            num_envs=512,
            env_spacing=22.0,
            replicate_physics=True,
        )