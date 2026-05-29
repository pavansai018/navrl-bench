from pathlib import Path
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from dynamic_obstacle_avoidance.assets import M3_ASSET_DIR

M3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(M3_ASSET_DIR, 'm3.urdf'),
        fix_base=False,
        copy_from_source=False,
        activate_contact_sensors=True,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            ),
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            'lwheel1_Joint': 0.0,
            'lwheel2_Joint': 0.0,
            'rwheel1_Joint': 0.0,
            'rwheel2_Joint': 0.0,
        },
        
        joint_vel={
            'lwheel1_Joint': 0.0,
            'lwheel2_Joint': 0.0,
            'rwheel1_Joint': 0.0,
            'rwheel2_Joint': 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        'wheel_motors': DelayedPDActuatorCfg(
            joint_names_expr=[
                'lwheel1_Joint',
                'lwheel2_Joint',
                'rwheel1_Joint',
                'rwheel2_Joint',
            ],
            # 0.64 N·m from rated torque.
            #
            # Yahboom rated torque:
            # 6.5 kg·cm.
            #
            # Conversion:
            # 6.5 × 0.0980665 = 0.637 N·m.
            #
            # Use rated torque for continuous RL training,
            # not stall torque.
            effort_limit=0.64,
            # 22.5 rad/s from Yahboom MD520Z56_12V output speed.
            #
            # Motor output speed after reduction:
            # 205 ± 10 RPM.
            #
            # Use upper bound:
            # 215 RPM × 2π / 60 = 22.51 rad/s.
            velocity_limit=22.5,
            # 0.0 because wheel motors are velocity-controlled.
            # A wheel should not behave like a position spring.
            # This is an engineering choice for velocity-driven wheel joints.
            stiffness=0,
            # 0.0284 N·m·s/rad.
            # Calculation:
            # rated torque / max wheel angular velocity
            # = 0.637 N·m / 22.51 rad/s
            # = 0.0283 ≈ 0.0284.
            #
            # Meaning:
            # If wheel velocity error is around max speed, the PD velocity term reaches
            # approximately rated motor torque.
            #
            # This is NOT a Yahboom datasheet damping value.
            # It is a physically bounded velocity-tracking gain derived from motor limits.
            damping=0.0284,
            # 0.0 because joint static friction is not provided in the M3 URDF
            # or Yahboom MD520Z56 public motor table.
            #
            # Do not invent friction for the first physically grounded config.
            # Add measured/static friction later only if you identify wheel breakaway torque.
            friction=0,
            # 0.0 because rotor/gearbox reflected inertia is not provided in the M3 URDF
            # or Yahboom MD520Z56 public motor table.
            #
            # The URDF already contains wheel link inertia, but not motor rotor inertia.
            # Do not invent armature unless you measure or obtain rotor inertia.
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)