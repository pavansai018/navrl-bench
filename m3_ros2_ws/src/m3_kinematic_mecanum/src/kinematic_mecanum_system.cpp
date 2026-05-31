#include <gz/plugin/Register.hh>

#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/JointVelocityCmd.hh>

#include <gz/transport/Node.hh>
#include <gz/msgs/twist.pb.h>

#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>

#include <mutex>
#include <string>
#include <vector>
#include <cmath>

namespace m3_kinematic_mecanum
{

class KinematicMecanumSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(_entity);
    if (_sdf->HasElement("base_link"))
      this->baseLinkName = _sdf->Get<std::string>("base_link");

    auto baseLinkEntity = this->model.LinkByName(_ecm, this->baseLinkName);

    if (baseLinkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[KinematicMecanumSystem] Could not find requested base link: "
            << this->baseLinkName
            << ". Falling back to canonical link.\n";

      baseLinkEntity = this->model.CanonicalLink(_ecm);
    }

    this->baseLink = gz::sim::Link(baseLinkEntity);

    if (!this->baseLink.Valid(_ecm))
    {
      gzerr << "[KinematicMecanumSystem] ERROR: No valid base/canonical link found.\n";
    }
    else
    {
      gzmsg << "[KinematicMecanumSystem] Base link resolved successfully.\n";
    }

    if (!this->model.Valid(_ecm))
    {
      gzerr << "[KinematicMecanumSystem] Invalid model entity.\n";
      return;
    }

    this->topic = "/cmd_vel";

    if (_sdf->HasElement("topic"))
      this->topic = _sdf->Get<std::string>("topic");

    if (_sdf->HasElement("wheel_radius"))
      this->wheelRadius = _sdf->Get<double>("wheel_radius");

    if (_sdf->HasElement("wheel_base_x"))
      this->wheelBaseX = _sdf->Get<double>("wheel_base_x");

    if (_sdf->HasElement("wheel_base_y"))
      this->wheelBaseY = _sdf->Get<double>("wheel_base_y");

    this->frontLeftJointName  = "lwheel1_Joint";
    this->rearLeftJointName   = "lwheel2_Joint";
    this->frontRightJointName = "rwheel1_Joint";
    this->rearRightJointName  = "rwheel2_Joint";

    if (_sdf->HasElement("front_left_joint"))
      this->frontLeftJointName = _sdf->Get<std::string>("front_left_joint");
    if (_sdf->HasElement("rear_left_joint"))
      this->rearLeftJointName = _sdf->Get<std::string>("rear_left_joint");
    if (_sdf->HasElement("front_right_joint"))
      this->frontRightJointName = _sdf->Get<std::string>("front_right_joint");
    if (_sdf->HasElement("rear_right_joint"))
      this->rearRightJointName = _sdf->Get<std::string>("rear_right_joint");

    this->frontLeftJoint = this->model.JointByName(_ecm, this->frontLeftJointName);
    this->rearLeftJoint = this->model.JointByName(_ecm, this->rearLeftJointName);
    this->frontRightJoint = this->model.JointByName(_ecm, this->frontRightJointName);
    this->rearRightJoint = this->model.JointByName(_ecm, this->rearRightJointName);

    if (this->frontLeftJoint == gz::sim::kNullEntity)
      gzerr << "[KinematicMecanumSystem] Missing joint: " << this->frontLeftJointName << "\n";
    if (this->rearLeftJoint == gz::sim::kNullEntity)
      gzerr << "[KinematicMecanumSystem] Missing joint: " << this->rearLeftJointName << "\n";
    if (this->frontRightJoint == gz::sim::kNullEntity)
      gzerr << "[KinematicMecanumSystem] Missing joint: " << this->frontRightJointName << "\n";
    if (this->rearRightJoint == gz::sim::kNullEntity)
      gzerr << "[KinematicMecanumSystem] Missing joint: " << this->rearRightJointName << "\n";

    this->node.Subscribe(this->topic, &KinematicMecanumSystem::OnCmdVel, this);

    gzmsg << "[KinematicMecanumSystem] Loaded.\n";
    gzmsg << "[KinematicMecanumSystem] topic: " << this->topic << "\n";
    gzmsg << "[KinematicMecanumSystem] wheel_radius: " << this->wheelRadius << "\n";
    gzmsg << "[KinematicMecanumSystem] wheel_base_x: " << this->wheelBaseX << "\n";
    gzmsg << "[KinematicMecanumSystem] wheel_base_y: " << this->wheelBaseY << "\n";
  }

  void PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    double vx = 0.0;
    double vy = 0.0;
    double wz = 0.0;

    {
      std::lock_guard<std::mutex> lock(this->mutex);
      vx = this->cmdVx;
      vy = this->cmdVy;
      wz = this->cmdWz;
    }

    // 1. Direct kinematic base velocity.
    // vx, vy are in robot/base frame. Convert to world frame using current yaw.
    const auto pose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);
    const double yaw = pose.Rot().Yaw();

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    const double vxWorld = c * vx - s * vy;
    const double vyWorld = s * vx + c * vy;

    if (this->baseLink.Valid(_ecm))
      {
        this->baseLink.SetLinearVelocity(_ecm, gz::math::Vector3d(vxWorld, vyWorld, 0.0));
        this->baseLink.SetAngularVelocity(_ecm, gz::math::Vector3d(0.0, 0.0, wz));
      }

    // 2. Spin wheels visually using your corrected convention.
    // Current URDF convention:
    // left wheel axes  = +Z
    // right wheel axes = -Z
    //
    // Pure forward: [+, +, +, +]
    // Pure lateral: [-, +, +, -]
    const double r = this->wheelRadius;
    const double l = this->wheelBaseX + this->wheelBaseY;

    if (r <= 0.0)
      return;

    const double wFL = (vx - vy - l * wz) / r;
    const double wRL = (vx + vy - l * wz) / r;
    const double wFR = (vx + vy + l * wz) / r;
    const double wRR = (vx - vy + l * wz) / r;

    this->SetJointVelocityCmd(_ecm, this->frontLeftJoint,  wFL);
    this->SetJointVelocityCmd(_ecm, this->rearLeftJoint,   wRL);
    this->SetJointVelocityCmd(_ecm, this->frontRightJoint, wFR);
    this->SetJointVelocityCmd(_ecm, this->rearRightJoint,  wRR);
  }

private:
  void OnCmdVel(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->cmdVx = _msg.linear().x();
    this->cmdVy = _msg.linear().y();
    this->cmdWz = _msg.angular().z();
  }

  void SetJointVelocityCmd(
    gz::sim::EntityComponentManager &_ecm,
    const gz::sim::Entity &_jointEntity,
    double _velocity)
  {
    if (_jointEntity == gz::sim::kNullEntity)
      return;

    auto comp = _ecm.Component<gz::sim::components::JointVelocityCmd>(_jointEntity);

    if (comp == nullptr)
    {
      _ecm.CreateComponent(
        _jointEntity,
        gz::sim::components::JointVelocityCmd({ _velocity }));
    }
    else
    {
      comp->Data()[0] = _velocity;
    }
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link baseLink{gz::sim::kNullEntity};
  std::string baseLinkName{"base_link"};

  gz::transport::Node node;
  std::mutex mutex;

  std::string topic{"/cmd_vel"};

  std::string frontLeftJointName;
  std::string rearLeftJointName;
  std::string frontRightJointName;
  std::string rearRightJointName;

  gz::sim::Entity frontLeftJoint{gz::sim::kNullEntity};
  gz::sim::Entity rearLeftJoint{gz::sim::kNullEntity};
  gz::sim::Entity frontRightJoint{gz::sim::kNullEntity};
  gz::sim::Entity rearRightJoint{gz::sim::kNullEntity};

  double wheelRadius{0.05};
  double wheelBaseX{0.0795};
  double wheelBaseY{0.09775};

  double cmdVx{0.0};
  double cmdVy{0.0};
  double cmdWz{0.0};
};

}  // namespace m3_kinematic_mecanum

GZ_ADD_PLUGIN(
  m3_kinematic_mecanum::KinematicMecanumSystem,
  gz::sim::System,
  m3_kinematic_mecanum::KinematicMecanumSystem::ISystemConfigure,
  m3_kinematic_mecanum::KinematicMecanumSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  m3_kinematic_mecanum::KinematicMecanumSystem,
  "m3_kinematic_mecanum::KinematicMecanumSystem")
