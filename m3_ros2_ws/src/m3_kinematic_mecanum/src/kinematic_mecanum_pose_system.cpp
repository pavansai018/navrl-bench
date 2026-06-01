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
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/time.pb.h>

#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>

#include <mutex>
#include <string>
#include <cmath>
#include <chrono>

namespace m3_mecanum_pose_drive
{

class MecanumPoseDriveSystem :
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

    if (!this->model.Valid(_ecm))
    {
      gzerr << "[MecanumPoseDriveSystem] Invalid model entity.\n";
      return;
    }

    if (_sdf->HasElement("base_link"))
      this->baseLinkName = _sdf->Get<std::string>("base_link");

    auto baseLinkEntity = this->model.LinkByName(_ecm, this->baseLinkName);

    if (baseLinkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[MecanumPoseDriveSystem] Could not find base link: "
            << this->baseLinkName
            << ". Falling back to canonical link.\n";
      baseLinkEntity = this->model.CanonicalLink(_ecm);
    }

    this->baseLink = gz::sim::Link(baseLinkEntity);

    if (!this->baseLink.Valid(_ecm))
    {
      gzerr << "[MecanumPoseDriveSystem] ERROR: No valid base/canonical link.\n";
      return;
    }

    gzmsg << "[MecanumPoseDriveSystem] Base link resolved: "
          << this->baseLinkName << "\n";

    this->baseLink.EnableVelocityChecks(_ecm, true);

    if (_sdf->HasElement("topic"))
      this->topic = _sdf->Get<std::string>("topic");
    if (_sdf->HasElement("wheel_radius"))
      this->wheelRadius = _sdf->Get<double>("wheel_radius");
    if (_sdf->HasElement("wheel_base_x"))
      this->wheelBaseX = _sdf->Get<double>("wheel_base_x");
    if (_sdf->HasElement("wheel_base_y"))
      this->wheelBaseY = _sdf->Get<double>("wheel_base_y");
    if (_sdf->HasElement("cmd_timeout"))
      this->cmdTimeout = _sdf->Get<double>("cmd_timeout");
    if (_sdf->HasElement("yaw_correction_gain"))
      this->yawCorrectionGain = _sdf->Get<double>("yaw_correction_gain");

    // NEW: Read odometry configuration from SDF
    if (_sdf->HasElement("odom_topic"))
      this->odomTopic = _sdf->Get<std::string>("odom_topic");
    if (_sdf->HasElement("tf_topic"))
      this->tfTopic = _sdf->Get<std::string>("tf_topic");
    if (_sdf->HasElement("frame_id"))
      this->odomFrameId = _sdf->Get<std::string>("frame_id");
    if (_sdf->HasElement("child_frame_id"))
      this->childFrameId = _sdf->Get<std::string>("child_frame_id");
    if (_sdf->HasElement("odom_publish_frequency"))
      this->odomPublishFrequency = _sdf->Get<double>("odom_publish_frequency");

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
      gzerr << "[MecanumPoseDriveSystem] Missing joint: " << this->frontLeftJointName << "\n";
    if (this->rearLeftJoint == gz::sim::kNullEntity)
      gzerr << "[MecanumPoseDriveSystem] Missing joint: " << this->rearLeftJointName << "\n";
    if (this->frontRightJoint == gz::sim::kNullEntity)
      gzerr << "[MecanumPoseDriveSystem] Missing joint: " << this->frontRightJointName << "\n";
    if (this->rearRightJoint == gz::sim::kNullEntity)
      gzerr << "[MecanumPoseDriveSystem] Missing joint: " << this->rearRightJointName << "\n";

    // Initialize desired yaw from current pose
    const auto initPose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);
    this->desiredYaw = initPose.Rot().Yaw();
    this->yawInitialized = true;

    // NEW: Store initial pose as odometry origin
    this->odomOriginPose = initPose;

    // NEW: Create publishers for odometry and TF
    this->odomPub = this->node.Advertise<gz::msgs::Odometry>(this->odomTopic);
    this->tfPub = this->node.Advertise<gz::msgs::Pose_V>(this->tfTopic);

    if (this->odomPublishFrequency > 0.0)
      this->odomPublishPeriod = 1.0 / this->odomPublishFrequency;

    this->node.Subscribe(this->topic, &MecanumPoseDriveSystem::OnCmdVel, this);

    gzmsg << "[MecanumPoseDriveSystem] Loaded successfully.\n";
    gzmsg << "[MecanumPoseDriveSystem] topic: " << this->topic << "\n";
    gzmsg << "[MecanumPoseDriveSystem] odom_topic: " << this->odomTopic << "\n";
    gzmsg << "[MecanumPoseDriveSystem] tf_topic: " << this->tfTopic << "\n";
    gzmsg << "[MecanumPoseDriveSystem] frame_id: " << this->odomFrameId << "\n";
    gzmsg << "[MecanumPoseDriveSystem] child_frame_id: " << this->childFrameId << "\n";
    gzmsg << "[MecanumPoseDriveSystem] odom_publish_frequency: " << this->odomPublishFrequency << " Hz\n";
    gzmsg << "[MecanumPoseDriveSystem] wheel_radius: " << this->wheelRadius << "\n";
    gzmsg << "[MecanumPoseDriveSystem] wheel_base_x: " << this->wheelBaseX << "\n";
    gzmsg << "[MecanumPoseDriveSystem] wheel_base_y: " << this->wheelBaseY << "\n";
    gzmsg << "[MecanumPoseDriveSystem] cmd_timeout: " << this->cmdTimeout << "s\n";
    gzmsg << "[MecanumPoseDriveSystem] yaw_correction_gain: " << this->yawCorrectionGain << "\n";
  }

  void PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    // Calculate dt
    auto dtDuration = _info.simTime - this->lastSimTime;
    double dt = std::chrono::duration<double>(dtDuration).count();
    this->lastSimTime = _info.simTime;

    if (dt <= 0.0 || dt > 1.0)
      return;

    double vx = 0.0;
    double vy = 0.0;
    double wz = 0.0;

    {
      std::lock_guard<std::mutex> lock(this->mutex);

      if (this->cmdReceived)
      {
        this->lastCmdTime = _info.simTime;
        this->cmdReceived = false;
      }

      auto timeSinceLastCmd = _info.simTime - this->lastCmdTime;
      double timeSinceLastCmdSec = std::chrono::duration<double>(timeSinceLastCmd).count();

      if (timeSinceLastCmdSec < this->cmdTimeout)
      {
        vx = this->cmdVx;
        vy = this->cmdVy;
        wz = this->cmdWz;
      }
    }

    if (!this->baseLink.Valid(_ecm))
      return;

    // Get current pose
    const auto currentPose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);
    const double currentYaw = currentPose.Rot().Yaw();

    // Initialize desired yaw on first run
    if (!this->yawInitialized)
    {
      this->desiredYaw = currentYaw;
      this->yawInitialized = true;
      this->odomOriginPose = currentPose;
    }

    // Update desired yaw based on command
    this->desiredYaw += wz * dt;

    // Normalize desired yaw to [-pi, pi]
    while (this->desiredYaw > M_PI) this->desiredYaw -= 2.0 * M_PI;
    while (this->desiredYaw < -M_PI) this->desiredYaw += 2.0 * M_PI;

    // Calculate yaw error
    double yawError = this->desiredYaw - currentYaw;
    while (yawError > M_PI) yawError -= 2.0 * M_PI;
    while (yawError < -M_PI) yawError += 2.0 * M_PI;

    // Use DESIRED yaw for velocity transform
    const double c = std::cos(this->desiredYaw);
    const double s = std::sin(this->desiredYaw);

    const double vxWorld = c * vx - s * vy;
    const double vyWorld = s * vx + c * vy;

    // Preserve Z velocity from physics (gravity)
    auto currentVelOpt = this->baseLink.WorldLinearVelocity(_ecm);
    double currentVz = 0.0;
    if (currentVelOpt.has_value())
    {
      currentVz = currentVelOpt.value().Z();
    }

    this->baseLink.SetLinearVelocity(
      _ecm,
      gz::math::Vector3d(vxWorld, vyWorld, currentVz));

    // Apply yaw with P-controller correction
    double correctedWz = wz + this->yawCorrectionGain * yawError;

    // Preserve roll/pitch angular velocities
    auto currentAngVelOpt = this->baseLink.WorldAngularVelocity(_ecm);
    double currentWx = 0.0;
    double currentWy = 0.0;
    if (currentAngVelOpt.has_value())
    {
      currentWx = currentAngVelOpt.value().X();
      currentWy = currentAngVelOpt.value().Y();
    }

    this->baseLink.SetAngularVelocity(
      _ecm,
      gz::math::Vector3d(currentWx, currentWy, correctedWz));

    // NEW: Publish odometry and TF at configured frequency
    this->odomTimeSinceLastPublish += dt;
    if (this->odomTimeSinceLastPublish >= this->odomPublishPeriod)
    {
      this->odomTimeSinceLastPublish = 0.0;
      this->PublishOdometry(_info, _ecm, currentPose, vx, vy, wz);
      this->PublishTF(_info, currentPose);
    }

    // Spin wheels visually
    const double r = this->wheelRadius;
    const double l = this->wheelBaseX + this->wheelBaseY;

    if (r > 0.0)
    {
      const double wFL = (vx - vy - l * wz) / r;
      const double wRL = (vx + vy - l * wz) / r;
      const double wFR = (vx + vy + l * wz) / r;
      const double wRR = (vx - vy + l * wz) / r;

      this->SetJointVelocityCmd(_ecm, this->frontLeftJoint,  wFL);
      this->SetJointVelocityCmd(_ecm, this->rearLeftJoint,   wRL);
      this->SetJointVelocityCmd(_ecm, this->frontRightJoint, wFR);
      this->SetJointVelocityCmd(_ecm, this->rearRightJoint,  wRR);
    }

    // DEBUG: print every 100 steps
    this->debugCounter++;
    if (this->debugCounter % 100 == 0)
    {
      gzmsg << "[DEBUG] pos=("
            << currentPose.Pos().X() << ", "
            << currentPose.Pos().Y() << ", "
            << currentPose.Pos().Z() << ") yaw="
            << currentYaw << " desiredYaw="
            << this->desiredYaw << " yawErr="
            << yawError << " cmd=("
            << vx << ", " << vy << ", " << wz << ")\n";
    }
  }

private:
  // NEW: Publish odometry message
  void PublishOdometry(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm,
    const gz::math::Pose3d &_currentPose,
    double _vx, double _vy, double _wz)
  {
    // Compute pose relative to odom origin
    gz::math::Pose3d relativePose =
      this->odomOriginPose.Inverse() * _currentPose;

    gz::msgs::Odometry odomMsg;

    // Set timestamp
    auto simTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _info.simTime).count();
    odomMsg.mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000));
    odomMsg.mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000));

    // Set frame IDs in header
    auto *frameData = odomMsg.mutable_header()->add_data();
    frameData->set_key("frame_id");
    frameData->add_value(this->odomFrameId);

    auto *childFrameData = odomMsg.mutable_header()->add_data();
    childFrameData->set_key("child_frame_id");
    childFrameData->add_value(this->childFrameId);

    // Set pose (position + orientation relative to odom origin)
    odomMsg.mutable_pose()->mutable_position()->set_x(relativePose.Pos().X());
    odomMsg.mutable_pose()->mutable_position()->set_y(relativePose.Pos().Y());
    odomMsg.mutable_pose()->mutable_position()->set_z(relativePose.Pos().Z());
    odomMsg.mutable_pose()->mutable_orientation()->set_x(relativePose.Rot().X());
    odomMsg.mutable_pose()->mutable_orientation()->set_y(relativePose.Rot().Y());
    odomMsg.mutable_pose()->mutable_orientation()->set_z(relativePose.Rot().Z());
    odomMsg.mutable_pose()->mutable_orientation()->set_w(relativePose.Rot().W());

    // Set twist (body-frame velocities)
    odomMsg.mutable_twist()->mutable_linear()->set_x(_vx);
    odomMsg.mutable_twist()->mutable_linear()->set_y(_vy);
    odomMsg.mutable_twist()->mutable_linear()->set_z(0.0);
    odomMsg.mutable_twist()->mutable_angular()->set_x(0.0);
    odomMsg.mutable_twist()->mutable_angular()->set_y(0.0);
    odomMsg.mutable_twist()->mutable_angular()->set_z(_wz);

    this->odomPub.Publish(odomMsg);
  }

  // NEW: Publish TF (odom → base_footprint)
  void PublishTF(
    const gz::sim::UpdateInfo &_info,
    const gz::math::Pose3d &_currentPose)
  {
    // Compute pose relative to odom origin
    gz::math::Pose3d relativePose =
      this->odomOriginPose.Inverse() * _currentPose;

    gz::msgs::Pose_V tfMsg;

    // Set timestamp on the Pose_V message header
    auto simTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _info.simTime).count();
    tfMsg.mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000));
    tfMsg.mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000));

    // Create the transform pose
    auto *poseMsg = tfMsg.add_pose();

    // Set frame names
    // In gz::msgs::Pose_V used for TF:
    //   header frame_id data = parent frame
    //   pose name = child frame
    auto *frameData = poseMsg->mutable_header()->add_data();
    frameData->set_key("frame_id");
    frameData->add_value(this->odomFrameId);

    auto *childFrameData = poseMsg->mutable_header()->add_data();
    childFrameData->set_key("child_frame_id");
    childFrameData->add_value(this->childFrameId);

    // Set timestamp on individual pose header too
    poseMsg->mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000));
    poseMsg->mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000));

    // Set the name (used as child_frame_id by ros_gz_bridge)
    poseMsg->set_name(this->childFrameId);

    // Set transform
    poseMsg->mutable_position()->set_x(relativePose.Pos().X());
    poseMsg->mutable_position()->set_y(relativePose.Pos().Y());
    poseMsg->mutable_position()->set_z(relativePose.Pos().Z());
    poseMsg->mutable_orientation()->set_x(relativePose.Rot().X());
    poseMsg->mutable_orientation()->set_y(relativePose.Rot().Y());
    poseMsg->mutable_orientation()->set_z(relativePose.Rot().Z());
    poseMsg->mutable_orientation()->set_w(relativePose.Rot().W());

    this->tfPub.Publish(tfMsg);
  }

  void OnCmdVel(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->cmdVx = _msg.linear().x();
    this->cmdVy = _msg.linear().y();
    this->cmdWz = _msg.angular().z();
    this->cmdReceived = true;
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
        gz::sim::components::JointVelocityCmd({_velocity}));
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

  std::string frontLeftJointName{"lwheel1_Joint"};
  std::string rearLeftJointName{"lwheel2_Joint"};
  std::string frontRightJointName{"rwheel1_Joint"};
  std::string rearRightJointName{"rwheel2_Joint"};

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

  double cmdTimeout{0.5};
  bool cmdReceived{false};
  std::chrono::steady_clock::duration lastSimTime{0};
  std::chrono::steady_clock::duration lastCmdTime{0};

  // Yaw correction
  double desiredYaw{0.0};
  bool yawInitialized{false};
  double yawCorrectionGain{50.0};

  // NEW: Odometry and TF publishing
  gz::transport::Node::Publisher odomPub;
  gz::transport::Node::Publisher tfPub;
  std::string odomTopic{"/odom_raw"};
  std::string tfTopic{"/tf"};
  std::string odomFrameId{"odom"};
  std::string childFrameId{"base_footprint"};
  double odomPublishFrequency{50.0};
  double odomPublishPeriod{0.02};
  double odomTimeSinceLastPublish{0.0};
  gz::math::Pose3d odomOriginPose;

  // Debug
  int debugCounter{0};
};

}  // namespace m3_mecanum_pose_drive

GZ_ADD_PLUGIN(
  m3_mecanum_pose_drive::MecanumPoseDriveSystem,
  gz::sim::System,
  m3_mecanum_pose_drive::MecanumPoseDriveSystem::ISystemConfigure,
  m3_mecanum_pose_drive::MecanumPoseDriveSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  m3_mecanum_pose_drive::MecanumPoseDriveSystem,
  "m3_mecanum_pose_drive::MecanumPoseDriveSystem")