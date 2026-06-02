#include <gz/plugin/Register.hh>

#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>

#include <gz/transport/Node.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/pose_v.pb.h>

#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>

#include <mutex>
#include <string>
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

    if (_sdf->HasElement("topic"))
      this->topic = _sdf->Get<std::string>("topic");

    if (_sdf->HasElement("cmd_timeout"))
      this->cmdTimeout = _sdf->Get<double>("cmd_timeout");

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

    if (this->odomPublishFrequency > 0.0)
      this->odomPublishPeriod = 1.0 / this->odomPublishFrequency;

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

    this->baseLink.EnableVelocityChecks(_ecm, true);

    const auto initPose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);
    this->odomOriginPose = initPose;
    this->odomInitialized = true;

    this->odomPub = this->node.Advertise<gz::msgs::Odometry>(this->odomTopic);
    this->tfPub = this->node.Advertise<gz::msgs::Pose_V>(this->tfTopic);

    this->node.Subscribe(this->topic, &MecanumPoseDriveSystem::OnCmdVel, this);

    gzmsg << "[MecanumPoseDriveSystem] Loaded successfully.\n";
    gzmsg << "  base_link: " << this->baseLinkName << "\n";
    gzmsg << "  topic: " << this->topic << "\n";
    gzmsg << "  odom_topic: " << this->odomTopic << "\n";
    gzmsg << "  tf_topic: " << this->tfTopic << "\n";
    gzmsg << "  frame_id: " << this->odomFrameId << "\n";
    gzmsg << "  child_frame_id: " << this->childFrameId << "\n";
    gzmsg << "  cmd_timeout: " << this->cmdTimeout << "\n";
  }

  void PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    const auto simTime = _info.simTime;

    double dt = 0.0;
    if (this->lastSimTime != std::chrono::steady_clock::duration::zero())
      dt = std::chrono::duration<double>(simTime - this->lastSimTime).count();

    this->lastSimTime = simTime;

    if (dt <= 0.0 || dt > 0.2)
      return;

    if (!this->baseLink.Valid(_ecm))
    {
      auto baseLinkEntity = this->model.LinkByName(_ecm, this->baseLinkName);
      if (baseLinkEntity == gz::sim::kNullEntity)
        baseLinkEntity = this->model.CanonicalLink(_ecm);

      this->baseLink = gz::sim::Link(baseLinkEntity);

      if (this->baseLink.Valid(_ecm))
        this->baseLink.EnableVelocityChecks(_ecm, true);
    }

    if (!this->baseLink.Valid(_ecm))
    {
      gzerr << "[MecanumPoseDriveSystem] No valid base link in PreUpdate.\n";
      return;
    }

    double vx = 0.0;
    double vy = 0.0;
    double wz = 0.0;

    {
      std::lock_guard<std::mutex> lock(this->mutex);

      if (this->cmdReceived)
      {
        this->lastCmdTime = simTime;
        this->cmdReceived = false;
      }

      const double sinceCmd =
        std::chrono::duration<double>(simTime - this->lastCmdTime).count();

      if (sinceCmd <= this->cmdTimeout)
      {
        vx = this->cmdVx;
        vy = this->cmdVy;
        wz = this->cmdWz;
      }
    }

    const auto currentPose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);

    if (!this->odomInitialized)
    {
      this->odomOriginPose = currentPose;
      this->odomInitialized = true;
    }

    // IMPORTANT FIX:
    // Do not rotate vx,vy by yaw here.
    // In your setup, doing so causes the post-rotation swap behavior.
    double vzWorld = 0.0;
    auto currentLinearVel = this->baseLink.WorldLinearVelocity(_ecm);
    if (currentLinearVel.has_value())
      vzWorld = currentLinearVel.value().Z();

    this->baseLink.SetLinearVelocity(
      _ecm,
      gz::math::Vector3d(vx, vy, vzWorld));

    double wx = 0.0;
    double wy = 0.0;
    auto currentAngVel = this->baseLink.WorldAngularVelocity(_ecm);
    if (currentAngVel.has_value())
    {
      wx = currentAngVel.value().X();
      wy = currentAngVel.value().Y();
    }

    this->baseLink.SetAngularVelocity(
      _ecm,
      gz::math::Vector3d(wx, wy, wz));

    this->odomTimeSinceLastPublish += dt;

    if (this->odomPublishFrequency > 0.0 &&
        this->odomTimeSinceLastPublish >= this->odomPublishPeriod)
    {
      const auto odomPose = gz::sim::worldPose(this->baseLink.Entity(), _ecm);

      this->PublishOdometry(_info, odomPose, vx, vy, wz);
      this->PublishTF(_info, odomPose);

      this->odomTimeSinceLastPublish = 0.0;
    }

    this->debugCounter++;
    if (this->debugCounter % 50 == 0)
    {
      gzmsg << "[MecanumPoseDriveSystem ACTIVE] "
            << "cmd=(" << vx << "," << vy << "," << wz << ") "
            << "pose=("
            << currentPose.Pos().X() << ","
            << currentPose.Pos().Y() << ","
            << currentPose.Pos().Z() << ")\n";
    }
  }

private:
  void OnCmdVel(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);

    this->cmdVx = _msg.linear().x();
    this->cmdVy = _msg.linear().y();
    this->cmdWz = _msg.angular().z();
    this->cmdReceived = true;
  }

  void PublishOdometry(
    const gz::sim::UpdateInfo &_info,
    const gz::math::Pose3d &_currentPose,
    double _vx,
    double _vy,
    double _wz)
  {
    const gz::math::Pose3d relativePose =
      this->odomOriginPose.Inverse() * _currentPose;

    gz::msgs::Odometry odomMsg;

    const auto simTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _info.simTime).count();

    odomMsg.mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000LL));
    odomMsg.mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000LL));

    auto *frameData = odomMsg.mutable_header()->add_data();
    frameData->set_key("frame_id");
    frameData->add_value(this->odomFrameId);

    auto *childFrameData = odomMsg.mutable_header()->add_data();
    childFrameData->set_key("child_frame_id");
    childFrameData->add_value(this->childFrameId);

    odomMsg.mutable_pose()->mutable_position()->set_x(relativePose.Pos().X());
    odomMsg.mutable_pose()->mutable_position()->set_y(relativePose.Pos().Y());
    odomMsg.mutable_pose()->mutable_position()->set_z(relativePose.Pos().Z());

    odomMsg.mutable_pose()->mutable_orientation()->set_x(relativePose.Rot().X());
    odomMsg.mutable_pose()->mutable_orientation()->set_y(relativePose.Rot().Y());
    odomMsg.mutable_pose()->mutable_orientation()->set_z(relativePose.Rot().Z());
    odomMsg.mutable_pose()->mutable_orientation()->set_w(relativePose.Rot().W());

    odomMsg.mutable_twist()->mutable_linear()->set_x(_vx);
    odomMsg.mutable_twist()->mutable_linear()->set_y(_vy);
    odomMsg.mutable_twist()->mutable_linear()->set_z(0.0);

    odomMsg.mutable_twist()->mutable_angular()->set_x(0.0);
    odomMsg.mutable_twist()->mutable_angular()->set_y(0.0);
    odomMsg.mutable_twist()->mutable_angular()->set_z(_wz);

    this->odomPub.Publish(odomMsg);
  }

  void PublishTF(
    const gz::sim::UpdateInfo &_info,
    const gz::math::Pose3d &_currentPose)
  {
    const gz::math::Pose3d relativePose =
      this->odomOriginPose.Inverse() * _currentPose;

    gz::msgs::Pose_V tfMsg;

    const auto simTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _info.simTime).count();

    tfMsg.mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000LL));
    tfMsg.mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000LL));

    auto *poseMsg = tfMsg.add_pose();

    poseMsg->mutable_header()->mutable_stamp()->set_sec(
      static_cast<int64_t>(simTimeNs / 1000000000LL));
    poseMsg->mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(simTimeNs % 1000000000LL));

    auto *frameData = poseMsg->mutable_header()->add_data();
    frameData->set_key("frame_id");
    frameData->add_value(this->odomFrameId);

    auto *childFrameData = poseMsg->mutable_header()->add_data();
    childFrameData->set_key("child_frame_id");
    childFrameData->add_value(this->childFrameId);

    poseMsg->set_name(this->childFrameId);

    poseMsg->mutable_position()->set_x(relativePose.Pos().X());
    poseMsg->mutable_position()->set_y(relativePose.Pos().Y());
    poseMsg->mutable_position()->set_z(relativePose.Pos().Z());

    poseMsg->mutable_orientation()->set_x(relativePose.Rot().X());
    poseMsg->mutable_orientation()->set_y(relativePose.Rot().Y());
    poseMsg->mutable_orientation()->set_z(relativePose.Rot().Z());
    poseMsg->mutable_orientation()->set_w(relativePose.Rot().W());

    this->tfPub.Publish(tfMsg);
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link baseLink{gz::sim::kNullEntity};
  std::string baseLinkName{"base_link"};

  gz::transport::Node node;
  std::mutex mutex;

  std::string topic{"/cmd_vel"};

  double cmdVx{0.0};
  double cmdVy{0.0};
  double cmdWz{0.0};

  double cmdTimeout{0.5};
  bool cmdReceived{false};

  std::chrono::steady_clock::duration lastSimTime{
    std::chrono::steady_clock::duration::zero()};
  std::chrono::steady_clock::duration lastCmdTime{
    std::chrono::steady_clock::duration::zero()};

  gz::math::Pose3d odomOriginPose;
  bool odomInitialized{false};

  gz::transport::Node::Publisher odomPub;
  gz::transport::Node::Publisher tfPub;
  std::string odomTopic{"/odom_raw"};
  std::string tfTopic{"/tf"};
  std::string odomFrameId{"odom"};
  std::string childFrameId{"base_footprint"};
  double odomPublishFrequency{50.0};
  double odomPublishPeriod{0.02};
  double odomTimeSinceLastPublish{0.0};

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