#include <depthai/depthai.hpp>
#include "config.h"
#include <opencv2/core/quaternion.hpp>
using namespace std;

shared_ptr<dai::Pipeline> Pipeline; shared_ptr<dai::Device> Device;
shared_ptr<dai::node::ColorCamera> Camera; shared_ptr<dai::node::XLinkOut> CameraLink;
shared_ptr<dai::node::IMU> IMU; shared_ptr<dai::node::XLinkOut> IMU_Link;
dai::CalibrationHandler Calibration;

void PrepareCameraDriver(cv::Mat& K)
{
    Pipeline = make_shared<dai::Pipeline>();

    Camera = Pipeline->create<dai::node::ColorCamera>(); Camera->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    Camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P); Camera->setFps(20);
    Camera->setPreviewKeepAspectRatio(false);
    CameraLink = Pipeline->create<dai::node::XLinkOut>(); CameraLink->setStreamName("rgb");
    Camera->preview.link(CameraLink->input);

    IMU = Pipeline->create<dai::node::IMU>(); IMU->enableIMUSensor(dai::IMUSensor::ARVR_STABILIZED_GAME_ROTATION_VECTOR, 100);
    IMU->enableIMUSensor(dai::IMUSensor::LINEAR_ACCELERATION, 400);
    IMU->setBatchReportThreshold(1); IMU->setMaxBatchReports(1); IMU_Link = Pipeline->create<dai::node::XLinkOut>();
    IMU_Link->setStreamName("imu"); IMU->out.link(IMU_Link->input);

    Device = make_shared<dai::Device>(*(Pipeline), dai::UsbSpeed::HIGH); Calibration = Device->readCalibration();
    K = cv::Mat::eye(3, 3, CV_64F); auto M = Calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, ImageSize, ImageSize);
    K.at<double>(0, 0) = M[0][0];
    K.at<double>(1, 1) = M[1][1];
    K.at<double>(0, 2) = M[0][2];
    K.at<double>(1, 2) = M[1][2];
}

void CameraUpdate(RobustVisualInertialOdometry::ImageMatrix& Img)
{
    static cv::Mat K; if (K.empty()) PrepareCameraDriver(K);

    Img = (RobustVisualInertialOdometry::ImageMatrix)Device->getOutputQueue("rgb", 3, false)->get<dai::ImgFrame>()->getCvFrame();
    cv::resize(Img, Img, cv::Size(ImageSize, ImageSize));

    auto imu = Device->getOutputQueue("imu", 3, false)->get<dai::IMUData>()->packets.back();
    Img.Metadata.CameraIntrinsicsMatrix = K;
    Img.Metadata.RotationMatrix = (cv::Mat)cv::Quatd(imu.rotationVector.k, -imu.rotationVector.i, -imu.rotationVector.j, imu.rotationVector.real).toRotMat3x3(cv::QuatAssumeType::QUAT_ASSUME_NOT_UNIT);
    Img.Metadata.RotationMatrix.convertTo(Img.Metadata.RotationMatrix, CV_64F);
    Img.Metadata.LinearAcceleration = cv::Vec3d(imu.acceleroMeter.x, imu.acceleroMeter.y, imu.acceleroMeter.z) * 100.0;

    static cv::Mat RotEye = Img.Metadata.RotationMatrix.clone();
    Img.Metadata.RotationMatrix *= RotEye.inv();
}