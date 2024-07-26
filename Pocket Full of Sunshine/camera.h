#include <depthai/depthai.hpp>
#include "cv++/cv++.h"
#include "config.h"
#include <opencv2/core/quaternion.hpp>
using namespace std;

shared_ptr<dai::Pipeline> pipeline; shared_ptr<dai::Device> device;
shared_ptr<dai::node::ColorCamera> camera; shared_ptr<dai::node::XLinkOut> cameraLink;
shared_ptr<dai::node::IMU> imu; shared_ptr<dai::node::XLinkOut> imuLink;
dai::CalibrationHandler calibration;

void cameraSetup(cv::Mat& k) 
{
    pipeline = make_shared<dai::Pipeline>();

    camera = pipeline->create<dai::node::ColorCamera>(); camera->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P); camera->setFps(20);
    camera->setPreviewSize(384, 384); camera->setPreviewKeepAspectRatio(false);
    cameraLink = pipeline->create<dai::node::XLinkOut>(); cameraLink->setStreamName("rgb");
    camera->preview.link(cameraLink->input);

    imu = pipeline->create<dai::node::IMU>(); imu->enableIMUSensor(dai::IMUSensor::ARVR_STABILIZED_ROTATION_VECTOR, 60);
    imu->setBatchReportThreshold(1); imu->setMaxBatchReports(1); imuLink = pipeline->create<dai::node::XLinkOut>();
    imuLink->setStreamName("imu"); imu->out.link(imuLink->input);

    device = make_shared<dai::Device>(*(pipeline), dai::UsbSpeed::HIGH); calibration = device->readCalibration();
    k = cv::Mat::eye(3, 3, CV_64F); auto m = calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, imageSize, imageSize);
    k.at<double>(0, 0) = m[0][0];
    k.at<double>(1, 1) = m[1][1];
    k.at<double>(0, 2) = m[0][2];
    k.at<double>(1, 2) = m[1][2];
}

void cameraUpdate(cvpp::Mat& img) 
{
    static cv::Mat k; if (k.empty()) cameraSetup(k);
    img = (cvpp::Mat)device->getOutputQueue("rgb", 3, false)->get<dai::ImgFrame>()->getCvFrame();
    cv::resize(img, img, cv::Size(imageSize, imageSize));
    auto imu = device->getOutputQueue("imu", 3, false)->get<dai::IMUData>()->packets.back().rotationVector;
    img.meta.k = k;
    img.meta.d = cv::Mat::zeros(1, 5, CV_64F);
    img.meta.r = (cv::Mat)cv::Quatd(imu.k, -imu.i, -imu.j, imu.real).toRotMat3x3(cv::QuatAssumeType::QUAT_ASSUME_NOT_UNIT);
    img.meta.r.convertTo(img.meta.r, CV_64F);
    static cv::Mat firstR = img.meta.r.clone();
    img.meta.r *= firstR.inv();
}