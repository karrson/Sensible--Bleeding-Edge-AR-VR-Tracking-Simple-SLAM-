#pragma once

#include <depthai/depthai.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

class BST_CAM {
public:
    BST_CAM()
        : Pipeline(), Device(), Calibration(), FocalX(0), FocalY(0), Baseline(0) {
        InitializePipeline();
        InitializeDevice();
    }

    struct InputData {
        float FocalX;
        float FocalY;
        float Baseline;
        cv::Mat Left;
        cv::Mat Right;
        cv::Mat Rotation;
    };

    InputData GetInputData() {
        InputData data;

        data.Left = LeftQueue->get<dai::ImgFrame>()->getCvFrame();
        data.Right = RightQueue->get<dai::ImgFrame>()->getCvFrame();

        data.Baseline = Baseline;
        data.FocalX = FocalX;
        data.FocalY = FocalY;

        auto imuData = IMUQueue->get<dai::IMUData>();
        data.Rotation = ComputeRotation(imuData);

        return data;
    }

private:
    std::shared_ptr<dai::Device> Device;
    dai::Pipeline Pipeline;
    dai::CalibrationHandler Calibration;

    std::shared_ptr<dai::DataOutputQueue> IMUQueue;
    std::shared_ptr<dai::DataOutputQueue> LeftQueue;
    std::shared_ptr<dai::DataOutputQueue> RightQueue;

    float FocalX;
    float FocalY;
    float Baseline;

    cv::Mat FirstRawRotation;

    void InitializeDevice() {
        Device = std::make_shared<dai::Device>(Pipeline, dai::UsbSpeed::SUPER_PLUS);

        Calibration = Device->readCalibration();
        auto intrinsics = Calibration.getCameraIntrinsics(
            dai::CameraBoardSocket::CAM_B, 640, 400);

        FocalX = intrinsics[0][0];
        FocalY = intrinsics[1][1];

        Baseline = Calibration.getBaselineDistance() * 0.01f;

        LeftQueue = Device->getOutputQueue("Left", 1, false);
        RightQueue = Device->getOutputQueue("Right", 1, false);
        IMUQueue = Device->getOutputQueue("IMU", 1, false);
    }

    std::shared_ptr<dai::node::MonoCamera> leftMono;
    std::shared_ptr<dai::node::MonoCamera> rightMono;

    void InitializePipeline() {
        leftMono = Pipeline.create<dai::node::MonoCamera>();
        rightMono = Pipeline.create<dai::node::MonoCamera>();

        auto leftXLinkOut = Pipeline.create<dai::node::XLinkOut>();
        auto rightXLinkOut = Pipeline.create<dai::node::XLinkOut>();

        leftXLinkOut->setStreamName("Left");
        rightXLinkOut->setStreamName("Right");

        ConfigureCamera(leftMono, dai::CameraBoardSocket::CAM_B);
        ConfigureCamera(rightMono, dai::CameraBoardSocket::CAM_C);

        leftMono->out.link(leftXLinkOut->input);
        rightMono->out.link(rightXLinkOut->input);

        auto imuNode = Pipeline.create<dai::node::IMU>();
        auto imuXLinkOut = Pipeline.create<dai::node::XLinkOut>();
        imuXLinkOut->setStreamName("IMU");
        ConfigureIMU(imuNode);
        imuNode->out.link(imuXLinkOut->input);
    }

    void ConfigureCamera(std::shared_ptr<dai::node::MonoCamera> camera,
        dai::CameraBoardSocket socket) {
        camera->setBoardSocket(socket);
        camera->setResolution(
            dai::MonoCameraProperties::SensorResolution::THE_400_P);
        camera->setFps(25);
    }

    void ConfigureIMU(std::shared_ptr<dai::node::IMU> imuNode) {
        imuNode->enableIMUSensor(dai::IMUSensor::GAME_ROTATION_VECTOR, 125);
        imuNode->setMaxBatchReports(1);
        imuNode->setBatchReportThreshold(2);
    }

    cv::Mat ComputeRotation(std::shared_ptr<dai::IMUData> IMU) {
        auto& rvec = IMU->packets.back().rotationVector;
        cv::Mat R = (cv::Mat)(cv::Quatf(rvec.real, -rvec.j, -rvec.k, rvec.i) *
            cv::Quatf::createFromEulerAngles(cv::Vec3f((90.0f / 180.0f) * CV_PI, 0, 0), cv::QuatEnum::EXT_XYZ)).toRotMat3x3();
        R.convertTo(R, CV_32F);
        return R;
    }
};