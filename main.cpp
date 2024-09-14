#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <depthai/depthai.hpp>

namespace BST_SLAM_2D {
    double CfgLoad = 100;
    double CfgReso = 180;
    double CfgBuff = 260;
    double CfgTime = 0.5;
    double CfgDist = 5;
    double CfgConf = 0.96;

    struct FrameMeta {
        cv::Mat R, K;
    };

    struct Frame : public cv::Mat {
        FrameMeta Meta;

        FrameMeta clone_meta() {
            FrameMeta Clone;
            Clone.R = Meta.R.clone();
            Clone.K = Meta.K.clone();
            return Clone;
        }

        Frame clone_all() {
            Frame Clone = (Frame)this->clone();
            Clone.Meta = clone_meta();
            return Clone;
        }
    };

    std::shared_ptr<dai::Pipeline> Pipeline;
    std::shared_ptr<dai::Device> Device;
    std::shared_ptr<dai::node::ColorCamera> Camera;
    std::shared_ptr<dai::node::XLinkOut> LinkCamera;
    std::shared_ptr<dai::node::IMU> IMU;
    std::shared_ptr<dai::node::XLinkOut> LinkIMU;
    dai::CalibrationHandler Calibration;

    void camera_setup(cv::Mat& K)
    {
        Pipeline = std::make_shared<dai::Pipeline>();

        Camera = Pipeline->create<dai::node::ColorCamera>(); Camera->setBoardSocket(dai::CameraBoardSocket::CAM_A);
        Camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P); Camera->setFps(20);
        Camera->setPreviewSize(384, 384); Camera->setPreviewKeepAspectRatio(false);
        LinkCamera = Pipeline->create<dai::node::XLinkOut>(); LinkCamera->setStreamName("rgb");
        Camera->preview.link(LinkCamera->input);

        IMU = Pipeline->create<dai::node::IMU>(); IMU->enableIMUSensor(dai::IMUSensor::ARVR_STABILIZED_ROTATION_VECTOR, 60);
        IMU->setBatchReportThreshold(1); IMU->setMaxBatchReports(1); LinkIMU = Pipeline->create<dai::node::XLinkOut>();
        LinkIMU->setStreamName("imu"); IMU->out.link(LinkIMU->input);

        Device = std::make_shared<dai::Device>(*(Pipeline), dai::UsbSpeed::HIGH); Calibration = Device->readCalibration();
        K = cv::Mat::eye(3, 3, CV_64F); auto M = Calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, CfgReso, CfgReso);
        K.at<double>(0, 0) = M[0][0];
        K.at<double>(1, 1) = M[1][1];
        K.at<double>(0, 2) = M[0][2];
        K.at<double>(1, 2) = M[1][2];
    }

    void camera_update(BST_SLAM_2D::Frame& Image)
    {
        static cv::Mat K; if (K.empty()) camera_setup(K);
        Image = (BST_SLAM_2D::Frame)Device->getOutputQueue("rgb", 3, false)->get<dai::ImgFrame>()->getCvFrame();
        cv::resize(Image, Image, cv::Size(CfgReso, CfgReso));
        auto IMU = Device->getOutputQueue("imu", 3, false)->get<dai::IMUData>()->packets.back().rotationVector;
        Image.Meta.K = K;
        Image.Meta.R = (cv::Mat)cv::Quatd(IMU.k, -IMU.i, -IMU.j, IMU.real).toRotMat3x3(cv::QuatAssumeType::QUAT_ASSUME_NOT_UNIT);
        Image.Meta.R.convertTo(Image.Meta.R, CV_64F);
        static cv::Mat Ro = Image.Meta.R.clone();
        Image.Meta.R *= Ro.inv();
    }

    static Frame find_homography(Frame PrevImg, Frame NextImg, bool Force = false) {
        Frame Temp = PrevImg.clone_all();
        PrevImg = NextImg.clone_all();
        NextImg = Temp.clone_all();

        cv::Mat PrevImgCopy = ((cv::Mat)PrevImg).clone();
        cv::Mat RelativeRotation = NextImg.Meta.R * PrevImg.Meta.R.inv();
        RelativeRotation.convertTo(RelativeRotation, CV_64F);
        cv::warpPerspective(PrevImgCopy, PrevImgCopy, PrevImg.Meta.K * RelativeRotation * PrevImg.Meta.K.inv(), PrevImg.size());

        cv::Ptr<cv::Stitcher> Stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
        Stitcher->setPanoConfidenceThresh(CfgConf);
        bool Success = true;
        try {
            Success = Stitcher->estimateTransform(std::vector<cv::Mat> {PrevImgCopy, NextImg}) == cv::Stitcher::OK &&
                Stitcher->cameras().size() == 2;
        }
        catch (cv::Exception) {
            Success = false;
        }

        Frame H;
        if (Success) {
            H = (Frame)(Stitcher->cameras()[0].R * Stitcher->cameras()[1].R.inv());
            H.convertTo(H, CV_64F);
            H = (Frame)((PrevImg.Meta.K * RelativeRotation.inv() * PrevImg.Meta.K.inv()) * H);
        }
        else if (Force) {
            H = (Frame)(PrevImg.Meta.K * RelativeRotation.inv() * PrevImg.Meta.K.inv());
        }

        H.Meta.R = RelativeRotation;
        H.Meta.K = (PrevImg.Meta.K + NextImg.Meta.K) / 2.0;
        return H;
    }

    struct MapNode {
        Frame Image;
        cv::Mat Pose;
        clock_t Time;
    };

    void draw_pose(Frame& RenderTexture, cv::Mat& CameraPose, cv::Mat& Pose, float Size, float Thickness) {
        std::vector<cv::Point3d> Obj = { cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0), cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0) };
        std::vector<cv::Point2d> Src = { cv::Point2d(0, 0), cv::Point2d(CfgReso, 0), cv::Point2d(CfgReso, CfgReso), cv::Point2d(0, CfgReso) };
        std::vector<cv::Point2d> Dst;

        cv::Mat ScreenPose = (cv::Mat)(CameraPose * Pose.inv());

        if (std::fabs(cv::Quatd::createFromRotMat(RenderTexture.Meta.K.inv() * ScreenPose * RenderTexture.Meta.K).toEulerAngles(cv::QuatEnum::EXT_XYZ)[0]) < 0.5 ||
            std::fabs(cv::Quatd::createFromRotMat(RenderTexture.Meta.K.inv() * ScreenPose * RenderTexture.Meta.K).toEulerAngles(cv::QuatEnum::EXT_XYZ)[2]) < 0.5) {
            cv::perspectiveTransform(Src, Dst, ScreenPose);

            cv::Vec3d R, T;
            cv::solvePnP(Obj, Dst, RenderTexture.Meta.K, cv::Mat(), R, T);

            cv::drawFrameAxes(RenderTexture, RenderTexture.Meta.K, cv::Mat(), R, T, Size, Thickness);
        }
    }
}

int main() {
    for (int i = 0; i < BST_SLAM_2D::CfgLoad; i++) {
        BST_SLAM_2D::Frame TempImg;
        camera_update(TempImg);
    }

    BST_SLAM_2D::Frame Curr;
    BST_SLAM_2D::Frame Prev;

    std::vector<BST_SLAM_2D::MapNode> MapNodes;
    cv::Mat RelocTarget = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat PrevR;
    cv::Mat LatestRenderedFrame;

    cv::Mat CameraPose = cv::Mat::eye(3, 3, CV_64F);

    bool IsTrackerSessionActive = true;
    bool DoInstantReloc = false;
    bool IsDeltaTimeCalculated = false;
    double DeltaTime = -1.0;
    int CurrentMapNodeIndex = 0;

    std::vector<cv::Mat> Anchors;

    while (true) {
        clock_t FrameStartTime = clock();

        camera_update(Curr);
        if (Prev.empty()) Prev = Curr.clone_all();
        cv::Mat RelCameraPose = find_homography(Prev, Curr);
        Prev = Curr.clone_all();

        if (!RelCameraPose.empty()) {
            CameraPose = RelCameraPose * CameraPose;
            RelocTarget = RelCameraPose * RelocTarget;
        }
        else IsTrackerSessionActive = false;

        bool AddNodeToMap = IsTrackerSessionActive && (MapNodes.empty() ||
            (((double)FrameStartTime - (double)MapNodes.back().Time) / (double)CLOCKS_PER_SEC) >= BST_SLAM_2D::CfgTime &&
            find_homography(MapNodes.back().Image, Curr).empty());

        if (AddNodeToMap) {
            BST_SLAM_2D::MapNode MapNode;
            MapNode.Image = Curr.clone_all();
            MapNode.Pose = CameraPose.clone();
            MapNode.Time = FrameStartTime;
            MapNodes.push_back(MapNode);

            if (MapNodes.size() > BST_SLAM_2D::CfgBuff) {
                MapNodes.erase(MapNodes.begin());
            }
        }

        bool HasReloc = false;

        if (!MapNodes.empty()) {
            CurrentMapNodeIndex %= (uint64_t)MapNodes.size();
            {
                BST_SLAM_2D::MapNode& CurrentMapNode = MapNodes[CurrentMapNodeIndex];
                cv::Mat RelRelocPose = find_homography(CurrentMapNode.Image, Curr);

                if (!RelRelocPose.empty()) {
                    RelocTarget = RelRelocPose * CurrentMapNode.Pose;
                    HasReloc = true;
                    CurrentMapNodeIndex = -1;
                }
            }
            CurrentMapNodeIndex++;
        }

        if (IsDeltaTimeCalculated) CameraPose += (RelocTarget - CameraPose) * std::clamp((DeltaTime / BST_SLAM_2D::CfgTime) *
            (cv::norm(RelocTarget - CameraPose) / BST_SLAM_2D::CfgDist), 0.0, 1.0);

        DoInstantReloc = !IsTrackerSessionActive;

        if (HasReloc) IsTrackerSessionActive = true;

        if (DoInstantReloc) CameraPose = RelocTarget.clone();

        BST_SLAM_2D::Frame RenderTexture = Curr.clone_all();

        if (IsTrackerSessionActive) {
            for (auto& Pose : Anchors) draw_pose(RenderTexture, CameraPose, Pose, 0.3f, 15);

            LatestRenderedFrame = RenderTexture.clone();
            PrevR = Curr.Meta.R.clone();
        }

        cv::Mat RelocGuide = Curr.Meta.K * (Curr.Meta.R * PrevR.inv()) * Curr.Meta.K.inv();
        cv::warpPerspective(LatestRenderedFrame, RenderTexture, RelocGuide, RenderTexture.size());

        cv::imshow("Tracker", RenderTexture);

        if (cv::waitKey(1) == 'c' && IsTrackerSessionActive) Anchors.push_back(CameraPose.clone());

        clock_t FrameEndTime = clock();

        DeltaTime = (FrameEndTime - FrameStartTime) / (double)CLOCKS_PER_SEC;
        IsDeltaTimeCalculated = DeltaTime > 0;
    }
}
