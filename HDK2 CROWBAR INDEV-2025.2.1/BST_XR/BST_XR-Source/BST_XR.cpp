#include <BST_CAM.hpp>
#include <BST_PNP.hpp>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

namespace XRPose {
    typedef struct {
        float Tx, Ty, Tz;
        float Rx, Ry, Rz, Rw;
    } MessageData;

    inline void SendMessage(const MessageData& Message) {
        stringstream PipeOut;
        PipeOut << Message.Tx << " " << Message.Ty << " " << Message.Tz << " "
                << Message.Rx << " " << Message.Ry << " " << Message.Rz << " " << Message.Rw;
        puts(PipeOut.str().c_str());
        cout << PipeOut.str().c_str() << endl;
    }

    inline void SendMessage(const Vec3f& T, const Vec4f& R) {
        MessageData Message;
        Message.Tx = T[0];
        Message.Ty = T[1];
        Message.Tz = T[2];
        Message.Rx = R[0];
        Message.Ry = R[1];
        Message.Rz = R[2];
        Message.Rw = R[3];
        SendMessage(Message);
    }
};

int main() {
    BST_CAM* cameraSystem = new BST_CAM();
    BST_PNP* pnpSystem = new BST_PNP();

    Mat previousLeftImage, previousRightImage, previousRotation;
    clock_t previousClock = clock();

    Vec3f invertedPositionRaw;
    while (true) {
        float deltaTime = -1.0f;

        while (deltaTime <= 0.0f) {
            waitKey(1);
            clock_t currentClock = clock();
            deltaTime = static_cast<float>((static_cast<float>(currentClock) -
                static_cast<float>(previousClock)) /
                static_cast<float>(CLOCKS_PER_SEC));
        }
        previousClock = clock();

        auto inputData = cameraSystem->GetInputData();
        Mat currentLeftImage = inputData.Left.clone();
        Mat currentRightImage = inputData.Right.clone();
        Mat currentRotation = inputData.Rotation.clone();
        currentRotation.convertTo(currentRotation, CV_32F);

        float fx =
            (static_cast<float>(inputData.FocalX) / currentLeftImage.size().width) *
            static_cast<float>(pnpSystem->Resolution);
        float fy = (static_cast<float>(inputData.FocalY) /
            currentLeftImage.size().height) *
            static_cast<float>(pnpSystem->Resolution);

        cv::resize(currentLeftImage, currentLeftImage,
            Size(pnpSystem->Resolution, pnpSystem->Resolution), 0, 0, INTER_AREA);
        cv::resize(currentRightImage, currentRightImage,
            Size(pnpSystem->Resolution, pnpSystem->Resolution), 0, 0, INTER_AREA);

        float baseline = inputData.Baseline;

        if (previousRotation.empty()) {
            previousLeftImage = currentLeftImage.clone();
            previousRightImage = currentRightImage.clone();
            previousRotation = currentRotation.clone();
        }

        Mat cameraMatrix = Mat::eye(3, 3, CV_32F);
        cameraMatrix.at<float>(0, 0) = fx;
        cameraMatrix.at<float>(1, 1) = fy;
        cameraMatrix.at<float>(0, 2) = static_cast<float>(pnpSystem->Resolution) / 2.0f;
        cameraMatrix.at<float>(1, 2) = static_cast<float>(pnpSystem->Resolution) / 2.0f;

        Mat relativeWarpMatrix =
            cameraMatrix *
            (RotationDirection > 0
                ? (currentRotation.inv() * previousRotation).inv()
                : ((currentRotation.inv() * previousRotation).inv()).inv()) *
            cameraMatrix.inv();

        Mat warpedPrevLeft, warpedPrevRight;
        warpPerspective(previousLeftImage, warpedPrevLeft, relativeWarpMatrix,
            previousLeftImage.size());
        warpPerspective(previousRightImage, warpedPrevRight, relativeWarpMatrix,
            previousRightImage.size());

        Mat leftHomography =
            pnpSystem->findHomography(warpedPrevLeft, currentLeftImage);

        if (!leftHomography.empty()) {
            Vec3f relativeRotationVec, tvec;
            if (pnpSystem->solvePnP_Inverted(
                warpedPrevLeft, warpedPrevRight, leftHomography, cameraMatrix,
                baseline, tvec)) {
                if (norm(tvec) > 0.1) tvec = Vec3f();
                tvec = (Vec3f)(Mat)(currentRotation * -tvec);
                invertedPositionRaw += tvec;
            }
        }

        previousLeftImage = currentLeftImage.clone();
        previousRightImage = currentRightImage.clone();
        previousRotation = currentRotation.clone();

        auto Q = Quatf::createFromRotMat(inputData.Rotation.inv());

        if (pnpSystem->Warmup > 0) {
            pnpSystem->Warmup--;
            invertedPositionRaw = Vec3f();
        }
        else if (deltaTime > 0) {
            Vec3f HeadsetTranslation(+invertedPositionRaw[0], -invertedPositionRaw[1],
                +invertedPositionRaw[2]);

            Vec4f HeadsetRotation(-Q.x, +Q.y, -Q.z, -Q.w);

            XRPose::SendMessage(HeadsetTranslation, HeadsetRotation);
        }
    }

    return 0;
}