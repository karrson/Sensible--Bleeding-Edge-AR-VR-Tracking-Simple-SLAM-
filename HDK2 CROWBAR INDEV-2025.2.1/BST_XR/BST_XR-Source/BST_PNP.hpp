#pragma once
#include <opencv2/opencv.hpp>

int RotationDirection = -1;
float SweetSpot = 35;

using namespace std;
using namespace cv;

class BST_PNP {
public:
    Mat findHomography(Mat prevImg, Mat currImg) {
        vector<Point2f> ptsPrev, ptsCurr;
        vector<DMatch> matches = matchFeatures(prevImg, currImg, ptsPrev, ptsCurr);

        if (ptsPrev.size() < 15 || ptsCurr.size() < 15) return Mat();

        vector<Point2f> inlierPrev, inlierCurr;
        filterByHomography(ptsPrev, ptsCurr, inlierPrev, inlierCurr);
        if (inlierPrev.size() < 15 || inlierCurr.size() < 15) return Mat();

        Mat H = estimateAffinePartial2D(inlierPrev, inlierCurr);
        if (H.empty()) return Mat();
        H.convertTo(H, CV_32F);
        vconcat(H, Vec3f(0, 0, 1).t(), H);
        return H;
    }

    bool solvePnP_Inverted(Mat prevLeftImage, Mat prevRightImage, Mat homography,
        Mat cameraMatrix, float baseline, Vec3f& tvec) {
        try {
            if (homography.rows != 3 || homography.cols != 3) return false;

            auto objectPoints = triangulatePoints(prevLeftImage, prevRightImage,
                cameraMatrix, baseline);
            if (objectPoints.size() < 15) return false;

            Vec3f rvec1, rvec2;
            Vec3f tvec1, tvec2;

            auto imagePoints1 = projectPoints(objectPoints, Mat::eye(3, 3, CV_32F), cameraMatrix);
            if (imagePoints1.size() < 15) return false;

            cv::solvePnP(objectPoints, imagePoints1, cameraMatrix, Mat(), rvec1, tvec1,
                true, SOLVEPNP_ITERATIVE);

            auto imagePoints2 = projectPoints(objectPoints, homography, cameraMatrix);
            if (imagePoints2.size() < 15) return false;

            cv::solvePnP(objectPoints, imagePoints2, cameraMatrix, Mat(), rvec2, tvec2,
                true, SOLVEPNP_ITERATIVE);

            tvec = tvec2 - tvec1;
            return true;
        }
        catch (...) {
            return false;
        }
    }

    float Slack = ((ceilf(SweetSpot / 10.0f) * 10.0f) - SweetSpot) * 10.0f;
    float Resolution = 500.0f * (1.0f - Slack / 100.0f);
    Ptr<ORB> detector = ORB::create(Resolution * 2.0f, 2.0f - SweetSpot / 100.0f);
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    float Warmup = 100.0f;

    vector<Point3f> triangulatePoints(Mat leftImage, const Mat& rightImage,
        Mat cameraMatrix, float baseline) {
        vector<Point3f> point_cloud;

        vector<Point2f> projPointsQuery1, projPointsTrain2;
        vector<DMatch> matches = matchFeatures(leftImage, rightImage,
            projPointsQuery1, projPointsTrain2);
        if (projPointsQuery1.size() < 15) return {};

        vector<Point2f> projPoints1, projPoints2;
        filterByHomography(projPointsQuery1, projPointsTrain2, projPoints1,
            projPoints2);
        if (projPoints1.size() < 15) return {};

        Mat projMatr1 = buildProjectionMatrix(cameraMatrix, +(baseline / 2.0f));
        Mat projMatr2 = buildProjectionMatrix(cameraMatrix, -(baseline / 2.0f));

        Mat points4D;
        cv::triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2,
            points4D);

        convertHomogeneous(points4D, point_cloud);
        return point_cloud;
    }

    vector<Point2f> projectPoints(vector<Point3f> objectPoints, Mat H,
        Mat cameraMatrix) {
        vector<Point2f> imagePoints;
        objectPoints = vector<Point3f>(objectPoints);
        for (auto& pt : objectPoints) pt.z *= 2.0f;
        cv::projectPoints(objectPoints, Vec3f(), Vec3f(), cameraMatrix, noArray(),
            imagePoints);
        if (!H.empty()) perspectiveTransform(imagePoints, imagePoints, H);
        return imagePoints;
    }

    vector<DMatch> matchFeatures(Mat img1, Mat img2, vector<Point2f>& pts1,
        vector<Point2f>& pts2) {
        vector<KeyPoint> kpts1, kpts2;
        Mat desc1, desc2;
        detector->detectAndCompute(img1, noArray(), kpts1, desc1);
        detector->detectAndCompute(img2, noArray(), kpts2, desc2);

        if (kpts1.size() < 15 || kpts2.size() < 15) return {};

        vector<DMatch> matches;
        matcher->match(desc1, desc2, matches);
        if (matches.size() < 15) return {};

        for (const auto& m : matches) {
            pts1.push_back(kpts1[m.queryIdx].pt);
            pts2.push_back(kpts2[m.trainIdx].pt);
        }
        return matches;
    }

    void filterByHomography(vector<Point2f> srcPts, vector<Point2f> dstPts,
        vector<Point2f>& inlierSrc,
        vector<Point2f>& inlierDst) {
        if (srcPts.size() < 15) return;

        Mat mask;
        if (cv::findHomography(srcPts, dstPts, mask, USAC_MAGSAC).empty()) return;

        for (int i = 0; i < (int)mask.total(); i++) {
            if (mask.at<uchar>(i)) {
                inlierSrc.push_back(srcPts[i]);
                inlierDst.push_back(dstPts[i]);
            }
        }
    }

    Mat buildProjectionMatrix(Mat cameraMatrix, float baseline) {
        Mat P = Mat::eye(3, 4, CV_32F);
        P.at<float>(0, 3) = baseline;
        return cameraMatrix * P;
    }

    void convertHomogeneous(Mat points4D, vector<Point3f>& outCloud) {
        if (points4D.total() < 15) return;
        convertPointsFromHomogeneous(points4D.t(), outCloud);
    }
};