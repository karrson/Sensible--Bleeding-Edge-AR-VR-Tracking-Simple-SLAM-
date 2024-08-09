#include "PhysicalHomography.h"
#include "Camera.h"
using namespace std;

cv::Mat ComposeRT(cv::Vec3d rvec, cv::Vec3d tvec, bool Is4x4) {
	cv::Mat M;
	cv::Rodrigues(rvec, M);
	cv::hconcat(M, tvec, M);
	if (Is4x4) {
		cv::vconcat(M, ((cv::Mat)cv::Vec4d(0, 0, 0, 1)).t(), M);
	}
	return M;
}

void DrawFrameAxes(cv::Mat& Image, cv::Mat CameraMatrix, cv::Mat Pose, const double& length) {
	std::vector<cv::Point3d> AxisPoints;
	AxisPoints.push_back(cv::Point3d(0, 0, 0));
	AxisPoints.push_back(cv::Point3d(length, 0, 0));
	AxisPoints.push_back(cv::Point3d(0, length, 0));
	AxisPoints.push_back(cv::Point3d(0, 0, length));

	std::vector<cv::Point2d> ImagePoints;
	CameraMatrix = CameraMatrix.clone();
	CameraMatrix.convertTo(CameraMatrix, CV_64F);
	Pose = Pose.clone();
	Pose.convertTo(Pose, CV_64F);
	for (const auto& P3D : AxisPoints) {
		cv::Mat _pt1 = (cv::Mat)P3D;
		cv::Mat _pt2 = Pose(cv::Range(0, 3), cv::Range(0, 4)) * cv::Vec4d(_pt1.at<double>(0), _pt1.at<double>(1), _pt1.at<double>(2), 1);
		_pt2.at<double>(0) /= _pt2.at<double>(2);
		_pt2.at<double>(1) /= _pt2.at<double>(2);
		_pt2.at<double>(0) = _pt2.at<double>(0) * CameraMatrix.at<double>(0, 0) + CameraMatrix.at<double>(0, 2);
		_pt2.at<double>(1) = _pt2.at<double>(1) * CameraMatrix.at<double>(1, 1) + CameraMatrix.at<double>(1, 2);
		ImagePoints.push_back(cv::Point2d(_pt2.at<double>(0), _pt2.at<double>(1)));
	}

	line(Image, ImagePoints[0], ImagePoints[1], cv::Scalar(0, 0, 255), 2);
	line(Image, ImagePoints[0], ImagePoints[2], cv::Scalar(0, 255, 0), 2);
	line(Image, ImagePoints[0], ImagePoints[3], cv::Scalar(255, 0, 0), 2);
}

int main() {
	cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
	Pose.at<double>(2, 3) = 30.48;

	double DT = 0;
	cvpp::Mat Prev;

	for (int i = 0; i < FirstFrame; i++) { cvpp::Mat Curr; CameraUpdate(Curr); }

	cv::Mat _tvec;
	while (true) {
		clock_t Start = clock();

		cvpp::Mat Curr; CameraUpdate(Curr);
		cvpp::MatMeta CurrMeta = Curr.cloneMeta();
		cv::resize(Curr, Curr, cv::Size(ImageSize, ImageSize));
		cv::resize(Curr, Curr, cv::Size(ImageSize, ImageSize));
		Curr.meta = CurrMeta;

		if (Prev.empty()) Prev = Curr.cloneAll();

		cv::Mat PhysicalHomography = cvpp::findPhysicalHomography(Prev, Curr, true);
		if (!PhysicalHomography.empty()) {
			vector<cv::Point3d> Obj = { cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0), cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0) };
			vector<cv::Point2d> Src = { cv::Point2d(0, 0), cv::Point2d(ImageSize, 0), cv::Point2d(ImageSize, ImageSize), cv::Point2d(0, ImageSize) };
			vector<cv::Point2d> Dst; cv::perspectiveTransform(Src, Dst, PhysicalHomography);

			cv::Vec3d tvec, _;
			cv::solvePnP(Obj, Dst, Curr.meta.K, cv::Mat(), _, tvec, false, cv::SOLVEPNP_AP3P);
			tvec /= (1.0 / 3.0);
			if (_tvec.empty()) _tvec = (cv::Mat)tvec;
			tvec -= (cv::Vec3d)_tvec;
			if (std::isnan(tvec[0])) tvec[0] = 0;
			if (std::isnan(tvec[1])) tvec[1] = 0;
			if (std::isnan(tvec[2])) tvec[2] = 0;

			tvec *= 30.48;

			cv::Mat rvec;
			cv::Rodrigues(Curr.meta.Rot * Prev.meta.Rot.inv(), rvec);

			cv::Mat Rel = ComposeRT(rvec, tvec, true).inv();
			((cv::Mat)((Curr.meta.Rot * Prev.meta.Rot.inv()).inv())).copyTo(Rel.rowRange(0, 3).colRange(0, 3));
			Pose *= Rel;

			std::cout << Pose.at<double>(0, 3) << std::endl;
		}

		Prev = Curr.cloneAll();

		cv::Mat Vis = cv::Mat::zeros(Curr.size(), CV_8UC(3));
		DrawFrameAxes(Vis, Prev.meta.K, Pose, 15.24);
		imshow("Vis", Vis);
		cv::waitKey(1);

		DT = (clock() - Start) / (double)CLOCKS_PER_SEC;
	}

	return 0;
}