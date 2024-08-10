#include "RobustVisualInertialOdometry.h"
#include "camera.h"

int main() {
	cv::Vec3d rvec;
	cv::Vec3d tvec(0, 0, 20.24);

	double DeltaTime = 0;
	cv::Vec3d Velocity;
	RobustVisualInertialOdometry::ImageMatrix PreviousImage;
	for (int i = 0; i < FirstFrame; i++) {
		RobustVisualInertialOdometry::ImageMatrix CurrentImage;
		CameraUpdate(CurrentImage);
	}
	while (true) {
		clock_t FrameStartTime = clock();
		RobustVisualInertialOdometry::ImageMatrix CurrentImage;
		CameraUpdate(CurrentImage);
		if (PreviousImage.empty()) PreviousImage = CurrentImage.clone_with_metadata();
		Velocity = RobustVisualInertialOdometry::from_images_to_global_translation(PreviousImage, CurrentImage,
			cv::Vec3d(0, 0, VIO_Beta), Velocity, 
			DeltaTime, RobustVisualInertialOdometry::PerSecond, VIO_Alpha);
		tvec += Velocity * DeltaTime;
		cv::Rodrigues(CurrentImage.Metadata.RotationMatrix.inv(), rvec);
		PreviousImage = CurrentImage.clone_with_metadata();

		cv::Vec3d Rot1 = cv::Quatd::createFromRotMat(CurrentImage.Metadata.RotationMatrix).inv().toRotVec();
		cv::Quatd Rot2 = cv::Quatd::createFromRvec(cv::Vec3d(Rot1[0] * -1, Rot1[1], Rot1[2] * -1));

		cv::Vec3f CameraPosition(tvec[0], tvec[1] * -1, tvec[2]);
		cv::Vec4f CameraRotation(Rot2.x, Rot2.y, Rot2.z, Rot2.w);

		std::cout << CameraPosition << CameraRotation << std::endl;

		cv::Mat PoseVis = cv::Mat::zeros(CurrentImage.size(), CV_8UC(3));
		cv::drawFrameAxes(PoseVis, CurrentImage.Metadata.CameraIntrinsicsMatrix, cv::Mat(), rvec, tvec, 15.24);
		cv::imshow("Pose", PoseVis);
		cv::imshow("Image", CurrentImage);
		static cv::Mat Map = cv::Mat::zeros(720, 720, CV_8UC3);
		try { cv::circle(Map, cv::Point(360 + tvec[0] * 0.5, 360 - tvec[2] * 0.5), 3, cv::Scalar(0, 0, 255), -1); } catch (std::exception) { ; }
		cv::imshow("Map", Map);
		cv::waitKey(1);
		clock_t FrameEndTime = clock();
		double DT = ((double)FrameEndTime - (double)FrameStartTime) / CLOCKS_PER_SEC;
		if (DT > 0 && DeltaTime <= 0) DeltaTime = DT;
	}

	return 0;
}