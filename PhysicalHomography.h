#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

namespace cvpp {
	struct MatMeta { cv::Mat Rot, K; };

	struct Mat : public cv::Mat {
		MatMeta meta;

		MatMeta cloneMeta() {
			MatMeta clone;
			clone.Rot = meta.Rot.clone();
			clone.K = meta.K.clone();
			return clone;
		}

		cvpp::Mat cloneAll() {
			cvpp::Mat clone = (cvpp::Mat)this->clone();
			clone.meta = cloneMeta();
			return clone;
		}
	};

	static cv::Mat findPhysicalHomography(cvpp::Mat prevImg, cvpp::Mat nextImg, bool force = false) { cv::Mat PhysicalHomography_IMU = nextImg.meta.K * (nextImg.meta.Rot * prevImg.meta.Rot.inv()) * nextImg.meta.K.inv(); prevImg = prevImg.cloneAll(); cv::warpPerspective(prevImg, prevImg, PhysicalHomography_IMU.inv(), prevImg.size()); PhysicalHomography_IMU.convertTo(PhysicalHomography_IMU, CV_64F); auto Stitcher = cv::Stitcher::create(cv::Stitcher::SCANS); bool ok = true; try { ok = Stitcher->estimateTransform(std::vector<cv::Mat> {prevImg, nextImg}) == cv::Stitcher::OK && Stitcher->cameras().size() == 2; } catch (cv::Exception) { ok = false; } cv::Mat PhysicalHomography; if (ok) { cv::Mat PhysicalHomography_VO = Stitcher->cameras()[1].R * Stitcher->cameras()[0].R.inv(); PhysicalHomography_VO.convertTo(PhysicalHomography_VO, CV_64F); PhysicalHomography = PhysicalHomography_IMU * PhysicalHomography_VO; } else if (force) PhysicalHomography = PhysicalHomography_IMU; return PhysicalHomography; }
}