#pragma once
#include <opencv2/opencv.hpp>

namespace cvpp {
	struct MatMeta {
		cv::Mat r, k, d;
	};

	struct Mat : public cv::Mat {
		MatMeta meta;

		MatMeta cloneMeta() {
			MatMeta clone;
			clone.r = meta.r.clone();
			clone.k = meta.k.clone();
			clone.d = meta.d.clone();
			return clone;
		}

		cvpp::Mat cloneAll() {
			cvpp::Mat clone = (cvpp::Mat)this->clone();
			clone.meta = cloneMeta();
			return clone;
		}
	};

	static cvpp::Mat findHomography(cvpp::Mat prevImg, cvpp::Mat nextImg, bool force=false) {
		assert(!prevImg.meta.r.empty());
		assert(!nextImg.meta.r.empty());
		assert(!prevImg.meta.k.empty());
		assert(!nextImg.meta.k.empty());
		assert(!prevImg.meta.d.empty());
		assert(!nextImg.meta.d.empty());
		assert(prevImg.meta.r.type() == CV_64F);
		assert(nextImg.meta.r.type() == CV_64F);
		assert(prevImg.meta.k.type() == CV_64F);
		assert(nextImg.meta.k.type() == CV_64F);
		assert(prevImg.meta.d.type() == CV_64F);
		assert(nextImg.meta.d.type() == CV_64F);

		cvpp::Mat temp = prevImg.cloneAll();
		prevImg = nextImg.cloneAll();
		nextImg = temp.cloneAll();

		cv::Mat prevImgCopy = ((cv::Mat)prevImg).clone();
		cv::Mat relR = nextImg.meta.r * prevImg.meta.r.inv();
		relR.convertTo(relR, CV_64F);
		cv::warpPerspective(prevImgCopy, prevImgCopy, prevImg.meta.k * relR * prevImg.meta.k.inv(), prevImg.size());

		auto stitch = cv::Stitcher::create(cv::Stitcher::SCANS);
		bool ok = true;
		try {
			ok = stitch->estimateTransform(std::vector<cv::Mat> {prevImgCopy, nextImg}) == cv::Stitcher::OK &&
				stitch->cameras().size() == 2;
		}
		catch (cv::Exception) {
			ok = false;
		}
		cvpp::Mat H;
		if (ok) {
			H = (cvpp::Mat)(stitch->cameras()[0].R * stitch->cameras()[1].R.inv());
			H.convertTo(H, CV_64F);
			H = (cvpp::Mat)((prevImg.meta.k * relR.inv() * prevImg.meta.k.inv()) * H);
		}
		else if (force) H = (cvpp::Mat)(prevImg.meta.k * relR.inv() * prevImg.meta.k.inv());
		stitch.release();

		H.meta.r = relR;
		H.meta.k = (prevImg.meta.k + nextImg.meta.k) / 2.0;
		H.meta.d = (prevImg.meta.d + nextImg.meta.d) / 2.0;
		return H;
	}
}