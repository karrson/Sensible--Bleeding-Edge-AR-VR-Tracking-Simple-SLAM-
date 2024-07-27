#pragma once
#include "cv++/cv++.h"
#include <opencv2/core/quaternion.hpp>
#include "config.h"
using namespace std;

class Transform 
{
public:
	cv::Matx33d pose;
	float size;
	float thickness;
};

class GameObject 
{
public:
	Transform transform;

	enum PrimitiveType 
	{
		Camera,
		FrameAxesGizmo
	};

private:
	PrimitiveType primitiveType = PrimitiveType::Camera;
	bool isPrimitive = false;

public:
	static GameObject CreatePrimitive(PrimitiveType primitiveType) 
	{
		GameObject gameObject;
		gameObject.primitiveType = primitiveType;
		gameObject.isPrimitive = true;
		
		return gameObject;
	}

	bool IsPrimitive() 
	{
		return isPrimitive;
	}

	bool IsPrimitive(PrimitiveType& primitiveType) 
	{	
		if (IsPrimitive())
		{	
			primitiveType = this->primitiveType;
		}

		return IsPrimitive();
	}
};

namespace detail::Graphics
{
	static cvpp::Mat* renderTexture = nullptr;
	static GameObject* camera = nullptr;
}

class Graphics 
{
public:
	static void SetRenderTarget(cvpp::Mat* renderTexture, GameObject* camera) 
	{
		GameObject::PrimitiveType primitiveType;
		
		if (camera->IsPrimitive(primitiveType) && primitiveType == GameObject::PrimitiveType::Camera)
		{
			detail::Graphics::renderTexture = renderTexture;
			detail::Graphics::camera = camera;
		}
	}

	static void DrawEntity(GameObject gameObject)
	{
		vector<cv::Point3d> obj = { cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0), cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0) };
		vector<cv::Point2d> src = { cv::Point2d(0, 0), cv::Point2d(imageSize, 0), cv::Point2d(imageSize, imageSize), cv::Point2d(0, imageSize) };
		vector<cv::Point2d> dst;

		cv::Mat screenPose = (cv::Mat)(detail::Graphics::camera->transform.pose * gameObject.transform.pose.inv());

		if (std::fabs(cv::Quatd::createFromRotMat(detail::Graphics::renderTexture->meta.k.inv() * screenPose * detail::Graphics::renderTexture->meta.k).toEulerAngles(cv::QuatEnum::EXT_XYZ)[0]) < 0.5 ||
			std::fabs(cv::Quatd::createFromRotMat(detail::Graphics::renderTexture->meta.k.inv() * screenPose * detail::Graphics::renderTexture->meta.k).toEulerAngles(cv::QuatEnum::EXT_XYZ)[2]) < 0.5)
		{
			cv::perspectiveTransform(src, dst, screenPose);

			cv::Vec3d r, t;
			cv::solvePnP(obj, dst, detail::Graphics::renderTexture->meta.k, detail::Graphics::renderTexture->meta.d, r, t);

			GameObject::PrimitiveType primitiveType;

			if (gameObject.IsPrimitive(primitiveType) && primitiveType == GameObject::PrimitiveType::FrameAxesGizmo)
			{
				cv::drawFrameAxes(*detail::Graphics::renderTexture, detail::Graphics::renderTexture->meta.k, detail::Graphics::renderTexture->meta.d, r, t, gameObject.transform.size, gameObject.transform.thickness);
			}
		}
	}
};