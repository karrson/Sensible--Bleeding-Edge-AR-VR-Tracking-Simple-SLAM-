#include "cv++/cv++.h"
#include "camera.h"
using namespace std;

struct MapNode
{
	cvpp::Mat img;
	cv::Mat pose;
	clock_t time;
};

vector<MapNode> mapNodes;
cv::Mat cameraPoseRelocTarget = cv::Mat::eye(3, 3, CV_64F);
cv::Mat cameraPose = cv::Mat::eye(3, 3, CV_64F);

cv::Mat previousRotation;
cvpp::Mat previousImage;
cv::Mat latestRenderedFrame;

bool isGameSessionActive = true;
bool doInstantReloc = false;
bool isDeltaTimeCalculated = false;
double deltaTime = -1.0;
int currentMapNodeIndex = 0;

vector<cv::Mat> Anchors;

void DrawEntity(cvpp::Mat& renderTexture, const cv::Mat& cameraPose, const cv::Mat& entityPose, float size, float thickness)
{
	vector<cv::Point3d> obj = { cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0), cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0) };
	vector<cv::Point2d> src = { cv::Point2d(0, 0), cv::Point2d(imageSize, 0), cv::Point2d(imageSize, imageSize), cv::Point2d(0, imageSize) };
	vector<cv::Point2d> dst;

	cv::Mat screenPose = (cv::Mat)(cameraPose * entityPose.inv());

	if (std::fabs(cv::Quatd::createFromRotMat(renderTexture.meta.k.inv() * screenPose * renderTexture.meta.k).toEulerAngles(cv::QuatEnum::EXT_XYZ)[0]) < 0.5 ||
		std::fabs(cv::Quatd::createFromRotMat(renderTexture.meta.k.inv() * screenPose * renderTexture.meta.k).toEulerAngles(cv::QuatEnum::EXT_XYZ)[2]) < 0.5)
	{
		cv::perspectiveTransform(src, dst, screenPose);

		cv::Vec3d r, t;
		cv::solvePnP(obj, dst, renderTexture.meta.k, renderTexture.meta.d, r, t);

		cv::drawFrameAxes(renderTexture, renderTexture.meta.k, renderTexture.meta.d, r, t, size, thickness);
	}
}

int main()
{
	for (int i = 0; i < firstFrame; i++)
	{
		cvpp::Mat tempImg;
		cameraUpdate(tempImg);
	}

	while (true)
	{
		clock_t frameStartTime = clock();

		cvpp::Mat currentImg;
		cameraUpdate(currentImg);
		if (previousImage.empty()) previousImage = currentImg.cloneAll();
		cvpp::Mat relCameraPose = cvpp::findHomography(previousImage, currentImg);
		previousImage = currentImg.cloneAll();

		if (!relCameraPose.empty())
		{
			cameraPose = relCameraPose * cameraPose;
			cameraPoseRelocTarget = relCameraPose * cameraPoseRelocTarget;
		}
		else
		{
			isGameSessionActive = false;
		}

		bool addNodeToMap = isGameSessionActive && (mapNodes.empty() || (((double)frameStartTime - (double)mapNodes.back().time) / (double)CLOCKS_PER_SEC) >= relocMaxSeconds && cvpp::findHomography(mapNodes.back().img, currentImg).empty());

		if (addNodeToMap)
		{
			MapNode mapNode;
			mapNode.img = currentImg.cloneAll();
			mapNode.pose = cameraPose.clone();
			mapNode.time = frameStartTime;
			mapNodes.push_back(mapNode);

			if (mapNodes.size() > maxNodesInMap)
			{
				mapNodes.erase(mapNodes.begin());
			}
		}

		bool hasReloc = false;

		if (!mapNodes.empty())
		{
			currentMapNodeIndex %= (uint64_t)mapNodes.size();
			{
				MapNode& currentMapNode = mapNodes[currentMapNodeIndex];
				cv::Mat relRelocPose = cvpp::findHomography(currentMapNode.img, currentImg);

				if (!relRelocPose.empty())
				{
					cameraPoseRelocTarget = relRelocPose * currentMapNode.pose;
					hasReloc = true;
					currentMapNodeIndex = -1;
				}
			}
			currentMapNodeIndex++;
		}

		if (isDeltaTimeCalculated)
		{
			cameraPose += (cameraPoseRelocTarget - cameraPose) * clamp((1.0 / relocMaxSeconds) * deltaTime, 0.0, 1.0);
		}

		doInstantReloc = !isGameSessionActive;

		if (hasReloc)
		{
			isGameSessionActive = true;
		}

		if (doInstantReloc) cameraPose = cameraPoseRelocTarget.clone();

		cvpp::Mat renderTexture = currentImg.cloneAll();

		if (isGameSessionActive)
		{
			for (const auto& pose : Anchors)
			{
				DrawEntity(renderTexture, cameraPose, pose, 0.3f, 15);
			}

			latestRenderedFrame = renderTexture.clone();
			previousRotation = currentImg.meta.r.clone();
		}

		cv::Mat relocGuide = currentImg.meta.k * (currentImg.meta.r * previousRotation.inv()) * currentImg.meta.k.inv();
		cv::warpPerspective(latestRenderedFrame, renderTexture, relocGuide, renderTexture.size());


		cv::imshow("Game", renderTexture);

		char keyPressed = cv::waitKey(1);

		if (isGameSessionActive)
		{
			bool createNewPose = keyPressed == 'c';

			if (createNewPose)
			{
				// Add new pose to frameAxesPoses
				Anchors.push_back(cameraPose.clone());
			}
		}

		clock_t frameEndTime = clock();

		deltaTime = (frameEndTime - frameStartTime) / (double)CLOCKS_PER_SEC;
		isDeltaTimeCalculated = deltaTime > 0;
	}
}