#include "cv++/cv++.h"
#include "camera.h"
#include "miniGE.h"
using namespace std;

vector<GameObject> scene;
GameObject gameCamera = GameObject::CreatePrimitive(GameObject::PrimitiveType::Camera);

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

int main()
{
	for (int i = 0; i < firstFrame; i++)
	{
		cvpp::Mat tempImg;
		cameraUpdate(tempImg);
	}

	scene.push_back(gameCamera);

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
			gameCamera.transform.pose = cameraPose.clone();
			Graphics::SetRenderTarget(&renderTexture, &gameCamera);

			for (const auto& gameObject : scene)
			{
				Graphics::DrawEntity(gameObject);
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
			bool createNewGameObject = keyPressed == 'c';

			if (createNewGameObject)
			{
				GameObject gameObject = GameObject::CreatePrimitive(GameObject::PrimitiveType::FrameAxesGizmo);
				gameObject.transform.pose = cameraPose.clone();
				gameObject.transform.size = 0.3;
				gameObject.transform.thickness = 15;

				scene.push_back(gameObject);
			}
		}

		clock_t frameEndTime = clock();

		deltaTime = (frameEndTime - frameStartTime) / (double)CLOCKS_PER_SEC;
		isDeltaTimeCalculated = deltaTime > 0;
	}
}