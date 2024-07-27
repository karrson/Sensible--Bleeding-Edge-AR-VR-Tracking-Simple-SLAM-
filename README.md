# Sensible [Bleeding Edge AR / VR Tracking]
 Sensible [Bleeding Edge AR / VR Tracking]

# Requires CMake, OpenCV, and OAK-D camera
 Setup is simple. If you don't have OAK-D, write a driver (easy to modify camera.h) and verify the homography of the rotation (K * R * K.inv())
 
# Rotation stored in Mat metadata
 Allows for better image2image registration. Study homography and come back to this. Tends to be better than cv::findHomography.
 
# You will need to update it to do 3D registration
 Write that yourself using cv::solvePnP. You'll need that to look around corners.

# A serious project
 Highly robust and production ready.
