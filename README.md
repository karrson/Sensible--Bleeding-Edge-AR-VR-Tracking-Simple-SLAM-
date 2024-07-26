# Sensible [Bleeding Edge AR / VR Tracking Simple SLAM]
 Sensible SLAM4XR [Bleeding Edge "AR/VR Tracking" "Simple SLAM"]

# My gift to the world
 If I ever die and am reincarnated (lul) I better bet I'm gonna wanna use this to track my VR / AR (XR) HMD (Head Mounted Display)
 
# The Juice
 As we say in America, this is the juice! It'll do what I want. Totally, if not almost totally production quality / ready.
 
# Requires CMake, OpenCV, and OAK-D camera
 I know... CMake isn't perfect... but the only library it uses is OpenCV. It's nice. If you don't have OAK-D, write a driver (easy to modify camera.h) and verify the homography of the rotation (K * R * K.inv())
 
# Rotation stored in Mat metadata
 Allows for better image2image registration. Study homography and come back to this. This is WAYYYYY better than cv::findHomography.
 
# You will need to update it to do 3D registration
 Write that yourself using cv::solvePnP. You'll need that to look around corners.
 
# USE THIS
 I didn't write it for nothing. It's INCREDIBLY robust. Use it. Your HMD depends on it. If it somehow doesn't work, IT'S YOUR FAULT!
 
# A "Pocket Full of Sunshine"
 To quote the song by "Natasha Bedingfield" (just a codename), it's codenamed "Pocket Full of Sunshine" because this REALLY IS SERIOUS SLAM. Keep it at ALL COSTS!