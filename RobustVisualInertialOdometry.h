/*
This code defines structures and functions for image processing using OpenCV:

1. ImageMatrixMetadata: Struct containing rotation matrix, linear acceleration, and camera intrinsics matrix.
2. ImageMatrix: Custom matrix type inheriting from cv::Mat, including metadata and methods to clone with metadata.
3. from_images_to_translation_to_projection: Function that takes two ImageMatrix objects and computes a transformation
matrix:
   - Applies rotation to align images
   - Attempts to stitch images using OpenCV's Stitcher
   - Returns a projection matrix representing the translation between images
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

#include <atomic>

namespace RobustVisualInertialOdometry {
    // 1. ImageMatrixMetadata: Struct containing rotation matrix, linear acceleration, and camera intrinsics matrix.
    struct ImageMatrixMetadata {
        cv::Mat RotationMatrix;
        cv::Vec3d LinearAcceleration;
        cv::Mat CameraIntrinsicsMatrix;
    };

    // 2. ImageMatrix: Custom matrix type inheriting from cv::Mat, including metadata and methods to clone with metadata.
    struct ImageMatrix : public cv::Mat {
        ImageMatrixMetadata Metadata;

        ImageMatrixMetadata clone_metadata()
        {
            ImageMatrixMetadata CloneOfMetadata;
            CloneOfMetadata.RotationMatrix = Metadata.RotationMatrix.clone();
            CloneOfMetadata.LinearAcceleration = Metadata.LinearAcceleration;
            CloneOfMetadata.CameraIntrinsicsMatrix = Metadata.CameraIntrinsicsMatrix.clone();

            return CloneOfMetadata;
        }

        ImageMatrix clone_with_metadata()
        {
            ImageMatrix CloneOfMatrixContainingCloneOfMetadata = (ImageMatrix)this->clone();
            CloneOfMatrixContainingCloneOfMetadata.Metadata = clone_metadata();

            return CloneOfMatrixContainingCloneOfMetadata;
        }
    };

    /*
    3. from_images_to_translation_to_projection: Function that takes two ImageMatrix objects and computes a transformation matrix
    :
        - Applies rotation to align images
        - Attempts to stitch images using OpenCV's Stitcher
        - Returns a projection matrix representing the translation between images
    */
    static cv::Mat from_images_to_translation_to_projection(ImageMatrix ImageA, ImageMatrix ImageB)
    {
        cv::Mat FromRotationToProjection = ImageB.Metadata.CameraIntrinsicsMatrix
            * (ImageB.Metadata.RotationMatrix * ImageA.Metadata.RotationMatrix.inv())
            * ImageB.Metadata.CameraIntrinsicsMatrix.inv();

        cv::Mat RotatedImageA;
        cv::warpPerspective(ImageA, RotatedImageA, FromRotationToProjection, ImageA.size());

        cv::Ptr<cv::Stitcher> ImageStitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
        bool IsTransformEstimated = true;
        try {
            IsTransformEstimated
                = ImageStitcher->estimateTransform(std::vector<cv::Mat> { RotatedImageA, ImageB }) == cv::Stitcher::OK
                && ImageStitcher->cameras().size() == 2;
        }
        catch (cv::Exception) {
            IsTransformEstimated = false;
        }

        cv::Mat FromTranslationToProjection;
        if (IsTransformEstimated) {
            FromTranslationToProjection = ImageStitcher->cameras()[1].R * ImageStitcher->cameras()[0].R.inv();
            FromTranslationToProjection.convertTo(FromTranslationToProjection, CV_64F);
        }

        return FromTranslationToProjection;
    }

    static cv::Vec3d from_images_to_local_direction(ImageMatrix ImageA, ImageMatrix ImageB)
    {
        cv::Mat ProjectiveTranslation = RobustVisualInertialOdometry::from_images_to_translation_to_projection(ImageA, ImageB);
        if (!ProjectiveTranslation.empty()) {
            std::vector<cv::Point3d> ObjPoints = { cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0), cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0) };
            std::vector<cv::Point2d> SrcPoints = { cv::Point2d(0, 0), cv::Point2d(ImageA.cols, 0), cv::Point2d(ImageA.cols, ImageA.rows), cv::Point2d(0, ImageA.rows) };
            std::vector<cv::Point2d> DstPoints; cv::perspectiveTransform(SrcPoints, DstPoints, ProjectiveTranslation);

            cv::Vec3d RawTranslation;
            {
                cv::Mat DiscardedRotation;
                cv::solvePnP(ObjPoints, DstPoints, ImageA.Metadata.CameraIntrinsicsMatrix, cv::Mat(), DiscardedRotation, RawTranslation, false, cv::SOLVEPNP_AP3P);
                static cv::Vec3d RawTranslationOrigin = RawTranslation;
                RawTranslation -= RawTranslationOrigin;
                if (cv::norm(RawTranslation) > 0) RawTranslation /= cv::norm(RawTranslation);
            }

            return RawTranslation;
        }
        else return cv::Vec3d(0, 0, 0);
    }

    static cv::Vec3d from_images_to_global_direction(ImageMatrix ImageA, ImageMatrix ImageB)
    {
        return (cv::Vec3d)(cv::Mat)(ImageB.Metadata.RotationMatrix.inv() * -from_images_to_local_direction(ImageA, ImageB));
    }

    enum TimeStepMode {
        PerFrame,
        PerSecond
    };

    static cv::Vec3d from_images_to_global_translation(ImageMatrix ImageA, ImageMatrix ImageB, 
        cv::Vec3d InitialGuess, cv::Vec3d InitialGlobalTranslation, 
        double DeltaTime, TimeStepMode TimeStepMode, double AccelerationWeight)
    {
        cv::Vec3d GlobalDirection = from_images_to_global_direction(ImageA, ImageB);
        std::vector<cv::Vec3d> GlobalTranslations;

        for (int n = 1; n <= 4; n++) {
            double TranslationalMagnitudeInitialGuess = cv::norm(InitialGuess);
            cv::Vec3d Acceleration = GlobalDirection * cv::norm(ImageB.Metadata.LinearAcceleration);

            if (TimeStepMode == TimeStepMode::PerFrame) {
                TranslationalMagnitudeInitialGuess *= DeltaTime;
                Acceleration *= DeltaTime;
            }

            double TranslationalMagnitudeNoisy = cv::norm(InitialGlobalTranslation + Acceleration);

            double TranslationalMagnitude = TranslationalMagnitudeInitialGuess * (1.0 - AccelerationWeight) +
                TranslationalMagnitudeNoisy * AccelerationWeight;

            InitialGlobalTranslation = GlobalDirection * TranslationalMagnitude;
            GlobalTranslations.push_back(InitialGlobalTranslation);
        }

        InitialGlobalTranslation = (GlobalTranslations[0] + GlobalTranslations[1] * 2 + GlobalTranslations[2] * 2 + GlobalTranslations[3]) / 6.0;
        return InitialGlobalTranslation;
    }
}