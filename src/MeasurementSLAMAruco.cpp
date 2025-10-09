#include "MeasurementSLAMAruco.h"
#include "SystemSLAM.h"
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <Eigen/Eigenvalues>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
//#include <autodiff/forward/dual2nd.hpp>

// Static member initialization - corner positions in marker local frame
// From assignment Equation 9: corners in order [top-left, top-right, bottom-right, bottom-left]
const std::array<Eigen::Vector3d, 4> MeasurementSLAMAruco::CORNER_POSITIONS_LOCAL = {{
    Eigen::Vector3d(-MeasurementSLAMAruco::MARKER_SIZE/2,  MeasurementSLAMAruco::MARKER_SIZE/2, 0),  // Corner 1: top-left
    Eigen::Vector3d( MeasurementSLAMAruco::MARKER_SIZE/2,  MeasurementSLAMAruco::MARKER_SIZE/2, 0),  // Corner 2: top-right
    Eigen::Vector3d( MeasurementSLAMAruco::MARKER_SIZE/2, -MeasurementSLAMAruco::MARKER_SIZE/2, 0),  // Corner 3: bottom-right
    Eigen::Vector3d(-MeasurementSLAMAruco::MARKER_SIZE/2, -MeasurementSLAMAruco::MARKER_SIZE/2, 0)   // Corner 4: bottom-left
}};

// Static storage for landmark tag IDs
std::vector<int> MeasurementSLAMAruco::landmarkTagIds_;

MeasurementSLAMAruco::MeasurementSLAMAruco(double time, 
                                           const Camera & camera,
                                           const std::vector<int> & tagIds,
                                           const std::vector<std::vector<cv::Point2f>> & corners)
    : MeasurementSLAM(time, camera)
    , tagIds_(tagIds)
    , corners_(corners)
    , sigma_(10.0)  // 5 pixel measurement noise - more confident ArUco detection
{
    assert(tagIds_.size() == corners_.size());
    
    // Verify each tag has exactly 4 corners
    for(const auto & cornerSet : corners_) {
        (void)cornerSet;  // Suppress unused variable warning
        assert(cornerSet.size() == 4);
    }
}

MeasurementSLAM * MeasurementSLAMAruco::clone() const
{
    return new MeasurementSLAMAruco(*this);
}

MeasurementSLAMAruco::~MeasurementSLAMAruco()
{
    // Default destructor
}

GaussianInfo<double> MeasurementSLAMAruco::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 8; // 8D measurement (4 corners × 2 coordinates)

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::Matrix<double, 8, 1> ya = predictArucoCorners(x, J, system, idxLandmark) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

GaussianInfo<double> MeasurementSLAMAruco::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 8 * idxLandmarks.size(); // 8D per landmark (4 corners × 2 coordinates)

    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        
        // Predict corners for all landmarks
        Eigen::VectorXd ya(ny);
        Eigen::MatrixXd J(ny, nx);
        
        for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
            // Predict corners for this landmark
            Eigen::MatrixXd Jlandmark;
            Eigen::Matrix<double, 8, 1> cornerPrediction = predictArucoCorners(x, Jlandmark, system, idxLandmarks[i]);
            
            // Store prediction in combined vector
            ya.segment<8>(8*i) = cornerPrediction;
            
            // Store Jacobian in combined matrix
            J.block(8*i, 0, 8, nx) = Jlandmark;
        }
        
        // Add noise: y = h(x) + v
        ya += v;
        
        // Set up combined Jacobian [dha/dx, dha/dv]
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    // Create noise covariance for all landmarks (independent noise per corner)
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_ * Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density * pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

// const std::vector<int> & MeasurementSLAMAruco::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
// {
//     idxFeatures_.clear();
//     idxFeatures_.resize(idxLandmarks.size(), -1);  // Initialize with "no association"
    
//     std::cout << "ArUco Data Association:" << std::endl;
//     std::cout << "  Detected " << tagIds_.size() << " tags, tracking " << idxLandmarks.size() << " landmarks" << std::endl;
    
//     // For each landmark in the map, try to find a corresponding detected tag
//     for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
//         std::size_t landmarkIdx = idxLandmarks[i];
        
//         // Check if we have a tag ID stored for this landmark
//         if (landmarkIdx < landmarkTagIds_.size()) {
//             int expectedTagId = landmarkTagIds_[landmarkIdx];
            
//             // Search for this tag ID in detected tags
//             for (std::size_t j = 0; j < tagIds_.size(); ++j) {
//                 if (tagIds_[j] == expectedTagId) {
//                     idxFeatures_[i] = static_cast<int>(j);  // Associate landmark i with detection j
//                     std::cout << "  Associated landmark " << landmarkIdx << " (tag " << expectedTagId << ") with detection " << j << std::endl;
//                     break;
//                 }
//             }
            
//             if (idxFeatures_[i] == -1) {
//                 std::cout << "  Landmark " << landmarkIdx << " (tag " << expectedTagId << ") not detected this frame" << std::endl;
//             }
//         }
//     }
    
//     // Report any unassociated detections (these would need new landmarks)
//     std::vector<bool> detectionUsed(tagIds_.size(), false);
//     for (int featureIdx : idxFeatures_) {
//         if (featureIdx >= 0) {
//             detectionUsed[featureIdx] = true;
//         }
//     }
    
//     std::cout << "  Unassociated detections (new landmarks needed):";
//     for (std::size_t i = 0; i < tagIds_.size(); ++i) {
//         if (!detectionUsed[i]) {
//             std::cout << " tag " << tagIds_[i];
//         }
//     }
//     std::cout << std::endl;
    
//     return idxFeatures_;
// }
// In MeasurementSLAMAruco.cpp
// Replace the existing associate() method with this version:

const std::vector<int> & MeasurementSLAMAruco::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    idxFeatures_.clear();
    idxFeatures_.resize(idxLandmarks.size(), -1);  // Initialize with "no association"
    
    std::cout << "ArUco Data Association:" << std::endl;
    std::cout << "  Detected " << tagIds_.size() << " tags, tracking " << idxLandmarks.size() << " landmarks" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // NEW: Pre-filter detections - check which tags are within reliable FOV
    // ═══════════════════════════════════════════════════════════════════════════════
    std::vector<bool> detectionInFOV(tagIds_.size(), false);
    int numInFOV = 0;
    
    for (std::size_t i = 0; i < tagIds_.size(); ++i) {
        // Compute tag center from the 4 corners
        cv::Point2f centerPixel(0, 0);
        for (const auto& corner : corners_[i]) {
            centerPixel.x += corner.x;
            centerPixel.y += corner.y;
        }
        centerPixel.x /= 4.0f;
        centerPixel.y /= 4.0f;
        
        // Convert pixel to unit vector in camera frame
        cv::Vec3d centerVector = camera_.pixelToVector(cv::Vec2d(centerPixel.x, centerPixel.y));
        
        // Check if within reliable field of view
        if (camera_.isVectorWithinFOV(centerVector)) {
            detectionInFOV[i] = true;
            numInFOV++;
        } else {
            std::cout << "  WARNING: Tag " << tagIds_[i] 
                      << " at pixel [" << centerPixel.x << ", " << centerPixel.y 
                      << "] is outside reliable FOV - measurement will be IGNORED" << std::endl;
        }
    }
    
    std::cout << "  FOV filter: " << numInFOV << "/" << tagIds_.size() 
              << " detected tags are within reliable FOV" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Data Association: Match landmarks to detections (only those within FOV)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    // For each landmark in the map, try to find a corresponding detected tag
    for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
        std::size_t landmarkIdx = idxLandmarks[i];
        
        // Check if we have a tag ID stored for this landmark
        if (landmarkIdx < landmarkTagIds_.size()) {
            int expectedTagId = landmarkTagIds_[landmarkIdx];
            
            // Search for this tag ID in detected tags
            for (std::size_t j = 0; j < tagIds_.size(); ++j) {
                if (tagIds_[j] == expectedTagId) {
                    // ═══════════════════════════════════════════════════════════════
                    // NEW: Only associate if detection is within reliable FOV
                    // ═══════════════════════════════════════════════════════════════
                    if (detectionInFOV[j]) {
                        idxFeatures_[i] = static_cast<int>(j);  // Associate landmark i with detection j
                        std::cout << "  Associated landmark " << landmarkIdx 
                                  << " (tag " << expectedTagId << ") with detection " << j 
                                  << " (within FOV)" << std::endl;
                    } else {
                        std::cout << "  REJECTED: Landmark " << landmarkIdx 
                                  << " (tag " << expectedTagId << ") detected at edge - outside reliable FOV" 
                                  << std::endl;
                    }
                    break;
                }
            }
            
            if (idxFeatures_[i] == -1) {
                std::cout << "  Landmark " << landmarkIdx 
                          << " (tag " << expectedTagId << ") not detected this frame" << std::endl;
            }
        }
    }
    
    // Report any unassociated detections (these would need new landmarks)
    std::vector<bool> detectionUsed(tagIds_.size(), false);
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0) {
            detectionUsed[featureIdx] = true;
        }
    }
    
    std::cout << "  Unassociated detections (new landmarks needed):";
    for (std::size_t i = 0; i < tagIds_.size(); ++i) {
        if (!detectionUsed[i] && detectionInFOV[i]) {  // Only report if within FOV
            std::cout << " tag " << tagIds_[i];
        }
    }
    std::cout << std::endl;
    
    return idxFeatures_;
}

int MeasurementSLAMAruco::findLandmarkByTagId(int tagId)
{
    auto it = std::find(landmarkTagIds_.begin(), landmarkTagIds_.end(), tagId);
    if (it != landmarkTagIds_.end()) {
        return static_cast<int>(std::distance(landmarkTagIds_.begin(), it));
    }
    return -1;  // Not found
}

void MeasurementSLAMAruco::addLandmarkTagId(int tagId)
{
    landmarkTagIds_.push_back(tagId);
    std::cout << "Added landmark for tag " << tagId << " at index " << (landmarkTagIds_.size() - 1) << std::endl;
}

Eigen::VectorXd MeasurementSLAMAruco::estimateArucoLandmarkPose(
    const std::vector<cv::Point2f> & corners, 
    const Camera & camera,
    const Eigen::VectorXd* previousPose)  // ADD THIS PARAMETER
{
    // Define marker corners in local frame (same as static member)
    std::vector<cv::Point3f> objectPoints;
    objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-left
    objectPoints.push_back(cv::Point3f( MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-right
    objectPoints.push_back(cv::Point3f( MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-right
    objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-left
    
    // Use OpenCV PnP solver to estimate pose
    cv::Mat rvec, tvec;
    bool success;
    
    // ADD TEMPORAL CONSISTENCY LOGIC
    // ADD TEMPORAL CONSISTENCY LOGIC
    if (previousPose != nullptr && previousPose->size() == 6) {
        // Previous pose is in BODY FRAME - need to convert to CAMERA FRAME for solvePnP
        Eigen::Vector3d prevPos_body = previousPose->segment<3>(0);
        Eigen::Vector3d prevRPY_body = previousPose->segment<3>(3);
        
        // Transform from body frame back to camera frame
        Eigen::Matrix3d R_bc;
        R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
                1, 0, 0,   // Body Y (East) = Camera X (right)
                0, 1, 0;   // Body Z (Down) = Camera Y (down)
        
        // Convert body frame pose back to camera frame for solvePnP initial guess
        Eigen::Vector3d prevPos_camera = R_bc.transpose() * prevPos_body;
        Eigen::Matrix3d R_prevBody = rpy2rot(prevRPY_body);
        Eigen::Matrix3d R_prevCamera = R_bc.transpose() * R_prevBody;
        Eigen::Vector3d prevRPY_camera = rot2rpy(R_prevCamera);
        
        // Convert to OpenCV format (now in camera frame)
        cv::Mat rvec_guess = (cv::Mat_<double>(3,1) << prevRPY_camera(0), prevRPY_camera(1), prevRPY_camera(2));
        cv::Mat tvec_guess = (cv::Mat_<double>(3,1) << prevPos_camera(0), prevPos_camera(1), prevPos_camera(2));
        
        // Use initial guess with iterative solver
        success = cv::solvePnP(objectPoints, corners, camera.cameraMatrix, camera.distCoeffs, 
                            rvec_guess, tvec_guess, true, cv::SOLVEPNP_ITERATIVE);
        rvec = rvec_guess;
        tvec = tvec_guess;
        
        std::cout << "Used temporal consistency for pose estimation" << std::endl;
    
    } else {
        // No previous pose available - use default solver
        success = cv::solvePnP(objectPoints, corners, camera.cameraMatrix, camera.distCoeffs, 
                              rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        std::cout << "No previous pose - using default solver" << std::endl;
    }
    // END TEMPORAL CONSISTENCY LOGIC
    
    if (!success) {
        std::cerr << "Warning: Failed to estimate ArUco pose via solvePnP" << std::endl;
        return Eigen::VectorXd::Zero(6);  // Return zero pose
    }
    
    // Convert OpenCV results to our format
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);
    
    // Convert to Eigen
    Eigen::Matrix3d R;
    cv::cv2eigen(rotationMatrix, R);
    
    Eigen::Vector3d position(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    Eigen::Vector3d orientation = rot2rpy(R);
    
    // Return 6DOF pose [position, orientation]
    // ADD COORDINATE TRANSFORMATION (same as in predictArucoCorners)
    Eigen::Matrix3d R_bc;
    R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
            1, 0, 0,   // Body Y (East) = Camera X (right)  
            0, 1, 0;   // Body Z (Down) = Camera Y (down)

    // Transform from camera frame to body frame
    Eigen::Vector3d position_body = R_bc * position;
    Eigen::Matrix3d R_body = R_bc * R;
    Eigen::Vector3d orientation_body = rot2rpy(R_body);

    // Return 6DOF pose [position, orientation] in body frame
    Eigen::VectorXd pose(6);
    pose.segment<3>(0) = position_body;      // Use transformed position
    pose.segment<3>(3) = orientation_body;   // Use transformed orientation
    
    return pose;
}

// Eigen::VectorXd MeasurementSLAMAruco::estimateArucoLandmarkPose(const std::vector<cv::Point2f> & corners, const Camera & camera)
// {
//     // Define marker corners in local frame (same as static member)
//     std::vector<cv::Point3f> objectPoints;
//     objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-left
//     objectPoints.push_back(cv::Point3f( MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-right
//     objectPoints.push_back(cv::Point3f( MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-right
//     objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-left
    
//     // Use OpenCV PnP solver to estimate pose
//     cv::Mat rvec, tvec;
//     bool success = cv::solvePnP(objectPoints, corners, camera.cameraMatrix, camera.distCoeffs, rvec, tvec);
    
//     if (!success) {
//         std::cerr << "Warning: Failed to estimate ArUco pose via solvePnP" << std::endl;
//         return Eigen::VectorXd::Zero(6);  // Return zero pose
//     }
    
//     // Convert OpenCV results to our format
//     cv::Mat rotationMatrix;
//     cv::Rodrigues(rvec, rotationMatrix);
    
//     // Convert to Eigen
//     Eigen::Matrix3d R;
//     cv::cv2eigen(rotationMatrix, R);
    
//     Eigen::Vector3d position(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
//     Eigen::Vector3d orientation = rot2rpy(R);
    
//     // Return 6DOF pose [position, orientation]
//     Eigen::VectorXd pose(6);
//     pose.segment<3>(0) = position;
//     pose.segment<3>(3) = orientation;
    
//     return pose;
// }

// void MeasurementSLAMAruco::initializeNewLandmark(SystemSLAM & system, int tagId, const std::vector<cv::Point2f> & corners, const Camera & camera)
// {
//     // Estimate landmark pose
//     Eigen::VectorXd landmarkPose = estimateArucoLandmarkPose(corners, camera);
    
//     if (landmarkPose.norm() == 0.0) {
//         std::cerr << "Warning: Cannot initialize landmark for tag " << tagId << " - pose estimation failed" << std::endl;
//         return;
//     }
    
//     // Get current state and sqrt covariance
//     Eigen::VectorXd currentState = system.density.mean();
//     Eigen::MatrixXd currentSqrtCov = system.density.sqrtCov();
    
//     // Expand state vector to include new landmark
//     int oldDim = currentState.size();
//     int newDim = oldDim + 6;  // Add 6DOF for new landmark
    
//     Eigen::VectorXd newState(newDim);
//     newState.head(oldDim) = currentState;
//     newState.tail(6) = landmarkPose;
    
//     // Expand sqrt covariance matrix
//     Eigen::MatrixXd newSqrtCov = Eigen::MatrixXd::Zero(newDim, newDim);
//     newSqrtCov.topLeftCorner(oldDim, oldDim) = currentSqrtCov;
    
//     // Set initial uncertainty for new landmark (high uncertainty for better adaptation)
//     double positionUncertainty = 0.5;  // 50cm position uncertainty - much less confident
//     double orientationUncertainty = 0.5;  // ~30 degree orientation uncertainty - much less confident
    
//     for (int i = 0; i < 3; ++i) {
//         newSqrtCov(oldDim + i, oldDim + i) = positionUncertainty;
//         newSqrtCov(oldDim + 3 + i, oldDim + 3 + i) = orientationUncertainty;
//     }
    
//     // Update system density
//     system.density = GaussianInfo<double>::fromSqrtMoment(newState, newSqrtCov);
    
//     // Add tag ID to our tracking
//     addLandmarkTagId(tagId);
    
//     std::cout << "Initialized new landmark for tag " << tagId 
//               << " at position: [" << landmarkPose.segment<3>(0).transpose() << "]"
//               << " orientation: [" << landmarkPose.segment<3>(3).transpose() << "]" << std::endl;
// }

// In MeasurementSLAMAruco.cpp

void MeasurementSLAMAruco::initializeNewLandmark(
    SystemSLAM & system, 
    int tagId, 
    const std::vector<cv::Point2f> & corners, 
    const Camera & camera)
{
    std::cout << "\n=== Initializing Landmark for Tag " << tagId << " ===" << std::endl;
    
    // Use the fixed estimateArucoLandmarkPose function (includes temporal consistency and coordinate transforms)
    Eigen::VectorXd landmarkPose = estimateArucoLandmarkPose(corners, camera, nullptr);

    if (landmarkPose.norm() == 0.0) {
        std::cerr << "ERROR: Failed to estimate pose for tag " << tagId << std::endl;
        return;
    }

    // Extract position and orientation (already in body frame, then transformed to world frame)
    Eigen::Vector3d r_L_b = landmarkPose.segment<3>(0);
    Eigen::Vector3d theta_Lb = landmarkPose.segment<3>(3);
    
    std::cout << "Estimated landmark pose (body frame):" << std::endl;
    std::cout << "  Position: [" << r_L_b.transpose() << "] meters" << std::endl;
    std::cout << "  Orientation (RPY): [" << theta_Lb.transpose() << "] radians" << std::endl;
    
    // Get current body pose from SLAM state  
    Eigen::VectorXd currentState = system.density.mean();
    Eigen::Vector3d r_B_n = currentState.segment<3>(6);
    Eigen::Vector3d theta_bn = currentState.segment<3>(9);
    Eigen::Matrix3d R_nb = rpy2rot(theta_bn);
    
    // Transform landmark pose to world frame
    Eigen::Vector3d r_L_n = r_B_n + R_nb * r_L_b;
    Eigen::Matrix3d R_Lb = rpy2rot(theta_Lb);
    Eigen::Matrix3d R_nL = R_nb * R_Lb;
    Eigen::Vector3d theta_Ln = rot2rpy(R_nL);
    
    std::cout << "Landmark pose (world frame):" << std::endl;
    std::cout << "  Position: [" << r_L_n.transpose() << "] meters (NED)" << std::endl;
    std::cout << "  Orientation (RPY): [" << theta_Ln.transpose() << "] radians" << std::endl;
    
    // Expand state vector
    int oldDim = currentState.size();
    int newDim = oldDim + 6;
    
    Eigen::VectorXd newState(newDim);
    newState.head(oldDim) = currentState;
    newState.segment<3>(oldDim) = r_L_n;
    newState.segment<3>(oldDim + 3) = theta_Ln;
    
    // Expand covariance matrix
    Eigen::MatrixXd currentSqrtCov = system.density.sqrtCov();
    Eigen::MatrixXd newSqrtCov = Eigen::MatrixXd::Zero(newDim, newDim);
    newSqrtCov.topLeftCorner(oldDim, oldDim) = currentSqrtCov;
    
    double positionUncertainty = 0.5;
    double orientationUncertainty = 0.5;
    
    for (int i = 0; i < 3; ++i) {
        newSqrtCov(oldDim + i, oldDim + i) = positionUncertainty;
        newSqrtCov(oldDim + 3 + i, oldDim + 3 + i) = orientationUncertainty;
    }
    
    // Update system
    system.density = GaussianInfo<double>::fromSqrtMoment(newState, newSqrtCov);
    addLandmarkTagId(tagId);
    
    std::cout << "SUCCESS: Initialized landmark for tag " << tagId << std::endl;
    std::cout << "=== Initialization Complete ===" << std::endl;
}

// void MeasurementSLAMAruco::initializeNewLandmark(
//     SystemSLAM & system, 
//     int tagId, 
//     const std::vector<cv::Point2f> & corners, 
//     const Camera & camera)
// {
//     std::cout << "\n=== Initializing Landmark for Tag " << tagId << " ===" << std::endl;
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 1: Estimate Tag Pose in CAMERA FRAME using PnP
//     // ═══════════════════════════════════════════════════════════════
    
//     // Define the 4 corners of the marker in its LOCAL frame
//     // Marker is centered at origin, lying in XY plane (Z=0)
//     // Corner order: top-left, top-right, bottom-right, bottom-left
//     std::vector<cv::Point3f> objectPoints;
//     double half = MARKER_SIZE / 2.0;  // 0.083 meters
//     objectPoints.push_back(cv::Point3f(-half,  half, 0.0));  // Top-left
//     objectPoints.push_back(cv::Point3f( half,  half, 0.0));  // Top-right
//     objectPoints.push_back(cv::Point3f( half, -half, 0.0));  // Bottom-right
//     objectPoints.push_back(cv::Point3f(-half, -half, 0.0));  // Bottom-left
    
//     // Solve PnP to get tag pose relative to camera
//     cv::Vec3d rvec, tvec;
//     bool success = cv::solvePnP(
//         objectPoints,           // 3D points in marker frame
//         corners,                // 2D points in image
//         camera.cameraMatrix,    // Camera intrinsics
//         camera.distCoeffs,      // Distortion coefficients
//         rvec,                   // OUTPUT: Rotation (Rodrigues)
//         tvec,                   // OUTPUT: Translation
//         false,                  // Don't use extrinsic guess
//         cv::SOLVEPNP_ITERATIVE  // Algorithm
//     );
    
//     if (!success) {
//         std::cerr << "ERROR: solvePnP failed for tag " << tagId << std::endl;
//         return;
//     }
    
//     // Convert rotation vector to rotation matrix
//     cv::Mat R_cL_cv;
//     cv::Rodrigues(rvec, R_cL_cv);
    
//     // Convert OpenCV matrices to Eigen
//     Eigen::Matrix3d R_cL;  // Rotation from landmark frame to camera frame
//     Eigen::Vector3d r_L_c; // Landmark position in camera frame
    
//     cv::cv2eigen(R_cL_cv, R_cL);
//     r_L_c << tvec[0], tvec[1], tvec[2];
    
//     std::cout << "PnP Result (in camera frame):" << std::endl;
//     std::cout << "  Position: [" << r_L_c.transpose() << "] meters" << std::endl;
//     std::cout << "  Distance: " << r_L_c.norm() << " meters" << std::endl;
//     std::cout << "  Rotation det: " << R_cL.determinant() << " (should be 1.0)" << std::endl;

//     // Right after PnP
//     std::cout << "=== Camera Frame Convention Check ===" << std::endl;
//     std::cout << "r_L_c (marker in camera): [" << r_L_c.transpose() << "]" << std::endl;
//     std::cout << "  X (right?): " << r_L_c(0) << std::endl;
//     std::cout << "  Y (down?):  " << r_L_c(1) << std::endl;
//     std::cout << "  Z (forward?): " << r_L_c(2) << std::endl;

//     // Check if Z is positive and points forward
//     if (r_L_c(2) < 0) {
//         std::cerr << "ERROR: Tag is behind camera! Frame convention wrong!" << std::endl;
//     }
    
//     // Sanity checks
//     if (r_L_c(2) <= 0.0) {
//         std::cerr << "WARNING: Tag is behind camera (z=" << r_L_c(2) << ")!" << std::endl;
//     }
//     if (r_L_c.norm() < 0.2 || r_L_c.norm() > 20.0) {
//         std::cerr << "WARNING: Suspicious distance: " << r_L_c.norm() << "m" << std::endl;
//     }
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 2: Get Current Body Pose from SLAM State
//     // ═══════════════════════════════════════════════════════════════
    
//     Eigen::VectorXd currentState = system.density.mean();
    
//     // Extract body position in world frame (NED coordinates)
//     Eigen::Vector3d r_B_n = currentState.segment<3>(6);
    
//     // Extract body orientation in world frame (RPY Euler angles)
//     Eigen::Vector3d theta_bn = currentState.segment<3>(9);
    
//     // Convert body orientation to rotation matrix
//     // R_nb: rotation matrix from body frame to world frame
//     Eigen::Matrix3d R_nb = rpy2rot(theta_bn);
    
//     std::cout << "Current Body Pose (in world frame):" << std::endl;
//     std::cout << "  Position: [" << r_B_n.transpose() << "] meters (NED)" << std::endl;
//     std::cout << "  Orientation (RPY): [" << theta_bn.transpose() << "] radians" << std::endl;
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 3: Transform Landmark from CAMERA FRAME to WORLD FRAME
//     // ═══════════════════════════════════════════════════════════════
//     // Define rotation from NED body frame to camera frame (same as in prediction)
//     Eigen::Matrix3d R_bc;
//     R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
//             1, 0, 0,   // Body Y (East) = Camera X (right)
//             0, 1, 0;   // Body Z (Down) = Camera Y (down)

//     // Transform landmark from camera frame to body frame
//     Eigen::Vector3d r_L_b = R_bc * r_L_c;
//     Eigen::Matrix3d R_bL = R_bc * R_cL;

//     // Transform landmark position to world frame
//     // r_L^n = r_B^n + R_n^b * r_L^b
//     Eigen::Vector3d r_L_n = r_B_n + R_nb * r_L_b;

//     // Transform landmark orientation to world frame
//     // R_n^L = R_n^b * R_b^L
//     Eigen::Matrix3d R_nL = R_nb * R_bL;


//     // CRITICAL ASSUMPTION: Body frame = Camera frame (per assignment)
//     // // Therefore: r_B_n = r_C_n  and  R_nb = R_nc
    
//     // // Transform landmark position to world frame
//     // // r_L^n = r_B^n + R_n^b * r_L^c
//     // Eigen::Vector3d r_L_n = r_B_n + R_nb * r_L_c;
    
//     // // Transform landmark orientation to world frame
//     // // R_n^L = R_n^b * R_c^L
//     // Eigen::Matrix3d R_nL = R_nb * R_cL;
    
//     // Convert rotation matrix to Euler angles
//     Eigen::Vector3d theta_Ln = rot2rpy(R_nL);
    
//     std::cout << "Landmark Pose (in world frame):" << std::endl;
//     std::cout << "  Position: [" << r_L_n.transpose() << "] meters (NED)" << std::endl;
//     std::cout << "  Orientation (RPY): [" << theta_Ln.transpose() << "] radians" << std::endl;
    
//     // Verification: transform back to camera frame
//     // Verification: transform back to camera frame
//     Eigen::Vector3d r_L_b_check = R_nb.transpose() * (r_L_n - r_B_n);
//     Eigen::Vector3d r_L_c_check = R_bc.transpose() * r_L_b_check;
//     double error = (r_L_c_check - r_L_c).norm();
//     std::cout << "Round-trip verification error: " << error << " meters (should be ~0)" << std::endl;
    
//     if (error > 0.01) {
//         std::cerr << "ERROR: Coordinate transformation is incorrect!" << std::endl;
//         std::cerr << "  Original r_L_c:   [" << r_L_c.transpose() << "]" << std::endl;
//         std::cerr << "  Round-trip r_L_c: [" << r_L_c_check.transpose() << "]" << std::endl;
//         return;
//     }
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 4: Expand State Vector to Include New Landmark
//     // ═══════════════════════════════════════════════════════════════
    
//     // Get current state dimension
//     int oldDim = currentState.size();
//     int newDim = oldDim + 6;  // Add 6-DOF landmark (3 pos + 3 orient)
    
//     // Create expanded state vector
//     Eigen::VectorXd newState(newDim);
//     newState.head(oldDim) = currentState;  // Copy existing state
//     newState.segment<3>(oldDim) = r_L_n;        // Landmark position
//     newState.segment<3>(oldDim + 3) = theta_Ln; // Landmark orientation
    
//     std::cout << "State expansion:" << std::endl;
//     std::cout << "  Old dimension: " << oldDim << std::endl;
//     std::cout << "  New dimension: " << newDim << std::endl;
//     std::cout << "  Landmark index: " << oldDim << std::endl;
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 5: Expand Covariance Matrix (Square Root Form)
//     // ═══════════════════════════════════════════════════════════════
    
//     Eigen::MatrixXd currentSqrtCov = system.density.sqrtCov();
//     Eigen::MatrixXd newSqrtCov = Eigen::MatrixXd::Zero(newDim, newDim);
    
//     // Copy existing covariance
//     newSqrtCov.topLeftCorner(oldDim, oldDim) = currentSqrtCov;
    
//     // Set initial uncertainty for new landmark
//     // These values reflect our confidence in the PnP estimate
//     double positionUncertainty = 0.5;      // 50cm position uncertainty
//     double orientationUncertainty = 0.5;   // ~30° orientation uncertainty
    
//     // Fill in diagonal elements for new landmark
//     for (int i = 0; i < 3; ++i) {
//         newSqrtCov(oldDim + i, oldDim + i) = positionUncertainty;         // Position
//         newSqrtCov(oldDim + 3 + i, oldDim + 3 + i) = orientationUncertainty; // Orientation
//     }
    
//     std::cout << "Initial uncertainty:" << std::endl;
//     std::cout << "  Position: " << positionUncertainty << " meters (sqrt)" << std::endl;
//     std::cout << "  Orientation: " << orientationUncertainty << " radians (sqrt)" << std::endl;
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 6: Update System Density
//     // ═══════════════════════════════════════════════════════════════
    
//     system.density = GaussianInfo<double>::fromSqrtMoment(newState, newSqrtCov);
    
    
//     // ═══════════════════════════════════════════════════════════════
//     // STEP 7: Track Tag ID for Data Association
//     // ═══════════════════════════════════════════════════════════════
    
//     addLandmarkTagId(tagId);
    
//     std::cout << "SUCCESS: Initialized landmark for tag " << tagId << std::endl;
//     std::cout << "  Landmark index: " << (landmarkTagIds_.size() - 1) << std::endl;
//     std::cout << "  Total landmarks: " << landmarkTagIds_.size() << std::endl;
//     std::cout << "=== Initialization Complete ===" << std::endl;
// }

Eigen::VectorXd MeasurementSLAMAruco::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    // TODO: Simulate ArUco measurements given state
    std::cout << "TODO: Implement ArUco measurement simulation" << std::endl;
    return Eigen::VectorXd();
}

double MeasurementSLAMAruco::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & slamSystem = static_cast<const SystemSLAM &>(system);
    double logLikelihood = 0.0;
    
    // Compute likelihood for each associated feature/landmark pair (Equation 7)
    for (std::size_t i = 0; i < idxFeatures_.size(); ++i) {
        int featureIdx = idxFeatures_[i];
        
        if (featureIdx >= 0) {  // Valid association
            // Get detected corners for this feature
            const std::vector<cv::Point2f> & detectedCorners = corners_[featureIdx];
            
            // Predict corners for this landmark
            Eigen::MatrixXd J;  // Jacobian not needed for likelihood-only computation
            Eigen::Matrix<double, 8, 1> predictedCorners = predictArucoCorners(x, J, slamSystem, i);
            
            // Compute likelihood for each of the 4 corners
            for (int c = 0; c < 4; ++c) {
                // Detected corner position
                Eigen::Vector2d detected(detectedCorners[c].x, detectedCorners[c].y);
                
                // Predicted corner position
                Eigen::Vector2d predicted(predictedCorners(2*c), predictedCorners(2*c + 1));
                
                // Compute Gaussian likelihood: log N(detected; predicted, σ²I)
                Eigen::Vector2d error = detected - predicted;
                double likelihood = -0.5 * error.squaredNorm() / (sigma_ * sigma_) 
                                  - std::log(2.0 * M_PI * sigma_ * sigma_);
                logLikelihood += likelihood;
            }
        }
    }
    
    // Add penalty for unassociated visible landmarks (Equation 7: -4|U| log |Y|)
    int numUnassociated = 0;
    for (int featureIdx : idxFeatures_) {
        if (featureIdx == -1) {
            numUnassociated++;
        }
    }
    
    // Image area in pixels (approximate from camera calibration)
    double imageArea = camera_.imageSize.width * camera_.imageSize.height;
    logLikelihood -= 4.0 * numUnassociated * std::log(imageArea);
    
    return logLikelihood;
}

double MeasurementSLAMAruco::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Use autodiff to compute gradient
    using namespace autodiff;
    
    // Convert x to dual numbers
    Eigen::VectorX<dual> xdual = x.cast<dual>();
    
    // Create lambda that calls the logLikelihood function
    auto func = [&](const Eigen::VectorX<dual>& xd) {
        // We need a templated version of logLikelihood
        return logLikelihoodTemplate(xd, system);
    };
    
    // Compute value and gradient using autodiff
    dual loglik_dual;
    g = gradient(func, wrt(xdual), at(xdual), loglik_dual);
    
    return val(loglik_dual);
}

double MeasurementSLAMAruco::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    using namespace autodiff;
    
    // Convert x to dual numbers for Hessian computation
    Eigen::VectorX<dual2nd> x_dual2nd = x.cast<dual2nd>();
    
    // Create lambda that calls the template function
    auto func = [&](const Eigen::VectorX<dual2nd>& xd) {
        return logLikelihoodTemplate(xd, system);
    };
    
    // Compute value, gradient, and Hessian using autodiff
    dual2nd loglik_dual2nd;
    H = hessian(func, wrt(x_dual2nd), at(x_dual2nd), loglik_dual2nd, g);
    
    return val(loglik_dual2nd);
}

Eigen::Matrix<double, 8, 1> MeasurementSLAMAruco::predictArucoCorners(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{   

//////////////////////////////////////////////////////////
    // DEBUG OUTPUT - just print, don't redeclare!
    std::cout << "\n=== predictArucoCorners DEBUG ===" << std::endl;
    std::cout << "MARKER_SIZE = " << MARKER_SIZE << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "Corner " << i << " (local): " << CORNER_POSITIONS_LOCAL[i].transpose() << std::endl;
    }
    
    // Get camera pose from state (body frame = camera frame assumption)
    Eigen::Vector3d rBNn = x.segment<3>(6);   // Body position in world frame
    Eigen::Vector3d thetaBN = x.segment<3>(9); // Body orientation (RPY Euler angles)
    
    std::cout << "Body position: " << rBNn.transpose() << std::endl;
    std::cout << "Body orientation: " << thetaBN.transpose() << std::endl;
    
    // Get landmark pose from state
    std::size_t landmarkIdx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rLNn = x.segment<3>(landmarkIdx);     // Landmark position
    Eigen::Vector3d thetaLN = x.segment<3>(landmarkIdx + 3); // Landmark orientation
    
    std::cout << "Landmark position: " << rLNn.transpose() << std::endl;
    std::cout << "Landmark orientation: " << thetaLN.transpose() << std::endl;

////////////////////////////////////////////////////

    // // Get camera pose from state (body frame = camera frame assumption)
    // Eigen::Vector3d rBNn = x.segment<3>(6);   // Body position in world frame
    // Eigen::Vector3d thetaBN = x.segment<3>(9); // Body orientation (RPY Euler angles)
    
    // // Get landmark pose from state
    // std::size_t landmarkIdx = system.landmarkPositionIndex(idxLandmark);
    // Eigen::Vector3d rLNn = x.segment<3>(landmarkIdx);     // Landmark position
    // Eigen::Vector3d thetaLN = x.segment<3>(landmarkIdx + 3); // Landmark orientation
    
    // Convert Euler angles to rotation matrices
    Eigen::Matrix3d Rnb = rpy2rot(thetaBN);  // World to body rotation
    Eigen::Matrix3d RnL = rpy2rot(thetaLN);  // World to landmark rotation

        // ========== ADD THIS SECTION ==========
    // Define fixed rotation from NED body frame to camera frame
    // Camera frame convention: X=right, Y=down, Z=forward
    // NED (body) frame: X=North, Y=East, Z=Down
    // When body is at zero orientation, camera points North:
    //   - Camera forward (Z) points North (body X)
    //   - Camera right (X) points East (body Y)  
    //   - Camera down (Y) points Down (body Z)
    Eigen::Matrix3d R_bc;  // Rotation from camera frame to body frame
    R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
            1, 0, 0,   // Body Y (East) = Camera X (right)
            0, 1, 0;   // Body Z (Down) = Camera Y (down)
    // ========== END OF ADDITION ==========
    
    // Predict 4 corners in pixel coordinates
    Eigen::Matrix<double, 8, 1> predictedCorners;
    
    for (int c = 0; c < 4; ++c) {
        // // Get corner position in landmark local frame (from static member)
        // Eigen::Vector3d rLcL = CORNER_POSITIONS_LOCAL[c];
        
        // // Transform corner to world frame (Equation 8)
        // Eigen::Vector3d rCNn = RnL * rLcL + rLNn;
        
        // // Transform to camera frame (body frame = camera frame)
        // Eigen::Vector3d rCBb = Rnb.transpose() * (rCNn - rBNn);
        
        // // Project to image coordinates using camera calibration
        // Eigen::Vector2d pixelCoords = camera_.vectorToPixel(rCBb);
                // Get corner position in landmark local frame (from static member)
        Eigen::Vector3d rLcL = CORNER_POSITIONS_LOCAL[c];
        
        // Transform corner to world frame (Equation 8)
        Eigen::Vector3d rCNn = RnL * rLcL + rLNn;
        
        // Transform to body frame (body frame is NED)
        Eigen::Vector3d rCBb = Rnb.transpose() * (rCNn - rBNn);
        
        // ========== CHANGE THIS LINE ==========
        // Transform from body frame (NED) to camera frame
        Eigen::Vector3d rCCc = R_bc.transpose() * rCBb;
        
        // Project to image coordinates using camera calibration
        Eigen::Vector2d pixelCoords = camera_.vectorToPixel(rCCc);  // Now using camera frame!
        // ========== END OF CHANGE ==========
        
        
        // Store in result vector [x1,y1,x2,y2,x3,y3,x4,y4]
        predictedCorners(2*c) = pixelCoords(0);     // x coordinate
        predictedCorners(2*c + 1) = pixelCoords(1); // y coordinate
    }
    
    // Use forward-mode automatic differentiation to compute Jacobian (following Lab 8 pattern)
    using namespace autodiff;
    
    // Convert x to dual numbers
    Eigen::VectorX<dual> xdual = x.cast<dual>();
    
    // Create lambda that calls the template function
    auto func = [&](const Eigen::VectorX<dual>& xd) {
        return predictArucoCorners(xd, system, idxLandmark);
    };
    
    // Compute Jacobian using autodiff
    Eigen::Matrix<dual, 8, 1> ydual;
    J = jacobian(func, wrt(xdual), at(xdual), ydual);
    
    // Update predicted corners with dual values converted to double
    for (int i = 0; i < 8; ++i) {
        predictedCorners(i) = val(ydual(i));
    }
    
    return predictedCorners;
}

// void MeasurementSLAMAruco::update(SystemBase & system)
// {
//     SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);
    
//     // Get visible landmarks for data association
//     std::vector<std::size_t> visibleLandmarks;
//     for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
//         visibleLandmarks.push_back(i);
//     }
    
//     // Perform data association
//     associate(systemSLAM, visibleLandmarks);
    
//     // Initialize new landmarks for unassociated detections
//     for (std::size_t i = 0; i < tagIds_.size(); ++i) {
//         int tagId = tagIds_[i];
        
//         // Check if this tag already has a landmark
//         if (findLandmarkByTagId(tagId) == -1) {
//             // New tag - initialize landmark
//             std::cout << "Update: Initializing new landmark for tag " << tagId << std::endl;
//             initializeNewLandmark(systemSLAM, tagId, corners_[i], camera_);
//         }
//     }
    
//     // Call base class update which performs the optimization
//     Measurement::update(system);
// }

// In MeasurementSLAMAruco.cpp

void MeasurementSLAMAruco::update(SystemBase & system)
{
    // Cast to correct system type
    SystemSLAM & slamSystem = dynamic_cast<SystemSLAM &>(system);
    
    std::cout << "\n=== MeasurementSLAMAruco::update() ===" << std::endl;
    std::cout << "Detected " << tagIds_.size() << " tags" << std::endl;
    std::cout << "Current map has " << slamSystem.numberLandmarks() << " landmarks" << std::endl;
    
    // ────────────────────────────────────────────────────────────
    // STEP 1: Data Association
    // ────────────────────────────────────────────────────────────
    // Build list of all existing landmarks
    std::vector<std::size_t> existingLandmarks;
    for (std::size_t i = 0; i < slamSystem.numberLandmarks(); ++i) {
        existingLandmarks.push_back(i);
    }
    
    // Perform data association (matches detected tags to landmarks)
    // This populates idxFeatures_ vector
    associate(slamSystem, existingLandmarks);
    
    
    // ────────────────────────────────────────────────────────────
    // STEP 2: Remove Failed Landmarks (OPTIONAL - implement later)
    // ────────────────────────────────────────────────────────────
    // For now, skip this step. You can add it later for robustness.
    // This would remove landmarks that haven't been seen for N consecutive frames
    
    
    // ────────────────────────────────────────────────────────────
    // STEP 3 & 4: Identify Surplus Features and Initialize New Landmarks
    // ────────────────────────────────────────────────────────────
    
    // Track which detections were matched to existing landmarks
    std::vector<bool> detectionUsed(tagIds_.size(), false);
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0) {
            detectionUsed[featureIdx] = true;
        }
    }
    
    // // Initialize new landmarks for unassociated detections
    // int numNewLandmarks = 0;
    // for (std::size_t i = 0; i < tagIds_.size(); ++i) {
    //     if (!detectionUsed[i]) {
    //         // This tag was detected but not associated with any landmark
            
    //         // Check if we've already created a landmark for this tag ID
    //         int existingLandmarkIdx = findLandmarkByTagId(tagIds_[i]);
            
    //         if (existingLandmarkIdx == -1) {
    //             // This is a NEW tag we've never seen before
    //             std::cout << "  Initializing new landmark for tag " << tagIds_[i] << std::endl;
                
    //             initializeNewLandmark(slamSystem, tagIds_[i], corners_[i], camera_);
    //             numNewLandmarks++;
    //         } else {
    //             std::cout << "  Tag " << tagIds_[i] << " already has landmark (idx=" 
    //                       << existingLandmarkIdx << "), skipping" << std::endl;
    //         }
    //     }
    // }
    // Initialize new landmarks for unassociated detections
    int numNewLandmarks = 0;
    for (std::size_t i = 0; i < tagIds_.size(); ++i) {
        if (!detectionUsed[i]) {
            // This tag was detected but not associated with any landmark
            
            // Check if we've already created a landmark for this tag ID
            int existingLandmarkIdx = findLandmarkByTagId(tagIds_[i]);
            
            if (existingLandmarkIdx == -1) {
                // This is a NEW tag we've never seen before
                
                // ═══════════════════════════════════════════════════════════
                // Check if tag center is within reliable field of view
                // ═══════════════════════════════════════════════════════════
                
                // Compute tag center from the 4 corners
                cv::Point2f centerPixel(0, 0);
                for (const auto& corner : corners_[i]) {
                    centerPixel.x += corner.x;
                    centerPixel.y += corner.y;
                }
                centerPixel.x /= 4.0f;
                centerPixel.y /= 4.0f;
                
                // Convert pixel to unit vector in camera frame (already in camera coordinates)
                cv::Vec3d centerVector = camera_.pixelToVector(cv::Vec2d(centerPixel.x, centerPixel.y));
                
                // Check if within reliable field of view (no rotation needed - already in camera frame)
                if (!camera_.isVectorWithinFOV(centerVector)) {
                    std::cout << "  SKIPPED: Tag " << tagIds_[i] 
                            << " at pixel [" << centerPixel.x << ", " << centerPixel.y 
                            << "] - outside reliable FOV. Not initializing landmark." << std::endl;
                    continue;  // Skip this tag
                }
                
                // ═══════════════════════════════════════════════════════════
                // Tag is within FOV, proceed with initialization
                // ═══════════════════════════════════════════════════════════
                
                std::cout << "  Initializing new landmark for tag " << tagIds_[i] 
                        << " (center pixel: [" << centerPixel.x << ", " << centerPixel.y << "])" << std::endl;
                
                initializeNewLandmark(slamSystem, tagIds_[i], corners_[i], camera_);
                numNewLandmarks++;
            } else {
                std::cout << "  Tag " << tagIds_[i] << " already has landmark (idx="
                        << existingLandmarkIdx << "), skipping" << std::endl;
            }
        }
    }

    std::cout << "Initialized " << numNewLandmarks << " new landmarks this frame" << std::endl;

    // std::cout << "Initialized " << numNewLandmarks << " new landmarks" << std::endl;
    std::cout << "Map now has " << slamSystem.numberLandmarks() << " landmarks" << std::endl;
    
    
    // ────────────────────────────────────────────────────────────
    // STEP 5: Perform the Actual Measurement Update
    // ────────────────────────────────────────────────────────────
    // This calls the base class which runs the optimization
    // to refine the state estimate based on the measurements
    
    // IMPORTANT: Only do measurement update if we have valid associations
    int numAssociations = 0;
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0) numAssociations++;
    }
    
    if (numAssociations > 0) {
        std::cout << "Performing measurement update with " << numAssociations 
                  << " associated landmarks..." << std::endl;
        
        // This runs the BFGS trust region optimization to refine the state
        Measurement::update(system);
        
        std::cout << "Measurement update complete" << std::endl;
    } else {
        std::cout << "No valid associations - skipping measurement update" << std::endl;
    }
    
    std::cout << "=== update() complete ===" << std::endl;
}

Eigen::Matrix2d MeasurementSLAMAruco::extractTagCenterCovariance(const SystemSLAM & system, std::size_t idxLandmark) const
{
    try {
        // Use proper uncertainty propagation through feature density prediction
        GaussianInfo<double> featureDensity = predictFeatureDensity(system, idxLandmark);
        Eigen::MatrixXd cornerCov = featureDensity.cov();
        
        // Verify covariance matrix is the right size
        if (cornerCov.rows() != 8 || cornerCov.cols() != 8) {
            std::cerr << "Warning: Invalid covariance matrix size " << cornerCov.rows() << "x" << cornerCov.cols() << std::endl;
            return (sigma_ * sigma_) * Eigen::Matrix2d::Identity();
        }
        
        // Compute tag center covariance by averaging the 4 corner covariances
        // Tag center = (corner1 + corner2 + corner3 + corner4) / 4
        // Cov(center) = (1/16) * sum(Cov(corners)) + cross-correlation terms
        // For simplicity, we'll use the average of corner covariances
        Eigen::Matrix2d centerCov = Eigen::Matrix2d::Zero();
        for (int c = 0; c < 4; ++c) {
            Eigen::Matrix2d cornerCovBlock = cornerCov.block<2, 2>(2*c, 2*c);
            
            // Check for valid covariance values
            if (cornerCovBlock.hasNaN() || (!cornerCovBlock.allFinite())) {
                std::cerr << "Warning: Invalid covariance values for corner " << c << std::endl;
                return (sigma_ * sigma_) * Eigen::Matrix2d::Identity();
            }
            
            centerCov += cornerCovBlock;
        }
        centerCov /= 4.0;  // Average the covariances
        
        // Ensure the covariance matrix is positive definite
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(centerCov);
        if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0) {
            std::cerr << "Warning: Non-positive definite covariance matrix" << std::endl;
            return (sigma_ * sigma_) * Eigen::Matrix2d::Identity();
        }
        
        return centerCov;
    } catch (const std::exception & e) {
        // Fallback to identity covariance if uncertainty propagation fails
        std::cerr << "Warning: Failed to compute feature density for landmark " << idxLandmark 
                  << ", using default uncertainty: " << e.what() << std::endl;
        return (sigma_ * sigma_) * Eigen::Matrix2d::Identity();
    }
}

void MeasurementSLAMAruco::drawConfidenceEllipses(cv::Mat & image, const SystemSLAM & system, 
                                                  const std::vector<std::size_t> & idxLandmarks, 
                                                  double nSigma) const
{
    for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
        std::size_t landmarkIdx = idxLandmarks[i];
        // int featureIdx = (i < idxFeatures_.size()) ? idxFeatures_[i] : -1;
        int featureIdx = (landmarkIdx < idxFeatures_.size()) ? idxFeatures_[landmarkIdx] : -1;
        
        // Determine color based on association status
        cv::Scalar ellipseColor;
        cv::Scalar textColor;
        std::string statusText;
        
        if (featureIdx >= 0) {
            ellipseColor = cv::Scalar(255, 0, 0);  // Blue for tracked landmarks
            textColor = cv::Scalar(255, 255, 0);   // Cyan text
            statusText = "L" + std::to_string(landmarkIdx) + " -> T" + std::to_string(tagIds_[featureIdx]);
        } else {
            ellipseColor = cv::Scalar(0, 0, 255);  // Red for visible but not detected
            textColor = cv::Scalar(0, 255, 255);   // Yellow text
            statusText = "L" + std::to_string(landmarkIdx) + " (MISSING)";
        }
        
        try {
            // Get tag center covariance for this landmark
            Eigen::Matrix2d centerCov = extractTagCenterCovariance(system, landmarkIdx);
            
            // Get predicted corner positions
            Eigen::VectorXd currentState = system.density.mean();
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = predictArucoCorners(currentState, J, system, landmarkIdx);
            
            // Compute tag center as average of 4 corners
            cv::Point2f tagCenter(
                (predictedCorners(0) + predictedCorners(2) + predictedCorners(4) + predictedCorners(6)) / 4.0f,
                (predictedCorners(1) + predictedCorners(3) + predictedCorners(5) + predictedCorners(7)) / 4.0f
            );
            
            // Compute ellipse parameters from covariance
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(centerCov);
            
            if (eigensolver.info() == Eigen::Success) {
                Eigen::Vector2d eigenvals = eigensolver.eigenvalues();
                Eigen::Matrix2d eigenvecs = eigensolver.eigenvectors();
                
                // Ensure eigenvalues are positive and reasonable
                double minEigenval = std::max(eigenvals(0), 1e-3);
                double maxEigenval = std::max(eigenvals(1), 1e-3);
                
                // Clamp eigenvalues to reasonable range for visualization
                minEigenval = std::min(minEigenval, 10000.0);
                maxEigenval = std::min(maxEigenval, 10000.0);
                
                // Ellipse semi-axes (scaled by nSigma)
                double a = nSigma * std::sqrt(maxEigenval);  // Major axis
                double b = nSigma * std::sqrt(minEigenval);  // Minor axis
                
                // Ensure minimum visible size (5 pixels)
                a = std::max(a, 5.0);
                b = std::max(b, 5.0);
                
                // Rotation angle
                double angle = std::atan2(eigenvecs(1, 1), eigenvecs(0, 1)) * 180.0 / M_PI;
                
                // Draw the ellipse
                if (a > 0 && b > 0 && a < 10000 && b < 10000 && 
                    tagCenter.x >= 0 && tagCenter.x < image.cols && 
                    tagCenter.y >= 0 && tagCenter.y < image.rows) {
                    cv::ellipse(image, tagCenter, cv::Size2f(a, b), angle, 0, 360, ellipseColor, 2);
                    
                    // Calculate position for landmark ID text (outside ellipse)
                    cv::Point2f textOffset;
                    textOffset.x = (a + 20) * cos(angle * M_PI / 180.0);  // 20 pixels outside ellipse
                    textOffset.y = (a + 20) * sin(angle * M_PI / 180.0);
                    
                    cv::Point textPos(tagCenter.x + textOffset.x, tagCenter.y + textOffset.y);
                    
                    // Ensure text stays within image bounds
                    textPos.x = std::max(10, std::min(textPos.x, image.cols - 100));
                    textPos.y = std::max(20, std::min(textPos.y, image.rows - 10));
                    
                    // Draw landmark ID with background
                    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                    double fontScale = 0.8;
                    int thickness = 2;
                    
                    // Get text size for background
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(statusText, fontFace, fontScale, thickness, &baseline);
                    
                    // Draw black background
                    cv::Point bgTopLeft(textPos.x - 2, textPos.y - textSize.height - 2);
                    cv::Point bgBottomRight(textPos.x + textSize.width + 2, textPos.y + baseline + 2);
                    cv::rectangle(image, bgTopLeft, bgBottomRight, cv::Scalar(0, 0, 0), -1);
                    
                    // Draw landmark ID text
                    cv::putText(image, statusText, textPos, fontFace, fontScale, textColor, thickness);
                    
                } else {
                    // Fallback: draw simple circle with text
                    cv::circle(image, tagCenter, 15, ellipseColor, 2);
                    cv::putText(image, statusText, cv::Point(tagCenter.x + 20, tagCenter.y), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
                }
            } else {
                // Fallback: draw simple circle with text
                cv::circle(image, tagCenter, 15, ellipseColor, 2);
                cv::putText(image, statusText, cv::Point(tagCenter.x + 20, tagCenter.y), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
            }
        } catch (const std::exception & e) {
            std::cerr << "Warning: Failed to draw confidence ellipse for landmark " << landmarkIdx << ": " << e.what() << std::endl;
        }
    }
}