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
    , sigma_(8.0)  // 5 pixel measurement noise - more confident ArUco detection
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

const std::vector<int> & MeasurementSLAMAruco::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    idxFeatures_.clear();
    idxFeatures_.resize(idxLandmarks.size(), -1);  // Initialize with "no association"
    
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
        }
    }
    
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
                    }
                    break;
                }
            }
            
            (void)landmarkIdx;
        }
    }
    
    // Report any unassociated detections (these would need new landmarks)
    std::vector<bool> detectionUsed(tagIds_.size(), false);
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0) {
            detectionUsed[featureIdx] = true;
        }
    }
    
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
    } else {
        // No previous pose available - use default solver
        success = cv::solvePnP(objectPoints, corners, camera.cameraMatrix, camera.distCoeffs,
                              rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    }
    
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

    // Transform from camera frame to body frame
    Eigen::Matrix3d R_bc;
    R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
            1, 0, 0,   // Body Y (East) = Camera X (right)
            0, 1, 0;   // Body Z (Down) = Camera Y (down)

    Eigen::Vector3d position_body = R_bc * position;
    Eigen::Matrix3d R_body = R_bc * R;
    Eigen::Vector3d orientation_body = rot2rpy(R_body);

    // Return 6DOF pose [position, orientation] in body frame
    Eigen::VectorXd pose(6);
    pose.segment<3>(0) = position_body;
    pose.segment<3>(3) = orientation_body;

    return pose;
}

void MeasurementSLAMAruco::initializeNewLandmark(
    SystemSLAM & system,
    int tagId,
    const std::vector<cv::Point2f> & corners,
    const Camera & camera)
{
    // Estimate initial landmark pose using PnP
    Eigen::VectorXd landmarkPose = estimateArucoLandmarkPose(corners, camera, nullptr);

    Eigen::Vector3d r_L_n;
    Eigen::Vector3d theta_Ln;

    if (landmarkPose.norm() != 0.0) {
        // Successfully estimated pose - transform to world frame for prior mean
        Eigen::Vector3d r_L_b = landmarkPose.segment<3>(0);
        Eigen::Vector3d theta_Lb = landmarkPose.segment<3>(3);

        // Get current body pose from SLAM state
        Eigen::VectorXd currentState = system.density.mean();
        Eigen::Vector3d r_B_n = currentState.segment<3>(6);
        Eigen::Vector3d theta_bn = currentState.segment<3>(9);
        Eigen::Matrix3d R_nb = rpy2rot(theta_bn);

        // Transform landmark pose to world frame for prior mean
        r_L_n = r_B_n + R_nb * r_L_b;
        Eigen::Matrix3d R_Lb = rpy2rot(theta_Lb);
        Eigen::Matrix3d R_nL = R_nb * R_Lb;
        theta_Ln = rot2rpy(R_nL);
    } else {
        // Failed to estimate - use zero prior mean
        std::cerr << "WARNING: PnP failed, using zero prior mean" << std::endl;
        r_L_n.setZero();
        theta_Ln.setZero();
    }

    // Create prior mean (6DOF: position + orientation)
    Eigen::VectorXd mu_plus(6);
    mu_plus.segment<3>(0) = r_L_n;      // Position in world frame
    mu_plus.segment<3>(3) = theta_Ln;   // Orientation in world frame

    // Create weakly informative prior (large uncertainty)
    double epsilon_position = 0.5;
    double epsilon_orientation = 0.5;

    Eigen::MatrixXd sqrtInfo_plus = Eigen::MatrixXd::Zero(6, 6);
    sqrtInfo_plus.block<3,3>(0,0) = epsilon_position * Eigen::Matrix3d::Identity();
    sqrtInfo_plus.block<3,3>(3,3) = epsilon_orientation * Eigen::Matrix3d::Identity();

    // Create landmark prior in information form
    GaussianInfo<double> landmarkPrior = GaussianInfo<double>::fromSqrtMoment(
        mu_plus,
        sqrtInfo_plus
    );

    // Augment state with new landmark using operator*=
    system.density *= landmarkPrior;

    // Track tag ID for future data association
    addLandmarkTagId(tagId);
}

Eigen::VectorXd MeasurementSLAMAruco::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    (void)x;
    (void)system;
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
    // Get camera pose from state (body frame = camera frame assumption)
    Eigen::Vector3d rBNn = x.segment<3>(6);   // Body position in world frame
    Eigen::Vector3d thetaBN = x.segment<3>(9); // Body orientation (RPY Euler angles)

    // Get landmark pose from state
    std::size_t landmarkIdx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rLNn = x.segment<3>(landmarkIdx);     // Landmark position
    Eigen::Vector3d thetaLN = x.segment<3>(landmarkIdx + 3); // Landmark orientation

    // Convert Euler angles to rotation matrices
    Eigen::Matrix3d Rnb = rpy2rot(thetaBN);  // World to body rotation
    Eigen::Matrix3d RnL = rpy2rot(thetaLN);  // World to landmark rotation

    // Define fixed rotation from NED body frame to camera frame
    // Camera frame convention: X=right, Y=down, Z=forward
    // NED (body) frame: X=North, Y=East, Z=Down
    Eigen::Matrix3d R_bc;  // Rotation from camera frame to body frame
    R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
            1, 0, 0,   // Body Y (East) = Camera X (right)
            0, 1, 0;   // Body Z (Down) = Camera Y (down)

    // Predict 4 corners in pixel coordinates
    Eigen::Matrix<double, 8, 1> predictedCorners;

    for (int c = 0; c < 4; ++c) {
        // Get corner position in landmark local frame (from static member)
        Eigen::Vector3d rLcL = CORNER_POSITIONS_LOCAL[c];

        // Transform corner to world frame (Equation 8)
        Eigen::Vector3d rCNn = RnL * rLcL + rLNn;

        // Transform to body frame (body frame is NED)
        Eigen::Vector3d rCBb = Rnb.transpose() * (rCNn - rBNn);

        // Transform from body frame (NED) to camera frame
        Eigen::Vector3d rCCc = R_bc.transpose() * rCBb;

        // Project to image coordinates using camera calibration
        Eigen::Vector2d pixelCoords = camera_.vectorToPixel(rCCc);

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

void MeasurementSLAMAruco::update(SystemBase & system)
{
    // Cast to correct system type
    SystemSLAM & slamSystem = dynamic_cast<SystemSLAM &>(system);
    
    // ─────────────────────────────────────────────────────────────────────
    // STEP 1: Data Association for Existing Landmarks
    // ─────────────────────────────────────────────────────────────────────
    
    // Build list of all existing landmarks
    std::vector<std::size_t> existingLandmarks;
    for (std::size_t i = 0; i < slamSystem.numberLandmarks(); ++i) {
        existingLandmarks.push_back(i);
    }
    
    // Perform data association (matches detected tags to existing landmarks)
    // This populates idxFeatures_ vector
    associate(slamSystem, existingLandmarks);

    // ─────────────────────────────────────────────────────────────────────
    // STEP 2: Identify Unassociated Detections (Need New Landmarks)
    // ─────────────────────────────────────────────────────────────────────
    
    // Mark which detections were used
    std::vector<bool> detectionUsed(tagIds_.size(), false);
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0 && featureIdx < static_cast<int>(tagIds_.size())) {
            detectionUsed[featureIdx] = true;
        }
    }
    
    // Find unassociated detections
    std::vector<int> unassociatedDetections;
    for (std::size_t i = 0; i < tagIds_.size(); ++i) {
        if (!detectionUsed[i]) {
            unassociatedDetections.push_back(i);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // STEP 3: Initialize New Landmarks with FOV Check
    // ─────────────────────────────────────────────────────────────────────
    
    int numNewLandmarks = 0;
    
    for (int detectionIdx : unassociatedDetections) {
        int tagId = tagIds_[detectionIdx];
        
        // Check if we've already created a landmark for this tag ID
        int existingLandmarkIdx = findLandmarkByTagId(tagId);
        
        if (existingLandmarkIdx == -1) {
            // This is a NEW tag we've never seen before
            // FOV check: Compute tag center and verify it's within reliable FOV
            const std::vector<cv::Point2f>& corners = corners_[detectionIdx];
            cv::Point2f tagCenter(0, 0);
            for (const auto& corner : corners) {
                tagCenter += corner;
            }
            tagCenter *= (1.0f / 4.0f);
            
            // Convert pixel to camera vector and check FOV
            cv::Vec2d tagCenterVec(tagCenter.x, tagCenter.y);  // Convert Point2f to Vec2d
            cv::Vec3d centerVector = camera_.pixelToVector(tagCenterVec);
            
            if (!camera_.isVectorWithinFOV(centerVector)) {
                continue;  // Skip this landmark initialization
            }

            // Get the landmark index BEFORE initialization
            int newLandmarkIdx = slamSystem.numberLandmarks();
            
            // Initialize new landmark using operator*= approach
            initializeNewLandmark(slamSystem, tagId, corners_[detectionIdx], camera_);
            
            // ═════════════════════════════════════════════════════════════
            // CRITICAL: Force association with the detection that created it
            // ═════════════════════════════════════════════════════════════
            
            // Expand idxFeatures_ to include the new landmark
            if (idxFeatures_.size() <= static_cast<size_t>(newLandmarkIdx)) {
                idxFeatures_.resize(newLandmarkIdx + 1, -1);
            }
            
            // Associate the new landmark with its detection
            idxFeatures_[newLandmarkIdx] = detectionIdx;

            numNewLandmarks++;
        }
    }

    (void)numNewLandmarks;

    // ─────────────────────────────────────────────────────────────────────
    // STEP 4: Perform the Actual Measurement Update
    // ─────────────────────────────────────────────────────────────────────
    
    // Count valid associations
    int numAssociations = 0;
    for (int featureIdx : idxFeatures_) {
        if (featureIdx >= 0) numAssociations++;
    }
    
    if (numAssociations > 0) {
        // This runs the BFGS trust region optimization to refine the state
        // It will use ALL landmarks in idxFeatures_ (including newly initialized ones!)
        Measurement::update(system);
    }
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