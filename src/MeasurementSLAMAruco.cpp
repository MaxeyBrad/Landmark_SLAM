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
    , sigma_(1.0)  // 1 pixel measurement noise - good ArUco detection
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
    // TODO: Implement feature prediction for bundle of ArUco landmarks
    // This should predict measurements for multiple landmarks simultaneously
    std::cout << "TODO: Implement predictFeatureBundleDensity for " << idxLandmarks.size() << " landmarks" << std::endl;
    
    // Return dummy for now - 8D per landmark
    int totalDim = 8 * idxLandmarks.size();
    return GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Zero(totalDim, totalDim));
}

const std::vector<int> & MeasurementSLAMAruco::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    idxFeatures_.clear();
    idxFeatures_.resize(idxLandmarks.size(), -1);  // Initialize with "no association"
    
    std::cout << "ArUco Data Association:" << std::endl;
    std::cout << "  Detected " << tagIds_.size() << " tags, tracking " << idxLandmarks.size() << " landmarks" << std::endl;
    
    // For each landmark in the map, try to find a corresponding detected tag
    for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
        std::size_t landmarkIdx = idxLandmarks[i];
        
        // Check if we have a tag ID stored for this landmark
        if (landmarkIdx < landmarkTagIds_.size()) {
            int expectedTagId = landmarkTagIds_[landmarkIdx];
            
            // Search for this tag ID in detected tags
            for (std::size_t j = 0; j < tagIds_.size(); ++j) {
                if (tagIds_[j] == expectedTagId) {
                    idxFeatures_[i] = static_cast<int>(j);  // Associate landmark i with detection j
                    std::cout << "  Associated landmark " << landmarkIdx << " (tag " << expectedTagId << ") with detection " << j << std::endl;
                    break;
                }
            }
            
            if (idxFeatures_[i] == -1) {
                std::cout << "  Landmark " << landmarkIdx << " (tag " << expectedTagId << ") not detected this frame" << std::endl;
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
        if (!detectionUsed[i]) {
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

Eigen::VectorXd MeasurementSLAMAruco::estimateArucoLandmarkPose(const std::vector<cv::Point2f> & corners, const Camera & camera)
{
    // Define marker corners in local frame (same as static member)
    std::vector<cv::Point3f> objectPoints;
    objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-left
    objectPoints.push_back(cv::Point3f( MARKER_SIZE/2,  MARKER_SIZE/2, 0));  // top-right
    objectPoints.push_back(cv::Point3f( MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-right
    objectPoints.push_back(cv::Point3f(-MARKER_SIZE/2, -MARKER_SIZE/2, 0));  // bottom-left
    
    // Use OpenCV PnP solver to estimate pose
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objectPoints, corners, camera.cameraMatrix, camera.distCoeffs, rvec, tvec);
    
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
    Eigen::VectorXd pose(6);
    pose.segment<3>(0) = position;
    pose.segment<3>(3) = orientation;
    
    return pose;
}

void MeasurementSLAMAruco::initializeNewLandmark(SystemSLAM & system, int tagId, const std::vector<cv::Point2f> & corners, const Camera & camera)
{
    // Estimate landmark pose
    Eigen::VectorXd landmarkPose = estimateArucoLandmarkPose(corners, camera);
    
    if (landmarkPose.norm() == 0.0) {
        std::cerr << "Warning: Cannot initialize landmark for tag " << tagId << " - pose estimation failed" << std::endl;
        return;
    }
    
    // Get current state and sqrt covariance
    Eigen::VectorXd currentState = system.density.mean();
    Eigen::MatrixXd currentSqrtCov = system.density.sqrtCov();
    
    // Expand state vector to include new landmark
    int oldDim = currentState.size();
    int newDim = oldDim + 6;  // Add 6DOF for new landmark
    
    Eigen::VectorXd newState(newDim);
    newState.head(oldDim) = currentState;
    newState.tail(6) = landmarkPose;
    
    // Expand sqrt covariance matrix
    Eigen::MatrixXd newSqrtCov = Eigen::MatrixXd::Zero(newDim, newDim);
    newSqrtCov.topLeftCorner(oldDim, oldDim) = currentSqrtCov;
    
    // Set initial uncertainty for new landmark (relatively high uncertainty)
    double positionUncertainty = 0.1;  // 10cm position uncertainty
    double orientationUncertainty = 0.1;  // ~6 degree orientation uncertainty
    
    for (int i = 0; i < 3; ++i) {
        newSqrtCov(oldDim + i, oldDim + i) = positionUncertainty;
        newSqrtCov(oldDim + 3 + i, oldDim + 3 + i) = orientationUncertainty;
    }
    
    // Update system density
    system.density = GaussianInfo<double>::fromSqrtMoment(newState, newSqrtCov);
    
    // Add tag ID to our tracking
    addLandmarkTagId(tagId);
    
    std::cout << "Initialized new landmark for tag " << tagId 
              << " at position: [" << landmarkPose.segment<3>(0).transpose() << "]"
              << " orientation: [" << landmarkPose.segment<3>(3).transpose() << "]" << std::endl;
}

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
    // TODO: Implement log-likelihood with gradient and Hessian
    std::cout << "TODO: Implement ArUco log-likelihood with gradient and Hessian" << std::endl;
    g = Eigen::VectorXd::Zero(x.size());
    H = Eigen::MatrixXd::Zero(x.size(), x.size());
    return 0.0;
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
    
    // Predict 4 corners in pixel coordinates
    Eigen::Matrix<double, 8, 1> predictedCorners;
    
    for (int c = 0; c < 4; ++c) {
        // Get corner position in landmark local frame (from static member)
        Eigen::Vector3d rLcL = CORNER_POSITIONS_LOCAL[c];
        
        // Transform corner to world frame (Equation 8)
        Eigen::Vector3d rCNn = RnL * rLcL + rLNn;
        
        // Transform to camera frame (body frame = camera frame)
        Eigen::Vector3d rCBb = Rnb.transpose() * (rCNn - rBNn);
        
        // Project to image coordinates using camera calibration
        Eigen::Vector2d pixelCoords = camera_.vectorToPixel(rCBb);
        
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
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);
    
    // Get visible landmarks for data association
    std::vector<std::size_t> visibleLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
        visibleLandmarks.push_back(i);
    }
    
    // Perform data association
    associate(systemSLAM, visibleLandmarks);
    
    // Initialize new landmarks for unassociated detections
    for (std::size_t i = 0; i < tagIds_.size(); ++i) {
        int tagId = tagIds_[i];
        
        // Check if this tag already has a landmark
        if (findLandmarkByTagId(tagId) == -1) {
            // New tag - initialize landmark
            std::cout << "Update: Initializing new landmark for tag " << tagId << std::endl;
            initializeNewLandmark(systemSLAM, tagId, corners_[i], camera_);
        }
    }
    
    // Call base class update which performs the optimization
    Measurement::update(system);
}

std::vector<Eigen::Matrix2d> MeasurementSLAMAruco::extractCornerCovariances(const SystemSLAM & system, std::size_t idxLandmark) const
{
    std::vector<Eigen::Matrix2d> cornerCovariances(4);
    
    try {
        // Use proper uncertainty propagation through feature density prediction
        GaussianInfo<double> featureDensity = predictFeatureDensity(system, idxLandmark);
        Eigen::MatrixXd cornerCov = featureDensity.cov();
        
        // Extract 2x2 covariance blocks for each corner
        for (int c = 0; c < 4; ++c) {
            cornerCovariances[c] = cornerCov.block<2, 2>(2*c, 2*c);
        }
    } catch (const std::exception & e) {
        // Fallback to identity covariances if uncertainty propagation fails
        std::cerr << "Warning: Failed to compute feature density for landmark " << idxLandmark 
                  << ", using default uncertainty: " << e.what() << std::endl;
        for (int c = 0; c < 4; ++c) {
            cornerCovariances[c] = (sigma_ * sigma_) * Eigen::Matrix2d::Identity();
        }
    }
    
    return cornerCovariances;
}

void MeasurementSLAMAruco::drawConfidenceEllipses(cv::Mat & image, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks, double nSigma) const
{
    for (std::size_t i = 0; i < idxLandmarks.size(); ++i) {
        std::size_t landmarkIdx = idxLandmarks[i];
        int featureIdx = (i < idxFeatures_.size()) ? idxFeatures_[i] : -1;
        
        // Determine color based on association status
        cv::Scalar ellipseColor;
        if (featureIdx >= 0) {
            ellipseColor = cv::Scalar(255, 0, 0);  // Blue for tracked landmarks
        } else {
            ellipseColor = cv::Scalar(0, 0, 255);  // Red for visible but not detected
        }
        
        try {
            // Get corner covariances for this landmark
            std::vector<Eigen::Matrix2d> cornerCovs = extractCornerCovariances(system, landmarkIdx);
            
            // Get predicted corner positions
            Eigen::VectorXd currentState = system.density.mean();
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = predictArucoCorners(currentState, J, system, landmarkIdx);
            
            // Draw ellipse for each corner
            for (int c = 0; c < 4; ++c) {
                cv::Point2f center(predictedCorners(2*c), predictedCorners(2*c + 1));
                
                // Compute ellipse parameters from covariance
                Eigen::Matrix2d cov = cornerCovs[c];
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
                
                if (eigensolver.info() == Eigen::Success) {
                    Eigen::Vector2d eigenvals = eigensolver.eigenvalues();
                    Eigen::Matrix2d eigenvecs = eigensolver.eigenvectors();
                    
                    // Ellipse semi-axes (scaled by nSigma)
                    double a = nSigma * std::sqrt(eigenvals(1));  // Major axis
                    double b = nSigma * std::sqrt(eigenvals(0));  // Minor axis
                    
                    // Rotation angle
                    double angle = std::atan2(eigenvecs(1, 1), eigenvecs(0, 1)) * 180.0 / M_PI;
                    
                    // Draw ellipse
                    cv::ellipse(image, center, cv::Size2f(a, b), angle, 0, 360, ellipseColor, 1);
                }
            }
        } catch (const std::exception & e) {
            // Skip this landmark if covariance extraction fails
            std::cerr << "Warning: Failed to draw confidence ellipse for landmark " << landmarkIdx << ": " << e.what() << std::endl;
        }
    }
}