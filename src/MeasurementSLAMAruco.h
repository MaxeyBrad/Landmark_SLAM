#ifndef MEASUREMENTSLAMARUCO_H
#define MEASUREMENTSLAMARUCO_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementSLAMAruco : public MeasurementSLAM
{
public:
    MeasurementSLAMAruco(double time, 
                         const Camera & camera,
                         const std::vector<int> & tagIds,
                         const std::vector<std::vector<cv::Point2f>> & corners);
    
    MeasurementSLAM * clone() const override;
    virtual ~MeasurementSLAMAruco() override;
    
    // Core SLAM interface methods
    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const override;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const override;
    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) override;
    
    // Measurement interface methods
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // ArUco-specific prediction methods
    template <typename Scalar> 
    Eigen::Matrix<Scalar, 8, 1> predictArucoCorners(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const;
    
    Eigen::Matrix<double, 8, 1> predictArucoCorners(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const;
    
    // Accessors
    const std::vector<int> & getTagIds() const { return tagIds_; }
    const std::vector<std::vector<cv::Point2f>> & getCorners() const { return corners_; }
    
    // Static landmark management for ArUco tags
    static std::vector<int> & getLandmarkTagIds() { return landmarkTagIds_; }
    static int findLandmarkByTagId(int tagId);
    static void addLandmarkTagId(int tagId);
    
    // Landmark initialization
    static Eigen::VectorXd estimateArucoLandmarkPose(const std::vector<cv::Point2f> & corners, const Camera & camera);
    static void initializeNewLandmark(SystemSLAM & system, int tagId, const std::vector<cv::Point2f> & corners, const Camera & camera);
    
    // Visualization support
    std::vector<Eigen::Matrix2d> extractCornerCovariances(const SystemSLAM & system, std::size_t idxLandmark) const;
    void drawConfidenceEllipses(cv::Mat & image, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks, double nSigma = 3.0) const;
    
protected:
    virtual void update(SystemBase & system) override;
    
    // ArUco marker data
    std::vector<int> tagIds_;                                    // Detected tag IDs
    std::vector<std::vector<cv::Point2f>> corners_;             // 4 corners per tag
    std::vector<int> idxFeatures_;                              // Association results
    
    // Static storage for landmark tag IDs (shared across all measurements)
    static std::vector<int> landmarkTagIds_;
    
    // ArUco marker parameters
    static constexpr double MARKER_SIZE = 0.166;                // 166mm edge length
    double sigma_;                                              // Corner measurement noise (pixels)
    
    // Corner positions in marker local frame (from assignment Eq. 9)
    static const std::array<Eigen::Vector3d, 4> CORNER_POSITIONS_LOCAL;
};

// Template implementation for ArUco corner prediction (works with autodiff)
template <typename Scalar>
Eigen::Matrix<Scalar, 8, 1> MeasurementSLAMAruco::predictArucoCorners(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Get camera pose from state (body frame = camera frame assumption)
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);   // Body position in world frame
    Eigen::Vector3<Scalar> thetaBN = x.template segment<3>(9); // Body orientation (RPY Euler angles)
    
    // Get landmark pose from state
    std::size_t landmarkIdx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rLNn = x.template segment<3>(landmarkIdx);     // Landmark position
    Eigen::Vector3<Scalar> thetaLN = x.template segment<3>(landmarkIdx + 3); // Landmark orientation
    
    // Convert Euler angles to rotation matrices
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(thetaBN);  // World to body rotation
    Eigen::Matrix3<Scalar> RnL = rpy2rot(thetaLN);  // World to landmark rotation
    
    // Predict 4 corners in pixel coordinates
    Eigen::Matrix<Scalar, 8, 1> predictedCorners;
    
    for (int c = 0; c < 4; ++c) {
        // Get corner position in landmark local frame (convert to Scalar type)
        Eigen::Vector3<Scalar> rLcL = CORNER_POSITIONS_LOCAL[c].cast<Scalar>();
        
        // Transform corner to world frame (Equation 8)
        Eigen::Vector3<Scalar> rCNn = RnL * rLcL + rLNn;
        
        // Transform to camera frame (body frame = camera frame)
        Eigen::Vector3<Scalar> rCBb = Rnb.transpose() * (rCNn - rBNn);
        
        // Project to image coordinates using camera calibration
        Eigen::Vector2<Scalar> pixelCoords = camera_.vectorToPixel(rCBb);
        
        // Store in result vector [x1,y1,x2,y2,x3,y3,x4,y4]
        predictedCorners(2*c) = pixelCoords(0);     // x coordinate
        predictedCorners(2*c + 1) = pixelCoords(1); // y coordinate
    }
    
    return predictedCorners;
}

#endif