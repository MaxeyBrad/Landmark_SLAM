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
    
protected:
    virtual void update(SystemBase & system) override;
    
    // ArUco marker data
    std::vector<int> tagIds_;                                    // Detected tag IDs
    std::vector<std::vector<cv::Point2f>> corners_;             // 4 corners per tag
    std::vector<int> idxFeatures_;                              // Association results
    
    // ArUco marker parameters
    static constexpr double MARKER_SIZE = 0.166;                // 166mm edge length
    double sigma_;                                              // Corner measurement noise (pixels)
    
    // Corner positions in marker local frame (from assignment Eq. 9)
    static const std::array<Eigen::Vector3d, 4> CORNER_POSITIONS_LOCAL;
};

// Template implementation for ArUco corner prediction
template <typename Scalar>
Eigen::Matrix<Scalar, 8, 1> MeasurementSLAMAruco::predictArucoCorners(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // TODO: Implement ArUco corner prediction
    // 1. Get camera pose from state
    // 2. Get landmark pose from state (6 DOF: position + orientation)
    // 3. Transform 4 local corner positions to world frame
    // 4. Project to image coordinates
    // 5. Return 8D vector [x1,y1,x2,y2,x3,y3,x4,y4]
    
    Eigen::Matrix<Scalar, 8, 1> predictedCorners;
    predictedCorners.setZero();
    return predictedCorners;
}

#endif