#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>  // ADD THIS - needed for SelfAdjointEigenSolver
#include <opencv2/core.hpp>
#include <cmath>
#include <numbers>

#include "../../src/MeasurementSLAMAruco.h"
#include "../../src/SystemSLAM.h"
#include "../../src/GaussianInfo.hpp"
#include "../../src/Camera.h"
#include "../../src/rotation.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

// Concrete test implementation of SystemSLAM
class TestSystemSLAM : public SystemSLAM
{
public:
    TestSystemSLAM() 
    : SystemSLAM(GaussianInfo<double>::fromSqrtMoment(
        Eigen::VectorXd::Zero(12), 
        Eigen::MatrixXd::Identity(12, 12)))
    {}
    
    virtual SystemSLAM * clone() const override
    {
        return new TestSystemSLAM(*this);
    }
    
    virtual std::size_t numberLandmarks() const override
    {
        // Calculate from state dimension
        // State structure: [unused(6), position(3), orientation(3), landmark1(6), landmark2(6), ...]
        if (density.dim() <= 12) return 0;
        return (density.dim() - 12) / 6;
    }
    
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override
    {
        // First landmark starts at index 12
        return 12 + idxLandmark * 6;
    }
};

// Helper function to create a simple test camera
static Camera createTestCamera()
{
    Camera cam;
    
    // Simple pinhole camera with no distortion
    cam.cameraMatrix = (cv::Mat_<double>(3, 3) << 
        500.0, 0.0, 320.0,  // fx, 0, cx
        0.0, 500.0, 240.0,  // 0, fy, cy
        0.0, 0.0, 1.0);     // 0, 0, 1
    
    cam.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);  // No distortion
    cam.imageSize = cv::Size(640, 480);
    
    return cam;
}

// Helper function to create a test SLAM system with known state
static TestSystemSLAM createTestSystem(const Eigen::VectorXd & state)
{
    TestSystemSLAM system;
    
    // Create Gaussian with given state and small uncertainty
    Eigen::MatrixXd sqrtCov = 0.01 * Eigen::MatrixXd::Identity(state.size(), state.size());
    system.density = GaussianInfo<double>::fromSqrtMoment(state, sqrtCov);
    
    return system;
}

// Helper function to create a minimal SLAM state vector
// State structure: [unused(6), position(3), orientation(3), landmark1(6), landmark2(6), ...]
static Eigen::VectorXd createMinimalState(const Eigen::Vector3d & bodyPos, 
                                          const Eigen::Vector3d & bodyRPY,
                                          const std::vector<Eigen::Vector3d> & landmarkPositions,
                                          const std::vector<Eigen::Vector3d> & landmarkOrientations)
{
    assert(landmarkPositions.size() == landmarkOrientations.size());
    
    const int numLandmarks = landmarkPositions.size();
    const int stateDim = 6 + 3 + 3 + 6 * numLandmarks;  // unused(6) + body(6) + landmarks(6*n)
    
    Eigen::VectorXd state = Eigen::VectorXd::Zero(stateDim);
    
    // Body pose (indices 6-11)
    state.segment<3>(6) = bodyPos;
    state.segment<3>(9) = bodyRPY;
    
    // Landmarks (starting at index 12)
    for (int i = 0; i < numLandmarks; ++i) {
        state.segment<3>(12 + 6*i) = landmarkPositions[i];
        state.segment<3>(12 + 6*i + 3) = landmarkOrientations[i];
    }
    
    return state;
}

SCENARIO("ArUco corner prediction - simple cases")
{
    GIVEN("A camera at origin looking forward, marker at origin facing camera")
    {
        Camera camera = createTestCamera();
        
        // Body/camera at origin, no rotation (NED frame: x=North, y=East, z=Down)
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);  // No rotation
        
        // Marker at 2 meters forward (positive Z in camera frame = North in NED)
        // Facing back towards camera (180° roll to flip marker)
        Eigen::Vector3d landmarkPos(2.0, 0.0, 0.0);  // 2m North
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);  // 180° roll
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        // Create measurement object
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};  // Empty corners, just for construction
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corner positions")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            THEN("Predicted corners are returned")
            {
                REQUIRE(predictedCorners.size() == 8);
                CAPTURE_EIGEN(predictedCorners);
                
                // All corners should be visible (positive depth)
                // and near image center due to marker being centered at 2m
                CHECK(predictedCorners(0) > 0);  // x1
                CHECK(predictedCorners(1) > 0);  // y1
                CHECK(predictedCorners(0) < camera.imageSize.width);
                CHECK(predictedCorners(1) < camera.imageSize.height);
            }
            
            THEN("Jacobian has correct dimensions")
            {
                REQUIRE(J.rows() == 8);  // 8 measurements (4 corners × 2 coords)
                REQUIRE(J.cols() == state.size());
                CAPTURE_EIGEN(J);
            }
            
            THEN("Jacobian is non-zero for body pose and landmark pose")
            {
                // Jacobian should be non-zero for body position (indices 6-8)
                CAPTURE_EIGEN(J.block(0, 6, 8, 3));
                CHECK(J.block(0, 6, 8, 3).norm() > 1e-6);
                
                // Jacobian should be non-zero for body orientation (indices 9-11)
                CAPTURE_EIGEN(J.block(0, 9, 8, 3));
                CHECK(J.block(0, 9, 8, 3).norm() > 1e-6);
                
                // Jacobian should be non-zero for landmark pose (indices 12-17)
                CAPTURE_EIGEN(J.block(0, 12, 8, 6));
                CHECK(J.block(0, 12, 8, 6).norm() > 1e-6);
            }
        }
    }
    
    GIVEN("A camera at origin, marker translated along X axis")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Marker at 2m forward, 1m to the right (East)
        Eigen::Vector3d landmarkPos(2.0, 1.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corner positions")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            THEN("Corners are shifted right in image")
            {
                CAPTURE_EIGEN(predictedCorners);
                
                // Compute mean x-coordinate of corners
                double meanX = (predictedCorners(0) + predictedCorners(2) + 
                               predictedCorners(4) + predictedCorners(6)) / 4.0;
                
                // Should be to the right of center (cx = 320)
                CHECK(meanX > 320.0);
            }
        }
    }
}

SCENARIO("ArUco corner prediction - geometry verification" * doctest::skip())
{
    GIVEN("A marker centered in front of camera")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        Eigen::Vector3d landmarkPos(2.0, 0.0, 0.0);  // 2m forward
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corners")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> corners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            THEN("Corner ordering is correct [TL, TR, BR, BL]")
            {
                // Extract corner coordinates
                cv::Point2f topLeft(corners(0), corners(1));
                cv::Point2f topRight(corners(2), corners(3));
                cv::Point2f bottomRight(corners(4), corners(5));
                cv::Point2f bottomLeft(corners(6), corners(7));
                
                CAPTURE_EIGEN(corners);
                
                // Top corners should have smaller y than bottom corners
                CHECK(topLeft.y < bottomLeft.y);
                CHECK(topRight.y < bottomRight.y);
                
                // Left corners should have smaller x than right corners
                CHECK(topLeft.x < topRight.x);
                CHECK(bottomLeft.x < bottomRight.x);
            }
            
            THEN("Marker appears square in image (approximately)")
            {
                // For a frontal marker, width and height should be similar
                double width1 = corners(2) - corners(0);   // TR.x - TL.x
                double width2 = corners(4) - corners(6);   // BR.x - BL.x
                double height1 = corners(6) - corners(0);  // BL.y - TL.y
                double height2 = corners(4) - corners(2);  // BR.y - TR.y
                
                double avgWidth = (width1 + width2) / 2.0;
                double avgHeight = (height1 + height2) / 2.0;
                
                // Should be approximately square (within 20% tolerance)
                CHECK(std::abs(avgWidth - avgHeight) / avgWidth < 0.2);
            }
        }
    }
}

SCENARIO("ArUco corner prediction - edge cases")
{
    GIVEN("A marker behind the camera")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Marker behind camera (negative Z in camera frame = South in NED)
        Eigen::Vector3d landmarkPos(-2.0, 0.0, 0.0);  // 2m South (behind)
        Eigen::Vector3d landmarkRPY(0.0, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corners")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners;
            
            // This might throw or return invalid values
            // We're testing that it doesn't crash
            REQUIRE_NOTHROW(
                predictedCorners = measurement.predictArucoCorners(state, J, system, 0)
            );
            
            THEN("Function completes without crashing")
            {
                CHECK(predictedCorners.size() == 8);
                CHECK(J.rows() == 8);
                CHECK(J.cols() == state.size());
            }
        }
    }
    
    GIVEN("A marker very close to camera")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Marker very close (0.3m)
        Eigen::Vector3d landmarkPos(0.3, 0.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corners")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            THEN("Corners may be outside image bounds")
            {
                CAPTURE_EIGEN(predictedCorners);
                
                // At least check they're finite
                CHECK(predictedCorners.allFinite());
            }
        }
    }
    
    GIVEN("A marker very far from camera")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Marker very far (100m)
        Eigen::Vector3d landmarkPos(100.0, 0.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting corners")
        {
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            THEN("Corners are very close together (small in image)")
            {
                CAPTURE_EIGEN(predictedCorners);
                
                // Compute bounding box size
                double minX = std::min({predictedCorners(0), predictedCorners(2), 
                                       predictedCorners(4), predictedCorners(6)});
                double maxX = std::max({predictedCorners(0), predictedCorners(2), 
                                       predictedCorners(4), predictedCorners(6)});
                double minY = std::min({predictedCorners(1), predictedCorners(3), 
                                       predictedCorners(5), predictedCorners(7)});
                double maxY = std::max({predictedCorners(1), predictedCorners(3), 
                                       predictedCorners(5), predictedCorners(7)});
                
                double width = maxX - minX;
                double height = maxY - minY;
                
                // Should be small (less than 20 pixels)
                CHECK(width < 20.0);
                CHECK(height < 20.0);
            }
        }
    }
}

SCENARIO("ArUco feature density prediction")
{
    GIVEN("A system with known state and uncertainty")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        Eigen::Vector3d landmarkPos(2.0, 0.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    {landmarkPos}, 
                                                    {landmarkRPY});
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting feature density for landmark 0")
        {
            GaussianInfo<double> featureDensity = 
                measurement.predictFeatureDensity(system, 0);
            
            THEN("Density has correct dimension")
            {
                REQUIRE(featureDensity.dim() == 8);  // 4 corners × 2 coords
            }
            
            THEN("Mean matches corner prediction")
            {
                Eigen::VectorXd mean = featureDensity.mean();
                REQUIRE(mean.size() == 8);
                
                // Compare with direct corner prediction
                Eigen::MatrixXd J;
                Eigen::Matrix<double, 8, 1> directPrediction = 
                    measurement.predictArucoCorners(state, J, system, 0);
                
                CAPTURE_EIGEN(mean);
                CAPTURE_EIGEN(directPrediction);
                
                // Should be very close (within numerical precision)
                CHECK(mean.isApprox(directPrediction, 1e-3));
            }
            
            THEN("Covariance is positive definite")
            {
                Eigen::MatrixXd cov = featureDensity.cov();
                REQUIRE(cov.rows() == 8);
                REQUIRE(cov.cols() == 8);
                
                // Check eigenvalues are positive
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
                CHECK(solver.eigenvalues().minCoeff() > 0);
            }
            
            THEN("Sqrt covariance is upper triangular")
            {
                Eigen::MatrixXd sqrtCov = featureDensity.sqrtCov();
                CHECK(sqrtCov.isUpperTriangular());
            }
        }
    }
}

SCENARIO("ArUco bundle feature density prediction")
{
    GIVEN("A system with multiple landmarks")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Two landmarks at different positions
        std::vector<Eigen::Vector3d> landmarkPositions = {
            Eigen::Vector3d(2.0, -1.0, 0.0),
            Eigen::Vector3d(2.0, 1.0, 0.0)
        };
        std::vector<Eigen::Vector3d> landmarkOrientations = {
            Eigen::Vector3d(std::numbers::pi, 0.0, 0.0),
            Eigen::Vector3d(std::numbers::pi, 0.0, 0.0)
        };
        
        Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                    landmarkPositions, 
                                                    landmarkOrientations);
        
        TestSystemSLAM system = createTestSystem(state);
        
        std::vector<int> tagIds = {0, 1};
        std::vector<std::vector<cv::Point2f>> corners = {{}, {}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        WHEN("Predicting bundle density for both landmarks")
        {
            std::vector<std::size_t> landmarkIndices = {0, 1};
            GaussianInfo<double> bundleDensity = 
                measurement.predictFeatureBundleDensity(system, landmarkIndices);
            
            THEN("Density dimension scales correctly")
            {
                REQUIRE(bundleDensity.dim() == 16);  // 2 landmarks × 4 corners × 2 coords
            }
            
            THEN("Mean contains predictions for both landmarks")
            {
                Eigen::VectorXd mean = bundleDensity.mean();
                REQUIRE(mean.size() == 16);
                
                CAPTURE_EIGEN(mean);
                
                // First 8 elements are landmark 0
                // Last 8 elements are landmark 1
                // Both should be finite
                CHECK(mean.segment<8>(0).allFinite());
                CHECK(mean.segment<8>(8).allFinite());
            }
            
            THEN("Covariance has proper block structure")
            {
                Eigen::MatrixXd cov = bundleDensity.cov();
                REQUIRE(cov.rows() == 16);
                REQUIRE(cov.cols() == 16);
                
                // Diagonal blocks should be non-zero
                CHECK(cov.block<8, 8>(0, 0).norm() > 1e-6);
                CHECK(cov.block<8, 8>(8, 8).norm() > 1e-6);
                
                // Off-diagonal blocks represent correlation between landmarks
                // These may be zero or non-zero depending on state correlations
                // Just check they're finite
                CHECK(cov.block<8, 8>(0, 8).allFinite());
                CHECK(cov.block<8, 8>(8, 0).allFinite());
            }
        }
    }
}

SCENARIO("ArUco corner prediction - coordinate frame consistency")
{
    GIVEN("Camera and landmark with known relative pose")
    {
        Camera camera = createTestCamera();
        
        // Camera at origin
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        
        // Landmark directly in front
        Eigen::Vector3d landmarkPos(2.0, 0.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        Eigen::VectorXd state1 = createMinimalState(bodyPos, bodyRPY, 
                                                     {landmarkPos}, 
                                                     {landmarkRPY});
        
        TestSystemSLAM system1 = createTestSystem(state1);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        Eigen::MatrixXd J1;
        Eigen::Matrix<double, 8, 1> corners1 = 
            measurement.predictArucoCorners(state1, J1, system1, 0);
        
        WHEN("Moving both camera and landmark by same amount")
        {
            // Move both 10m North
            Eigen::Vector3d bodyPos2(10.0, 0.0, 0.0);
            Eigen::Vector3d landmarkPos2(12.0, 0.0, 0.0);  // Still 2m ahead
            
            Eigen::VectorXd state2 = createMinimalState(bodyPos2, bodyRPY, 
                                                         {landmarkPos2}, 
                                                         {landmarkRPY});
            
            TestSystemSLAM system2 = createTestSystem(state2);
            
            Eigen::MatrixXd J2;
            Eigen::Matrix<double, 8, 1> corners2 = 
                measurement.predictArucoCorners(state2, J2, system2, 0);
            
            THEN("Predicted corners are identical (relative pose unchanged)")
            {
                CAPTURE_EIGEN(corners1);
                CAPTURE_EIGEN(corners2);
                
                CHECK(corners1.isApprox(corners2, 1e-6));
            }
        }
        
        WHEN("Rotating camera by 90 degrees about Z axis")
        {
            // Rotate camera 90° yaw (looking East instead of North)
            Eigen::Vector3d bodyRPY2(0.0, 0.0, std::numbers::pi / 2.0);
            
            // Landmark now needs to be East of camera to appear in same place
            Eigen::Vector3d landmarkPos2(0.0, 2.0, 0.0);  // 2m East
            Eigen::Vector3d landmarkRPY2(std::numbers::pi, 0.0, -std::numbers::pi / 2.0);
            
            Eigen::VectorXd state2 = createMinimalState(bodyPos, bodyRPY2, 
                                                         {landmarkPos2}, 
                                                         {landmarkRPY2});
            
            TestSystemSLAM system2 = createTestSystem(state2);
            
            Eigen::MatrixXd J2;
            Eigen::Matrix<double, 8, 1> corners2 = 
                measurement.predictArucoCorners(state2, J2, system2, 0);
            
            THEN("Corners appear in similar positions")
            {
                CAPTURE_EIGEN(corners1);
                CAPTURE_EIGEN(corners2);
                
                // Due to rotation, corners might be in different order
                // but center should be same
                double centerX1 = (corners1(0) + corners1(2) + corners1(4) + corners1(6)) / 4.0;
                double centerY1 = (corners1(1) + corners1(3) + corners1(5) + corners1(7)) / 4.0;
                double centerX2 = (corners2(0) + corners2(2) + corners2(4) + corners2(6)) / 4.0;
                double centerY2 = (corners2(1) + corners2(3) + corners2(5) + corners2(7)) / 4.0;
                
                CHECK(centerX1 == doctest::Approx(centerX2).epsilon(0.01));
                CHECK(centerY1 == doctest::Approx(centerY2).epsilon(0.01));
            }
        }
    }
}

SCENARIO("ArUco corner prediction - numerical stability")
{
    GIVEN("Marker at various distances")
    {
        Camera camera = createTestCamera();
        
        Eigen::Vector3d bodyPos(0.0, 0.0, 0.0);
        Eigen::Vector3d bodyRPY(0.0, 0.0, 0.0);
        Eigen::Vector3d landmarkRPY(std::numbers::pi, 0.0, 0.0);
        
        std::vector<int> tagIds = {0};
        std::vector<std::vector<cv::Point2f>> corners = {{}};
        MeasurementSLAMAruco measurement(0.0, camera, tagIds, corners);
        
        std::vector<double> distances = {0.5, 1.0, 2.0, 5.0, 10.0, 50.0};
        
        for (double distance : distances) {
            // Use INFO instead of dynamic WHEN
            INFO("Testing distance: ", distance, "m");
            
            Eigen::Vector3d landmarkPos(distance, 0.0, 0.0);
            Eigen::VectorXd state = createMinimalState(bodyPos, bodyRPY, 
                                                        {landmarkPos}, 
                                                        {landmarkRPY});
            
            TestSystemSLAM system = createTestSystem(state);
            
            Eigen::MatrixXd J;
            Eigen::Matrix<double, 8, 1> predictedCorners = 
                measurement.predictArucoCorners(state, J, system, 0);
            
            CHECK(predictedCorners.allFinite());
            CHECK(J.allFinite());
            
            // Jacobian shouldn't be too large or too small
            double jacobianNorm = J.norm();
            CHECK(jacobianNorm > 1e-10);
            CHECK(jacobianNorm < 1e10);
        }
    }
}