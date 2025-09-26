#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"
#include "Pose.hpp"

struct Chessboard
{
    cv::Size boardSize;
    float squareSize;

    void write(cv::FileStorage & fs) const;                 // OpenCV serialisation
    void read(const cv::FileNode & node);                   // OpenCV serialisation

    std::vector<cv::Point3f> gridPoints() const;
    friend std::ostream & operator<<(std::ostream &, const Chessboard &);
};

struct Camera;

struct ChessboardImage
{
    ChessboardImage(const cv::Mat &, const Chessboard &, const std::filesystem::path & = "");
    cv::Mat image;
    std::filesystem::path filename;
    Pose<double> Tnc;                                               // Extrinsic camera parameters
    std::vector<cv::Point2f> corners;                       // Chessboard corners in image [rQOi]
    bool isFound;
    void drawCorners(const Chessboard &);
    void drawBox(const Chessboard &, const Camera &);
    void recoverPose(const Chessboard &, const Camera &);
};

struct ChessboardData
{
    explicit ChessboardData(const std::filesystem::path &); // Load from config file

    Chessboard chessboard;
    std::vector<ChessboardImage> chessboardImages;

    void drawCorners();
    void drawBoxes(const Camera &);
    void recoverPoses(const Camera &);
};

namespace Eigen {
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera from chessboard data
    void printCalibration() const;

    template <typename Scalar> Pose<Scalar> cameraToBody(const Pose<Scalar> & Tnc) const { return Tnc*Tbc.inverse(); } // Tnb = Tnc*Tcb
    template <typename Scalar> Pose<Scalar> bodyToCamera(const Pose<Scalar> & Tnb) const { return Tnb*Tbc; } // Tnc = Tnb*Tbc
    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose<double> &) const;
    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size

    Pose<double> Tbc;                                       // Relative pose of camera in body coordinates (Rbc, rCBb)

private:
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
};

#include <cmath>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    assert(isRationalModel && isThinPrismModel);

    // Extract camera matrix parameters
    Scalar fx = Scalar(cameraMatrix.at<double>(0, 0));
    Scalar fy = Scalar(cameraMatrix.at<double>(1, 1));
    Scalar cx = Scalar(cameraMatrix.at<double>(0, 2));
    Scalar cy = Scalar(cameraMatrix.at<double>(1, 2));
    
    // Extract coordinates
    Scalar x = rPCc(0);
    Scalar y = rPCc(1);
    Scalar z = rPCc(2);
    
    // Compute normalized coordinates u, v (equation 2)
    Scalar u = x / z;
    Scalar v = y / z;
    Scalar r2 = u * u + v * v;

    // Extract distortion coefficients
    Scalar k1 = Scalar(distCoeffs.at<double>(0));
    Scalar k2 = Scalar(distCoeffs.at<double>(1));
    Scalar p1 = Scalar(distCoeffs.at<double>(2));
    Scalar p2 = Scalar(distCoeffs.at<double>(3));
    Scalar k3 = Scalar(distCoeffs.at<double>(4));
    Scalar k4 = Scalar(distCoeffs.at<double>(5));
    Scalar k5 = Scalar(distCoeffs.at<double>(6));
    Scalar k6 = Scalar(distCoeffs.at<double>(7));
    Scalar s1 = Scalar(distCoeffs.at<double>(8));
    Scalar s2 = Scalar(distCoeffs.at<double>(9));
    Scalar s3 = Scalar(distCoeffs.at<double>(10));
    Scalar s4 = Scalar(distCoeffs.at<double>(11));

    // Compute rational distortion model (equations 4, 5, 3)
    Scalar r4 = r2 * r2;
    Scalar r6 = r4 * r2;
    
    Scalar alpha = k1 * r2 + k2 * r4 + k3 * r6;
    Scalar beta = k4 * r2 + k5 * r4 + k6 * r6;
    Scalar c = (Scalar(1) + alpha) / (Scalar(1) + beta);

    // Apply distortion model (equation 1)
    // Radial distortion
    Scalar u_radial = c * u;
    Scalar v_radial = c * v;
    
    // Decentering distortion
    Scalar u_decentering = Scalar(2) * p1 * u * v + p2 * (r2 + Scalar(2) * u * u);
    Scalar v_decentering = p1 * (r2 + Scalar(2) * v * v) + Scalar(2) * p2 * u * v;
    
    // Thin prism distortion
    Scalar u_prism = s1 * r2 + s2 * r4;
    Scalar v_prism = s3 * r2 + s4 * r4;
    
    // Total distorted coordinates
    Scalar u_prime = u_radial + u_decentering + u_prism;
    Scalar v_prime = v_radial + v_decentering + v_prism;
    
    // Convert to pixel coordinates (equation 6)
    Eigen::Vector2<Scalar> rQOi;
    rQOi(0) = fx * u_prime + cx;
    rQOi(1) = fy * v_prime + cy;
    
    return rQOi;
}

#endif

