#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <format>
#include <vector>
#include <filesystem>
#include <regex>
#include <print>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "to_string.hpp"
#include "rotation.hpp"
#include "Pose.hpp"
#include "Camera.h"

void Chessboard::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode & node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    // Detect chessboard corners
    isFound = cv::findChessboardCorners(image, chessboard.boardSize, corners);
    
    // Optional: Refine corners with subpixel accuracy
    if (isFound) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    }
}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    const double boxHeight = -0.23;
    const int cornersX = chessboard.boardSize.width - 1;
    const int cornersY = chessboard.boardSize.height - 1;
    const double squareSize = chessboard.squareSize;
    
    Pose Tnb = camera.cameraToBody(Tnc);
    
    // 8 vertices
    std::vector<cv::Vec3d> vertices = {
        {0, 0, 0}, {cornersX * squareSize, 0, 0},
        {cornersX * squareSize, cornersY * squareSize, 0}, {0, cornersY * squareSize, 0},
        {0, 0, boxHeight}, {cornersX * squareSize, 0, boxHeight},
        {cornersX * squareSize, cornersY * squareSize, boxHeight}, {0, cornersY * squareSize, boxHeight}
    };
    
    struct Edge {
        int start, end;
        cv::Scalar color;
    };
    
    std::vector<Edge> edges = {
        // X-axis (Red)
        {0,1, {0,0,255}}, {2,3, {0,0,255}}, {4,5, {0,0,255}}, {6,7, {0,0,255}},
        // Y-axis (Green)
        {1,2, {0,255,0}}, {3,0, {0,255,0}}, {5,6, {0,255,0}}, {7,4, {0,255,0}},
        // Z-axis (Blue)
        {0,4, {255,0,0}}, {1,5, {255,0,0}}, {2,6, {255,0,0}}, {3,7, {255,0,0}}
    };
    
    for (auto& edge : edges) {
        cv::Vec3d p1 = vertices[edge.start];
        cv::Vec3d p2 = vertices[edge.end];
        
            int numSegments = 1000;  // Adjust this value (was 10)
            for (int i = 0; i < numSegments; i++) {
                double t1 = i / (double)numSegments;
                double t2 = (i + 1) / (double)numSegments;
            cv::Vec3d world1 = p1 + t1 * (p2 - p1);
            cv::Vec3d world2 = p1 + t2 * (p2 - p1);
            
            // Check FOV for both endpoints
            if (!camera.isWorldWithinFOV(world1, Tnb) || 
                !camera.isWorldWithinFOV(world2, Tnb)) {
                continue;  // Skip this segment
            }
            
            cv::Vec2d pixel1 = camera.worldToPixel(world1, Tnb);
            cv::Vec2d pixel2 = camera.worldToPixel(world2, Tnb);
            
            cv::line(image, cv::Point(pixel1), cv::Point(pixel2), edge.color, 2);
        }
    }
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    Pose<double> Tcn(Thetacn, rNCc);
    Tnc = Tcn.inverse();
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    // Ensure the config file exists
    if (!std::filesystem::exists(configPath))
    {
        throw std::runtime_error("Config file does not exist: " + configPath.string());
    }

    // Open the config file
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }

    // Read chessboard configuration
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::println("Chessboard: {}", to_string(chessboard));

    // Read file pattern for chessboard images
    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();

    // Create regex object from pattern
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Get the directory containing the config file
    std::filesystem::path root = configPath.parent_path();
    std::println("Scanning directory {} for file pattern \"{}\"", root.string(), pattern);

    // Populate chessboard images from regex
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        // Iterate through all files in the directory and its subdirectories
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                // Check if the file matches the regex pattern
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::print("Loading {}...", p.path().filename().string());

                    // Try to load the file as an image
                    cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

                    bool isImage = !image.empty();
                    if (isImage)
                    {
                        // If it's an image, detect chessboard
                        std::print(" done, detecting chessboard...");
                        ChessboardImage ci(image, chessboard, p.path().filename());
                        std::println("{}", ci.isFound ? " found" : " not found");
                        if (ci.isFound)
                        {
                            chessboardImages.push_back(ci);
                        }
                    }
                    else
                    {
                        // If it's not an image, try to load it as a video
                        cv::VideoCapture cap(p.path().string());
                        bool isVideo = cap.isOpened();
                        if (isVideo)
                        {
                            // Get number of video frames
                            int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
                            std::println(" done, found {} frames", nFrames);

                            // Select frames to process (every 30th frame, max 30 images)
                            int frameInterval = 10;
                            int maxImages = 30;
                            int imagesFound = 0;
                            

                            // Loop through selected frames
                            for (int idxFrame = 0; idxFrame < nFrames && imagesFound < maxImages; idxFrame += frameInterval)
                            {
                                // Read frame
                                std::print("Reading {} frame {}...", p.path().filename().string(), idxFrame);
                                cv::Mat frame;
                                cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
                                cap.read(frame);
                                
                                if (frame.empty())
                                {
                                    std::println(" end of file found");
                                    break;
                                }

                                // Detect chessboard in frame
                                std::print(" done, detecting chessboard...");
                                std::string baseName = p.path().stem().string();
                                std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
                                ChessboardImage ci(frame, chessboard, frameFilename);
                                std::println("{}", ci.isFound ? " found" : " not found");
                                if (ci.isFound)
                                {
                                    chessboardImages.push_back(ci);
                                    imagesFound++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners);
    }
    assert(!rQOi_all.empty());

    imageSize = chessboardData.chessboardImages[0].image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::print("Calibrating camera...");

    // Calibrate camera from detected chessboard corners
    std::vector<std::vector<cv::Point3f>> objectPoints(rQOi_all.size(), rPNn_all);
    rms = cv::calibrateCamera(objectPoints, rQOi_all, imageSize, 
                            cameraMatrix, distCoeffs, 
                            Thetacn_all, rNCc_all, flags);

    std::println(" done");
    
    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Write extrinsic camera parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        // Set the camera orientation and position (extrinsic camera parameters)
        Pose<double> & Tnc = chessboardData.chessboardImages[k].Tnc;
        
        // Convert from camera coordinates to world coordinates
        Pose<double> Tcn(Thetacn_all[k], rNCc_all[k]);
        Tnc = Tcn.inverse();
    }
    
    printCalibration();
    std::println("{:>30} {}", "RMS reprojection error:", rms);

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::println("\nCalibration data:");
    std::println("{:>30} {}", "Bit flags:", bitflag.to_string());
    std::println("{:>30}\n{}", "cameraMatrix:", to_string(cameraMatrix));
    std::println("{:>30}\n{}", "distCoeffs:", to_string(distCoeffs.t()));
    std::println("{:>30} (fx, fy) = ({}, {})", "Focal lengths:",
              cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));       
    std::println("{:>30} (cx, cy) = ({}, {})", "Principal point:",
              cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));     
    std::println("{:>30} {} deg", "Field of view (horizontal):", 180.0/CV_PI*hFOV);
    std::println("{:>30} {} deg", "Field of view (vertical):", 180.0/CV_PI*vFOV);
    std::println("{:>30} {} deg", "Field of view (diagonal):", 180.0/CV_PI*dFOV);
}

void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // Get unit vectors for corners and edge centers
    cv::Vec3d leftCenter = pixelToVector(cv::Vec2d(0, imageSize.height/2.0));
    cv::Vec3d rightCenter = pixelToVector(cv::Vec2d(imageSize.width-1, imageSize.height/2.0));
    cv::Vec3d topCenter = pixelToVector(cv::Vec2d(imageSize.width/2.0, 0));
    cv::Vec3d bottomCenter = pixelToVector(cv::Vec2d(imageSize.width/2.0, imageSize.height-1));
    cv::Vec3d topLeft = pixelToVector(cv::Vec2d(0, 0));
    cv::Vec3d bottomRight = pixelToVector(cv::Vec2d(imageSize.width-1, imageSize.height-1));
    
    // Calculate angles using dot product
    hFOV = std::acos(leftCenter.dot(rightCenter));
    vFOV = std::acos(topCenter.dot(bottomCenter));
    dFOV = std::acos(topLeft.dot(bottomRight));
}

// cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
// {
//     // Tnb = Tnc*Tcb
//     return Tnc*Tbc.inverse();
// }

// Pose<double> Camera::bodyToCamera(const Pose<double> & Tnb) const
// {
//     // Tnc = Tnb*Tbc
//     return Tnb*Tbc;
// }

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    // Camera pose Tnc (i.e., Rnc, rCNn)
    Pose<double> Tnc = bodyToCamera(Tnb); // Tnb*Tbc

    // Compute the unit vector uPCc from the world position rPNn and camera pose Tnc
    cv::Vec3d rPCc = Tnc.inverse() * rPNn;  // Transform to camera coordinates
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc); // Normalize to unit vector
  
    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Vec2d rQOi;
    std::vector<cv::Point3d> objectPoints = {rPCc};
    std::vector<cv::Point2d> imagePoints;
    
    // Use identity rotation and zero translation since already in camera coordinates
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
    
    rQOi = imagePoints[0];
    return rQOi;
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc, Eigen::Matrix23d & J) const
{
    Eigen::Vector2d rQOi;
    // TODO: Lab 8 (optional)
    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    cv::Vec3d uPCc;
    
    std::vector<cv::Point2d> distorted = {rQOi};
    std::vector<cv::Point2d> undistorted;
    
    cv::undistortPoints(distorted, undistorted, cameraMatrix, distCoeffs);
    
    // undistortPoints gives normalized coordinates, create unit vector
    uPCc[0] = undistorted[0].x;
    uPCc[1] = undistorted[0].y;
    uPCc[2] = 1.0;
    
    // Normalize to ensure unit vector
    uPCc = uPCc / cv::norm(uPCc);
    
    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    if (rPCc[2] <= 0) return false;  // Behind camera
    
    // Calculate angles from optical axis
    double angleX = std::atan2(std::abs(rPCc[0]), rPCc[2]);  // Horizontal angle
    double angleY = std::atan2(std::abs(rPCc[1]), rPCc[2]);  // Vertical angle
    
    // Check against half FOVs
    return angleX < (hFOV / 2.0) && angleY < (vFOV / 2.0);
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}

