#include <filesystem>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/aruco.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "Plot.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMAruco.h"
#include "GaussianInfo.hpp"
#include "visualNavigation.h"
#include <opencv2/highgui.hpp>
#include "rotation.hpp"
#include <opencv2/imgproc.hpp>

// Function to read a single character without requiring Enter
char getChar() {
    struct termios oldt, newt;
    char ch;
    
    // Get current terminal settings
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    
    // Disable canonical mode and echo
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    
    // Read single character
    ch = getchar();
    
    // Restore original terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    
    return ch;
}

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    // Load camera calibration
    Camera camera;
    if (std::filesystem::exists(cameraPath)) {
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        fs["camera"] >> camera;
        fs.release();
    } else {
        std::cout << "Warning: Camera calibration not found at " << cameraPath.string() << std::endl;
        return;
    }
    // Display loaded calibration data

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    bool videoWriterInitialized = false;

    // Visual navigation

    // Initialisation
    // Initialize Plot for 3D visualization
    Plot plot(camera);
    
    // Enable interactive VTK controls
    plot.enableInteraction();
    
    // Set interactive mode based on parameter
    if (interactive == 2) {
        plot.setInteractiveMode(true);
    }

    // Initialize ArUco detector  
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary);

    // Initialize SLAM system for ArUco scenario
    SystemSLAMPoseLandmarks* slamSystem = nullptr;
    if (scenario == 1) {  // ArUco scenario
        // State layout: [ v_B^b(3), ω_B^b(3), r_B^n = (N,E,D)(3), Θ_nb = (roll, pitch, yaw)[ZYX] (3) ]
        Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(12);

        // Position: 1.6 m UP in NED  => D = -1.6
        initialMean.segment<3>(6) << 0.0, 0.0, -1.6;

        // Orientation: facing North, level (roll = 0, pitch = 0, yaw = 0) in radians
        initialMean.segment<3>(9) << 0.0, 0.0, 0.0;

        Eigen::MatrixXd initialCovariance = Eigen::MatrixXd::Identity(12, 12);

        // Velocity uncertainty (totally unknown)
        initialCovariance.block<3,3>(0,0) *= 1.0;      // Linear velocity: 1 m/s std
        initialCovariance.block<3,3>(3,3) *= 0.5;      // Angular velocity: 0.5 rad/s std

        // Position uncertainty (you KNOW you start at [0,0,-1.6])
        initialCovariance.block<3,3>(6,6) *= 0.01;     // Position: 1cm std (very confident!)

        // Orientation uncertainty (you KNOW you start level, facing north)
        initialCovariance.block<3,3>(9,9) *= 0.01;     // Orientation: ~0.5° std (very confident!)

        GaussianInfo<double> initialDensity =
            GaussianInfo<double>::fromSqrtMoment(initialMean, initialCovariance);

        slamSystem = new SystemSLAMPoseLandmarks(initialDensity);
    }


    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        // Process frame
        // Process frame
        cv::Mat imgProcessed = imgin.clone();

        // Detect ArUco markers
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector.detectMarkers(imgin, corners, ids);

        // Draw detected markers
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(imgProcessed, corners, ids);  // Green outline and ID

            // Draw BIGGER tag IDs (positioned ABOVE the tag)
            for (size_t i = 0; i < ids.size(); ++i) {
                // Calculate marker center
                cv::Point2f center(0, 0);
                for (const auto& corner : corners[i]) {
                    center.x += corner.x;
                    center.y += corner.y;
                }
                center.x /= 4.0f;
                center.y /= 4.0f;
                
                // Draw large tag ID
                std::string tagText = "T:" + std::to_string(ids[i]);
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 1.2;  // Bigger font
                int thickness = 3;
                cv::Scalar textColor(0, 255, 255);  // Yellow text
                cv::Scalar backgroundColor(0, 0, 0);  // Black background
                
                // Get text size for background rectangle
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(tagText, fontFace, fontScale, thickness, &baseline);
                
                // Position text ABOVE the tag (subtract pixels to move up)
                int textOffset = -40;  // Move 40 pixels above the tag center
                
                // Draw black background rectangle (above the tag)
                cv::Point bgTopLeft(center.x - textSize.width/2 - 5, center.y + textOffset - textSize.height - 5);
                cv::Point bgBottomRight(center.x + textSize.width/2 + 5, center.y + textOffset + 10);
                cv::rectangle(imgProcessed, bgTopLeft, bgBottomRight, backgroundColor, -1);
                
                // Draw tag ID text (above the tag)
                cv::Point textPos(center.x - textSize.width/2, center.y + textOffset);
                cv::putText(imgProcessed, tagText, textPos, fontFace, fontScale, textColor, thickness);
            }

        if (slamSystem != nullptr && slamSystem->numberLandmarks() > 0) {
            
            // Get current camera pose from SLAM state
            Eigen::VectorXd currentState = slamSystem->density.mean();
            Eigen::Vector3d r_B_n = currentState.segment<3>(6);   // Body position in world
            Eigen::Vector3d theta_bn = currentState.segment<3>(9); // Body orientation
            Eigen::Matrix3d R_nb = rpy2rot(theta_bn);             // World to body rotation
            
            // Define body-to-camera rotation
            Eigen::Matrix3d R_bc;
            R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
                    1, 0, 0,   // Body Y (East) = Camera X (right)
                    0, 1, 0;   // Body Z (Down) = Camera Y (down)
            
            // Get landmark tag IDs for association
            const auto& landmarkTagIds = MeasurementSLAMAruco::getLandmarkTagIds();
            
            // Draw axes for each landmark in the SLAM map
            for (std::size_t landmarkIdx = 0; landmarkIdx < slamSystem->numberLandmarks(); ++landmarkIdx) {
                // Get landmark pose from SLAM state (in world frame)
                std::size_t stateIdx = slamSystem->landmarkPositionIndex(landmarkIdx);
                Eigen::Vector3d r_L_n = currentState.segment<3>(stateIdx);      // Landmark position (world)
                Eigen::Vector3d theta_Ln = currentState.segment<3>(stateIdx + 3); // Landmark orientation (world)
                Eigen::Matrix3d R_nL = rpy2rot(theta_Ln);                       // World to landmark rotation
                
                // Transform landmark pose from world frame to camera frame
                // Step 1: World → Body
                Eigen::Vector3d r_L_b = R_nb.transpose() * (r_L_n - r_B_n);
                Eigen::Matrix3d R_bL = R_nb.transpose() * R_nL;
                
                // Step 2: Body → Camera
                Eigen::Vector3d r_L_c = R_bc.transpose() * r_L_b;
                Eigen::Matrix3d R_cL = R_bc.transpose() * R_bL;
                
                // Check if landmark is in front of camera (positive Z)
                if (r_L_c(2) <= 0.0) {
                    continue;
                }

                // Check if landmark is within camera field of view
                cv::Vec3d landmarkPosCV(r_L_n(0), r_L_n(1), r_L_n(2));
                Pose<double> Tnb(R_nb, r_B_n);
                if (!camera.isWorldWithinFOV(landmarkPosCV, Tnb)) {
                    continue;
                }
                
                // Convert rotation matrix to Rodrigues vector for OpenCV
                cv::Mat R_cv;
                cv::eigen2cv(R_cL, R_cv);
                cv::Mat rvec_mat;
                cv::Rodrigues(R_cv, rvec_mat);
                cv::Vec3d rvec(rvec_mat.at<double>(0), rvec_mat.at<double>(1), rvec_mat.at<double>(2));
                cv::Vec3d tvec(r_L_c(0), r_L_c(1), r_L_c(2));
                
                // Draw axes for this landmark (0.1m = 100mm length)
                cv::drawFrameAxes(imgProcessed, camera.cameraMatrix, camera.distCoeffs, 
                                rvec, tvec, 0.1);
                
                // Draw landmark ID label near the axes
                if (landmarkIdx < landmarkTagIds.size()) {
                    // Project landmark center to image
                    std::vector<cv::Point3d> landmarkCenter = {cv::Point3d(r_L_c(0), r_L_c(1), r_L_c(2))};
                    std::vector<cv::Point2d> imagePoint;
                    cv::Mat rvec_zero = cv::Mat::zeros(3, 1, CV_64F);
                    cv::Mat tvec_zero = cv::Mat::zeros(3, 1, CV_64F);
                    cv::projectPoints(landmarkCenter, rvec_zero, tvec_zero, 
                                    camera.cameraMatrix, camera.distCoeffs, imagePoint);
                    
                    if (!imagePoint.empty() && 
                        imagePoint[0].x >= 0 && imagePoint[0].x < imgProcessed.cols &&
                        imagePoint[0].y >= 0 && imagePoint[0].y < imgProcessed.rows) {
                        
                        int tagId = landmarkTagIds[landmarkIdx];
                        std::string labelText = "L" + std::to_string(landmarkIdx) + 
                                            " (T" + std::to_string(tagId) + ")";
                        
                        // Draw label with background
                        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                        double fontScale = 0.6;
                        int thickness = 2;
                        int baseline = 0;
                        cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);
                    }
                }
            }
        }

            // SLAM processing for ArUco scenario
            if (scenario == 1 && slamSystem != nullptr) {
                // Create ArUco measurement
                double timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;  // Convert to seconds
                
                // IMPORTANT: Predict system state to current timestamp using motion model
                slamSystem->predict(timestamp);
                
                MeasurementSLAMAruco arucoMeasurement(timestamp, camera, ids, corners);
                
                // Get current landmarks that should be visible (within camera FOV)
                std::vector<std::size_t> visibleLandmarks;
                
                // Get current camera pose for FOV checking
                Eigen::VectorXd currentState = slamSystem->density.mean();
                Eigen::Vector3d rBNn = currentState.segment<3>(6);       // Body position
                Eigen::Vector3d thetaBN = currentState.segment<3>(9);    // Body orientation
                
                // Convert to OpenCV pose format for Camera::isWorldWithinFOV()
                Eigen::Matrix3d Rnb = rpy2rot(thetaBN);
                Pose<double> Tnb(Rnb, rBNn);
                
                // Check each landmark to see if it's within camera field of view
                for (std::size_t i = 0; i < slamSystem->numberLandmarks(); ++i) {
                    // Get landmark position from state
                    std::size_t landmarkIdx = slamSystem->landmarkPositionIndex(i);
                    Eigen::Vector3d landmarkPos = currentState.segment<3>(landmarkIdx);
                    cv::Vec3d landmarkPosCV(landmarkPos(0), landmarkPos(1), landmarkPos(2));
                    
                    // Check if landmark is within camera field of view
                    if (camera.isWorldWithinFOV(landmarkPosCV, Tnb)) {
                        visibleLandmarks.push_back(i);
                    }
                }

                // Perform data association
                const std::vector<int> & associations = arucoMeasurement.associate(*slamSystem, visibleLandmarks);

                // Process the measurement through SLAM (this will optimize the state and initialize new landmarks)
                arucoMeasurement.process(*slamSystem);

                // Draw confidence ellipses for landmarks in field of view (FOV)
                // This includes both detected landmarks and potentially occluded ones
                if (visibleLandmarks.size() > 0) {
                    arucoMeasurement.drawConfidenceEllipses(imgProcessed, *slamSystem, visibleLandmarks, 3.0);
                }
                
                // Update 3D plot with SLAM data (following Lab 8 pattern)
                slamSystem->view() = imgProcessed.clone();  // Set image for left pane
                plot.setData(*slamSystem, arucoMeasurement);
                plot.render();
                
                // Process VTK interactive events (non-blocking)
                plot.processEvents();
            }
        }

        // Update state

        // Get rendered frame from Plot system (includes 3D visualization)
        cv::Mat imgout = plot.getFrame();

        // Use Plot system output if available, otherwise fallback to manual split-screen
        cv::Mat combinedFrame;
        if (!imgout.empty()) {
            combinedFrame = imgout;
        } else {
            // Fallback to manual split-screen
            cv::Mat leftFrame = imgProcessed;
            cv::Mat rightFrame = cv::Mat::zeros(leftFrame.size(), leftFrame.type());
            cv::hconcat(leftFrame, rightFrame, combinedFrame);
        }
        
        // Handle interactive modes
        if (interactive == 2) {
            // Wait for key press in VTK window
            plot.resetAdvanceFrame();
            while (!plot.shouldAdvanceFrame()) {
                plot.processEvents();
                // Small delay to prevent busy waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Initialize video writer on first frame (when we know the actual frame size)
        if (doExport && !videoWriterInitialized && !combinedFrame.empty()) {
            cv::Size frameSize(combinedFrame.cols, combinedFrame.rows);
            double outputFps = fps;
            int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            videoOut.open(outputPath.string(), codec, outputFps, frameSize);
            
            if (!videoOut.isOpened()) {
                std::cerr << "Error: Failed to open video writer!" << std::endl;
                std::cerr << "Output path: " << outputPath.string() << std::endl;
                std::cerr << "Frame size: " << frameSize.width << "x" << frameSize.height << std::endl;
                std::cerr << "FPS: " << outputFps << std::endl;
                doExport = false;
            } else {
                std::cout << "Video writer successfully opened for export:" << std::endl;
                std::cout << "  Output: " << outputPath.string() << std::endl;
                std::cout << "  Size: " << frameSize.width << "x" << frameSize.height << std::endl;
                std::cout << "  FPS: " << outputFps << std::endl;
                bufferedVideoWriter.start(videoOut);
                videoWriterInitialized = true;
            }
        }
        
        // Write output frame
        if (doExport && videoWriterInitialized)
        {
            bufferedVideoWriter.write(combinedFrame);
        }


    }

    if (doExport && videoWriterInitialized)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();

    // Cleanup SLAM system
    if (slamSystem != nullptr) {
        delete slamSystem;
    }

}
