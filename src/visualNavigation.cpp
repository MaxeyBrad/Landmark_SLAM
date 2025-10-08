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
        std::cout << "Loaded camera calibration from: " << cameraPath.string() << std::endl;
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

    // // Initialize SLAM system for ArUco scenario
    // SystemSLAMPoseLandmarks * slamSystem = nullptr;
    // if (scenario == 1) {  // ArUco scenario
    //     // Initialize with minimal state (body velocities + pose)
    //     Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(12);  // [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3)]
    //     Eigen::MatrixXd initialCovariance = 1 * Eigen::MatrixXd::Identity(12, 12);  // Small initial uncertainty
    //     GaussianInfo<double> initialDensity = GaussianInfo<double>::fromSqrtMoment(initialMean, initialCovariance);
    //     slamSystem = new SystemSLAMPoseLandmarks(initialDensity);
    //     std::cout << "Initialized SLAM system for ArUco markers" << std::endl;
    // }

    // Initialize SLAM system for ArUco scenario
    SystemSLAMPoseLandmarks* slamSystem = nullptr;
    if (scenario == 1) {  // ArUco scenario
        // State layout: [ v_B^b(3), ω_B^b(3), r_B^n = (N,E,D)(3), Θ_nb = (roll, pitch, yaw)[ZYX] (3) ]
        Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(12);

        // Position: 1.6 m UP in NED  => D = -1.6
        initialMean.segment<3>(6) << 0.0, 0.0, -1.6;

        // Orientation: facing North, level (roll = 0, pitch = 0, yaw = 0) in radians
        // initialMean.segment<3>(9) << -M_PI/2.0, M_PI, 0.0;
        // initialMean.segment<3>(9) << M_PI/2, 0.0, M_PI/2;
        initialMean.segment<3>(9) << 0.0, 0.0, 0.0;

        // // Initial covariance (tweak if you want looser priors on pose)
        // Eigen::MatrixXd initialCovariance = Eigen::MatrixXd::Identity(12, 12);

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
        std::cout << "Initialised SLAM (Aruco): r_B^n=[0,0,-1.6], Θ_nb=[0,0,0]" << std::endl;
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

        // Debug: Always report detection status
        std::cout << "Frame " << cap.get(cv::CAP_PROP_POS_FRAMES) 
                 << ": ArUco detection found " << ids.size() << " markers" << std::endl;

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
                        
        // Draw pose axes for each detected tag using fixed pose estimation
        std::vector<cv::Vec3d> rvecs, tvecs;
        static std::map<int, Eigen::VectorXd> previousPoses; // Store previous poses for temporal consistency

        for (size_t i = 0; i < ids.size(); ++i) {
            int tagId = ids[i];
            
            // Get previous pose for temporal consistency (if available)
            Eigen::VectorXd* prevPose = nullptr;
            auto it = previousPoses.find(tagId);
            if (it != previousPoses.end()) {
                prevPose = &(it->second);
            }
            
            // Use your fixed pose estimation function
            Eigen::VectorXd pose = MeasurementSLAMAruco::estimateArucoLandmarkPose(corners[i], camera, prevPose);
            
            if (pose.norm() > 0.0) {  // Valid pose estimated
                // Store for next frame's temporal consistency
                previousPoses[tagId] = pose;
                
                // Convert back to camera frame for visualization (OpenCV expects camera frame)
                Eigen::Matrix3d R_bc;
                R_bc << 0, 0, 1,   // Body X (North) = Camera Z (forward)
                        1, 0, 0,   // Body Y (East) = Camera X (right)
                        0, 1, 0;   // Body Z (Down) = Camera Y (down)
                
                // Transform from body frame back to camera frame
                Eigen::Vector3d pos_body = pose.segment<3>(0);
                Eigen::Vector3d rpy_body = pose.segment<3>(3);
                
                Eigen::Vector3d pos_camera = R_bc.transpose() * pos_body;
                Eigen::Matrix3d R_body = rpy2rot(rpy_body);
                Eigen::Matrix3d R_camera = R_bc.transpose() * R_body;
                
                // Convert rotation matrix to Rodrigues vector (what OpenCV expects)
                cv::Mat R_cv;
                cv::eigen2cv(R_camera, R_cv);
                cv::Mat rvec_mat;
                cv::Rodrigues(R_cv, rvec_mat);
                cv::Vec3d rvec(rvec_mat.at<double>(0), rvec_mat.at<double>(1), rvec_mat.at<double>(2));
                
                cv::Vec3d tvec(pos_camera(0), pos_camera(1), pos_camera(2));
                
                // Draw stable axes
                cv::drawFrameAxes(imgProcessed, camera.cameraMatrix, camera.distCoeffs, rvec, tvec, 0.1);
            }
        }
            
            // SLAM processing for ArUco scenario
            std::cout << "Checking SLAM conditions: scenario=" << scenario << ", slamSystem=" << (slamSystem ? "valid" : "null") << std::endl;
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
                
                std::cout << "FOV check: " << visibleLandmarks.size() 
                          << " landmarks visible out of " << slamSystem->numberLandmarks() << " total" << std::endl;
                
                // Perform data association
                const std::vector<int> & associations = arucoMeasurement.associate(*slamSystem, visibleLandmarks);
                
                // Process the measurement through SLAM (this will optimize the state and initialize new landmarks)
                arucoMeasurement.process(*slamSystem);
                
                // Test corner predictions for existing landmarks
                if (slamSystem->numberLandmarks() > 0) {
                    Eigen::VectorXd currentState = slamSystem->density.mean();
                    
                    // Debug: Show camera pose
                    std::cout << "Camera pose: pos[" 
                              << currentState(6) << "," << currentState(7) << "," << currentState(8) 
                              << "] rot[" 
                              << currentState(9) << "," << currentState(10) << "," << currentState(11) 
                              << "]" << std::endl;
                    
                    std::cout << "Testing corner predictions for " << slamSystem->numberLandmarks() << " landmarks:" << std::endl;
                    
                    for (std::size_t landmarkIdx = 0; landmarkIdx < slamSystem->numberLandmarks(); ++landmarkIdx) {
                        Eigen::MatrixXd J;  // Jacobian (not used here)
                        Eigen::Matrix<double, 8, 1> predictedCorners = arucoMeasurement.predictArucoCorners(currentState, J, *slamSystem, landmarkIdx);
                        
                        std::cout << "  Landmark " << landmarkIdx << " predicted corners: [";
                        for (int c = 0; c < 4; ++c) {
                            std::cout << "(" << predictedCorners(2*c) << "," << predictedCorners(2*c+1) << ")";
                            if (c < 3) std::cout << " ";
                        }
                        std::cout << "]" << std::endl;
                    }
                }
                
                // Report detection results
                std::cout << "Frame " << cap.get(cv::CAP_PROP_POS_FRAMES) 
                         << ": Detected " << ids.size() << " ArUco markers, " 
                         << slamSystem->numberLandmarks() << " landmarks tracked" << std::endl;
                
                // Draw confidence ellipses for landmarks in field of view (FOV)
                // This includes both detected landmarks and potentially occluded ones
                if (visibleLandmarks.size() > 0) {
                    std::cout << "Drawing ellipses for " << visibleLandmarks.size() 
                              << " landmarks in FOV (includes detected + potentially occluded)" << std::endl;
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
        if (!imgout.empty()) {
            std::cout << "Plot frame size: " << imgout.cols << "x" << imgout.rows << std::endl;
        } else {
            std::cout << "Warning: Plot getFrame() returned empty image" << std::endl;
        }
        // Write output frame 
        // if (doExport)
        // {
        //     // cv::Mat imgout /* = plot.getFrame()*/; // TODO: Uncomment this to get the frame image
        //     // bufferedVideoWriter.write(imgout);
        //     bufferedVideoWriter.write(imgProcessed);
        // }
        // Use Plot system output if available, otherwise fallback to manual split-screen
        cv::Mat combinedFrame;
        if (!imgout.empty()) {
            combinedFrame = imgout;  // Professional VTK-based split-screen
            std::cout << "Using Plot system output: " << combinedFrame.cols << "x" << combinedFrame.rows << std::endl;
        } else {
            // Fallback to manual split-screen
            cv::Mat leftFrame = imgProcessed;  // Video with ArUco markers
            cv::Mat rightFrame = cv::Mat::zeros(leftFrame.size(), leftFrame.type());  // Blank right side
            cv::hconcat(leftFrame, rightFrame, combinedFrame);
            std::cout << "Using manual split-screen: " << combinedFrame.cols << "x" << combinedFrame.rows << std::endl;
        }
        
        // Handle interactive modes
        if (interactive == 2) {
            // Interactive mode 2: Pause on every frame (VTK window key press)
            std::cout << "Frame " << cap.get(cv::CAP_PROP_POS_FRAMES) << " - Press any key in VTK window to continue..." << std::endl;
            
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
        std::cout << "SLAM system cleanup complete" << std::endl;
    }

}
