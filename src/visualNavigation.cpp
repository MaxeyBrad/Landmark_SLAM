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
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = 2*cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

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
        initialMean.segment<3>(9) << -M_PI/2.0, 0.0, 0.0;

        // Initial covariance (tweak if you want looser priors on pose)
        Eigen::MatrixXd initialCovariance = Eigen::MatrixXd::Identity(12, 12);

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
            
            // Draw pose axes for each detected tag
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, 0.166, camera.cameraMatrix, camera.distCoeffs, rvecs, tvecs);
            
            for (size_t i = 0; i < ids.size(); ++i) {
                cv::drawFrameAxes(imgProcessed, camera.cameraMatrix, camera.distCoeffs, rvecs[i], tvecs[i], 0.1);
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

        // Write output frame
        if (doExport)
        {
            bufferedVideoWriter.write(combinedFrame);
        }


    }

    if (doExport)
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
