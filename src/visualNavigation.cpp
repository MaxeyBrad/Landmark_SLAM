#include <filesystem>
#include <string>
#include <iostream>
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

    // Initialize ArUco detector  
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary);

    // Initialize SLAM system for ArUco scenario
    SystemSLAMPoseLandmarks * slamSystem = nullptr;
    if (scenario == 1) {  // ArUco scenario
        // Initialize with minimal state (body velocities + pose)
        Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(12);  // [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3)]
        Eigen::MatrixXd initialCovariance = 0.1 * Eigen::MatrixXd::Identity(12, 12);  // Small initial uncertainty
        GaussianInfo<double> initialDensity = GaussianInfo<double>::fromSqrtMoment(initialMean, initialCovariance);
        slamSystem = new SystemSLAMPoseLandmarks(initialDensity);
        std::cout << "Initialized SLAM system for ArUco markers" << std::endl;
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
            cv::aruco::drawDetectedMarkers(imgProcessed, corners, ids);
            
            // SLAM processing for ArUco scenario
            std::cout << "Checking SLAM conditions: scenario=" << scenario << ", slamSystem=" << (slamSystem ? "valid" : "null") << std::endl;
            if (scenario == 1 && slamSystem != nullptr) {
                // Create ArUco measurement
                double timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;  // Convert to seconds
                MeasurementSLAMAruco arucoMeasurement(timestamp, camera, ids, corners);
                
                // Get current landmarks that should be visible
                std::vector<std::size_t> visibleLandmarks;
                for (std::size_t i = 0; i < slamSystem->numberLandmarks(); ++i) {
                    visibleLandmarks.push_back(i);
                }
                
                // Perform data association
                const std::vector<int> & associations = arucoMeasurement.associate(*slamSystem, visibleLandmarks);
                
                // Initialize new landmarks for unassociated detections
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    int tagId = ids[i];
                    
                    // Check if this tag already has a landmark
                    if (MeasurementSLAMAruco::findLandmarkByTagId(tagId) == -1) {
                        // New tag - initialize landmark
                        std::cout << "Initializing new landmark for tag " << tagId << std::endl;
                        MeasurementSLAMAruco::initializeNewLandmark(*slamSystem, tagId, corners[i], camera);
                    }
                }
                
                // Test corner predictions for existing landmarks
                if (slamSystem->numberLandmarks() > 0) {
                    std::cout << "Testing corner predictions for " << slamSystem->numberLandmarks() << " landmarks:" << std::endl;
                    Eigen::VectorXd currentState = slamSystem->density.mean();
                    
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
                
                // Draw confidence ellipses for uncertainty visualization
                if (slamSystem->numberLandmarks() > 0) {
                    std::vector<std::size_t> allLandmarks;
                    for (std::size_t i = 0; i < slamSystem->numberLandmarks(); ++i) {
                        allLandmarks.push_back(i);
                    }
                    arucoMeasurement.drawConfidenceEllipses(imgProcessed, *slamSystem, allLandmarks, 3.0);
                }
                
                // Update 3D plot with SLAM data (following Lab 8 pattern)
                slamSystem->view() = imgProcessed.clone();  // Set image for left pane
                plot.setData(*slamSystem, arucoMeasurement);
                plot.render();
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
        
        cv::namedWindow("Visual Navigation", cv::WINDOW_NORMAL);
        cv::resizeWindow("Visual Navigation", 1200, 400); // Adjust size as needed
        cv::imshow("Visual Navigation", combinedFrame);

        // Display in real-time based on interactive mode
        if (interactive == 2) {
            cv::imshow("Visual Navigation", combinedFrame);
            cv::waitKey(0); // Wait for key press each frame
        } else if (interactive == 0 && !doExport) {
            cv::imshow("Visual Navigation", combinedFrame);
            cv::waitKey(1); // Non-blocking display
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

    cv::destroyAllWindows();
}
