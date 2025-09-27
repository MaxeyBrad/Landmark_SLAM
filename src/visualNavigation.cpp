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
    // Initialize Plot
    // Plot plot(camera);

    // Initialize ArUco detector  
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary);

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
            cv::aruco::drawDetectedMarkers(imgProcessed, corners, ids);
}

        // Update state

        // Update plot (you'll need minimal SLAM system/measurement for this)
        // plot.setData(slamSystem, measurement);
        // plot.render();

        // Get rendered frame
        //cv::Mat imgout = plot.getFrame();
        // Write output frame 
        // if (doExport)
        // {
        //     // cv::Mat imgout /* = plot.getFrame()*/; // TODO: Uncomment this to get the frame image
        //     // bufferedVideoWriter.write(imgout);
        //     bufferedVideoWriter.write(imgProcessed);
        // }
        // Replace the Plot approach with simple split-screen
        cv::Mat leftFrame = imgProcessed;  // Video with ArUco markers
        cv::Mat rightFrame = cv::Mat::zeros(leftFrame.size(), leftFrame.type());  // Blank right side

        // Create split-screen output
        cv::Mat combinedFrame;
        cv::hconcat(leftFrame, rightFrame, combinedFrame);
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

    cv::destroyAllWindows();
}
