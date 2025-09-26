#include <filesystem>
#include "calibrate.h"
#include "Camera.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // TODO
    // - Read XML at configPath
    // - Parse XML and extract relevant frames from source video containing the chessboard
    // - Perform camera calibration
    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // - Visualise the camera calibration results



    // Use Lab 8's implementation
    ChessboardData chessboardData(configPath);
    Camera camera;
    camera.calibrate(chessboardData);
    
    // Write results to camera.xml (as assignment expects)
    std::filesystem::path outputPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs(outputPath.string(), cv::FileStorage::WRITE);
    fs << "camera" << camera;
    fs.release();
    
    std::cout << "Camera calibration saved to: " << outputPath.string() << std::endl;

    // SUDO-CODE - Thanks to claude:



                            // FUNCTION calibrateCamera(configPath):
                            // // Step 1: Read configuration
                            // READ XML file at configPath
                            // EXTRACT calibration video path from XML
                            // EXTRACT chessboard dimensions (rows, cols) from XML
                            
                            // // Step 2: Process video frames
                            // OPEN video file using cv::VideoCapture
                            // CREATE empty lists: imagePoints, objectPoints
                            
                            // FOR each frame in video:
                            //     CONVERT frame to grayscale
                            //     IF findChessboardCorners(frame, boardSize, corners) succeeds:
                            //         ADD corners to imagePoints
                            //         ADD 3D world coordinates to objectPoints
                            //         OPTIONALLY: draw corners and display frame
                            
                            // // Step 3: Perform calibration
                            // IF sufficient frames found (>= 10):
                            //     CALL cv::calibrateCamera(objectPoints, imagePoints, imageSize, ...)
                            //     GET camera matrix, distortion coefficients, reprojection error
                            // ELSE:
                            //     PRINT error: insufficient calibration frames
                            //     RETURN
                            
                            // // Step 4: Save results
                            // CREATE camera.xml in same directory as configPath
                            // WRITE camera matrix and distortion coefficients to XML
                            
                            // // Step 5: Display results
                            // PRINT camera parameters to console
                            // PRINT reprojection error
                            // SHOW sample undistorted image for validation

}
