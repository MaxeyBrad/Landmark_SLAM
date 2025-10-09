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
}
