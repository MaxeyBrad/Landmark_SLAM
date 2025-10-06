#ifndef PLOT_H
#define PLOT_H

#include <string>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkAxesActor.h>
#include <vtkCellArray.h>
#include <vtkColor.h>
#include <vtkContourFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageMapper.h>
#include <vtkLine.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPyramid.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCommand.h>

#include "Camera.h"
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "MeasurementSLAM.h"


// -------------------------------------------------------
// Bounds
// -------------------------------------------------------
struct Bounds
{
    Bounds();
    void getVTKBounds(double * bounds) const;
    void setExtremity(Bounds & extremity) const;
    void calculateMaxMinSigmaPoints(const GaussianInfo<double> & positionDensity, const double sigma);

    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

// -------------------------------------------------------
// QuadricPlot
// -------------------------------------------------------
struct QuadricPlot
{
    QuadricPlot();
    void update(const GaussianInfo<double> & positionDensity);
    vtkActor * getActor() const;
    Bounds bounds;
    vtkSmartPointer<vtkActor>            contourActor;
    vtkSmartPointer<vtkContourFilter>    contours;
    vtkSmartPointer<vtkPolyDataMapper>   contourMapper;
    vtkSmartPointer<vtkQuadric>          quadric;
    vtkSmartPointer<vtkSampleFunction>   sample;
    const double value;
    bool isInit;
};

// -------------------------------------------------------
// FrustumPlot
// -------------------------------------------------------
struct FrustumPlot
{
    explicit FrustumPlot(const Camera & camera);
    void update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc);
    vtkActor * getActor() const;

    vtkSmartPointer<vtkActor> pyramidActor;
    vtkSmartPointer<vtkCellArray> cells;
    vtkSmartPointer<vtkDataSetMapper> mapper;
    vtkSmartPointer<vtkPoints> pyramidPts;
    vtkSmartPointer<vtkPyramid> pyramid;
    vtkSmartPointer<vtkUnstructuredGrid> ug;
    Eigen::MatrixXd rPCc;
    Eigen::MatrixXd rPNn;
    bool isInit;
};

// -------------------------------------------------------
// AxisPlot
// -------------------------------------------------------
struct AxisPlot
{
    AxisPlot();
    void init(vtkCamera *cam);
    void update(const Bounds & bounds);
    vtkActor * getActor() const;

    vtkColor3d axis1Color;
    vtkColor3d axis2Color;
    vtkColor3d axis3Color;
    vtkSmartPointer<vtkCubeAxesActor> cubeAxesActor;
    bool isInit;
};

// -------------------------------------------------------
// BasisPlot
// -------------------------------------------------------
struct BasisPlot
{
    BasisPlot();
    void update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc);
    vtkProp3D * getActor() const;

    vtkSmartPointer<vtkAxesActor>       axesActor;
    vtkSmartPointer<vtkTransform>       transform;
    bool                                isInit;
};

// -------------------------------------------------------
// ImagePlot
// -------------------------------------------------------
struct ImagePlot
{
    ImagePlot();
    void init(double rendererWidth, double rendererHeight);
    void update(const cv::Mat & view);
    vtkActor2D * getActor() const;

    vtkSmartPointer<vtkImageData> viewVTK;
    vtkSmartPointer<vtkActor2D> imageActor2d;
    vtkSmartPointer<vtkImageMapper> imageMapper;
    cv::Mat cvVTKBuffer;
    double width, height;
    bool isInit;
};

// -------------------------------------------------------
// Custom Interactor Style with Keyboard Shortcuts
// -------------------------------------------------------
class SLAMInteractorStyle : public vtkInteractorStyleTrackballCamera
{
public:
    static SLAMInteractorStyle* New();
    vtkTypeMacro(SLAMInteractorStyle, vtkInteractorStyleTrackballCamera);
    
    virtual void OnKeyPress() override;
    
    // Frame advancement control for interactive mode
    void setInteractiveMode(bool isInteractive) { interactiveMode = isInteractive; }
    bool shouldAdvanceFrame() const { return advanceFrame; }
    void resetAdvanceFrame() { advanceFrame = false; }
    
protected:
    SLAMInteractorStyle() : interactiveMode(false), advanceFrame(false) {}
    virtual ~SLAMInteractorStyle() {}
    
private:
    bool interactiveMode;
    bool advanceFrame;
};

// -------------------------------------------------------
// Plot
// -------------------------------------------------------
struct Plot
{
public:
    explicit Plot(const Camera & camera);
    void render();
    void start() const;
    void enableInteraction();  // Enable non-blocking interaction
    void processEvents();      // Process VTK events without blocking
    void setInteractiveMode(bool isInteractive);  // Set interactive frame advancement mode
    bool shouldAdvanceFrame() const;              // Check if frame should advance
    void resetAdvanceFrame();                     // Reset frame advancement flag
    void setData(const SystemSLAM & system, const MeasurementSLAM & measurement);
    cv::Mat getFrame() const;

private:
    std::unique_ptr<SystemSLAM> pSystem;
    std::unique_ptr<MeasurementSLAM> pMeasurement;
    const Camera & camera;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer>     threeDimRenderer;
    vtkSmartPointer<vtkRenderer>     imageRenderer;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    vtkSmartPointer<SLAMInteractorStyle> interactorStyle;
    QuadricPlot qpCamera;
    std::vector<QuadricPlot> qpLandmarks;
    FrustumPlot fp;
    AxisPlot ap;
    BasisPlot bp;
    ImagePlot ip;
};

#endif
