#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(10.0) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementPointBundle::clone() const
{
    return new MeasurementPointBundle(*this);
}

Eigen::VectorXd MeasurementPointBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);

    // Select visible landmarks
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // FIXME: This just selects all landmarks
    // TODO: Assignment(s)
    return 0.0;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system);
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system, g);
}

void MeasurementPointBundle::update(SystemBase & system)
{
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(system);    // Do the actual measurement update
}

Eigen::Vector2d MeasurementPointBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Use forward-mode automatic differentiation to compute Jacobian
    using namespace autodiff;
    
    // Convert x to dual numbers
    Eigen::VectorX<dual> xdual = x.cast<dual>();
    
    // Create lambda that calls the template function
    auto func = [&](const Eigen::VectorX<dual>& xd) {
        return predictFeature(xd, system, idxLandmark);
    };
    
    // Compute Jacobian using autodiff
    Eigen::Vector2<dual> ydual;
    J = jacobian(func, wrt(xdual), at(xdual), ydual);
    
    // Extract the function value (convert dual to double)
    Eigen::Vector2d result;
    result(0) = val(ydual(0));
    result(1) = val(ydual(1));
    return result;
}
// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementPointBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2;

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeature(x, J, system, idxLandmark) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = system.density.dim();
    assert(x.size() == nx);

    Eigen::VectorXd h(2*nL);
    J.resize(2*nL, nx);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::MatrixXd Jfeature;
        Eigen::Vector2d rQOi = predictFeature(x, Jfeature, system, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 9
        // Set pair of rows of J
        // TODO: Lab 9
    }
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementPointBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2*idxLandmarks.size();

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeatureBundle(x, J, system, idxLandmarks) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };

    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

#include "association_util.h"
const std::vector<int> & MeasurementPointBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    snn(system, featureBundleDensity, idxLandmarks, Y_, camera_, idxFeatures_);
    return idxFeatures_;
}
