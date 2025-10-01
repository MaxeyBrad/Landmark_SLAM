#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "rotation.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

SystemSLAM::SystemSLAM(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{}

// Transformation matrix from body angular velocities to Euler angle derivatives
template <typename Scalar>
Eigen::Matrix3<Scalar> TK(const Eigen::Vector3<Scalar> & Theta)
{
    using std::cos, std::sin, std::tan;
    Scalar phi = Theta(0);    // roll
    Scalar theta = Theta(1);  // pitch
    Scalar psi = Theta(2);    // yaw
    
    Eigen::Matrix3<Scalar> T;
    T << 1,  sin(phi)*tan(theta),  cos(phi)*tan(theta),
         0,  cos(phi),             -sin(phi),
         0,  sin(phi)/cos(theta),   cos(phi)/cos(theta);
    
    return T;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(density.dim() == x.size());
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //
    Eigen::VectorXd f(x.size());
    f.setZero();

    // Extract state components
    Eigen::Vector3d vBNb = x.segment<3>(0);       // Body translational velocity (indices 0-2)
    Eigen::Vector3d omegaBNb = x.segment<3>(3);   // Body angular velocity (indices 3-5)  
    Eigen::Vector3d rBNn = x.segment<3>(6);       // Body position (indices 6-8)
    Eigen::Vector3d thetaBN = x.segment<3>(9);    // Body orientation RPY (indices 9-11)

    // Implement motion model:
    // d(vBNb)/dt = 0 (constant velocity assumption)
    // d(omegaBNb)/dt = 0 (constant angular velocity assumption)
    // d(rBNn)/dt = Rnb(thetaBN) * vBNb
    // d(thetaBN)/dt = TK(thetaBN) * omegaBNb
    // d(landmarks)/dt = 0 (static landmarks)
    
    // Velocity derivatives are zero (constant velocity model)
    f.segment<3>(0).setZero();  // d(vBNb)/dt = 0
    f.segment<3>(3).setZero();  // d(omegaBNb)/dt = 0
    
    // Position derivative: dr/dt = R * v
    Eigen::Matrix3d Rnb = rpy2rot(thetaBN);
    f.segment<3>(6) = Rnb * vBNb;
    
    // Orientation derivative: dtheta/dt = TK * omega
    Eigen::Matrix3d TKmat = TK(thetaBN);
    f.segment<3>(9) = TKmat * omegaBNb;
    
    // Landmark derivatives are zero (static landmarks)
    // (already set to zero by f.setZero())

    return f;
}

// // Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
// Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
// {
//     Eigen::VectorXd f = dynamics(t, x, u);

//     // Jacobian J = df/dx
//     //    
//     //     [  0                  0 0 ]
//     // J = [ JK d(JK(eta)*nu)/deta 0 ]
//     //     [  0                  0 0 ]
//     //
//     J.resize(f.size(), x.size());
//     J.setZero();

//     // Extract state components
//     Eigen::Vector3d vBNb = x.segment<3>(0);       
//     Eigen::Vector3d omegaBNb = x.segment<3>(3);   
//     Eigen::Vector3d thetaBN = x.segment<3>(9);    

//     // Jacobian for position derivative: df6_8/dx = d(Rnb * vBNb)/dx
//     // df6_8/dvBNb = Rnb (derivative w.r.t. velocity)
//     Eigen::Matrix3d Rnb = rpy2rot(thetaBN);
//     J.block<3,3>(6, 0) = Rnb;

//     // df6_8/dthetaBN = d(Rnb)/dthetaBN * vBNb (derivative w.r.t. orientation)
//     // This requires computing derivative of rotation matrix - complex but important for accuracy
//     // For now, we'll use a simplified approach focusing on the main coupling
    
//     // Jacobian for orientation derivative: df9_11/dx = d(TK * omegaBNb)/dx  
//     // df9_11/domegaBNb = TK (derivative w.r.t. angular velocity)
//     Eigen::Matrix3d TKmat = TK(thetaBN);
//     J.block<3,3>(9, 3) = TKmat;

//     // df9_11/dthetaBN = d(TK)/dthetaBN * omegaBNb (derivative w.r.t. current orientation)
//     // This also requires derivative of TK matrix - simplified for now

//     return f;
// }

// Add these helper functions to your SystemSLAM.cpp file

// Helper function for autodiff: position dynamics dr/dt = Rnb(θ) * v
template<typename Scalar>
Eigen::Vector3<Scalar> positionDynamics(const Eigen::VectorX<Scalar>& x) {
    Eigen::Vector3<Scalar> vBNb = x.template segment<3>(0);     // velocity [0-2]
    Eigen::Vector3<Scalar> thetaBN = x.template segment<3>(9);  // orientation [9-11]
    return rpy2rot(thetaBN) * vBNb;
}

// Helper function for autodiff: orientation dynamics dθ/dt = TK(θ) * ω
template<typename Scalar>
Eigen::Vector3<Scalar> orientationDynamics(const Eigen::VectorX<Scalar>& x) {
    Eigen::Vector3<Scalar> omegaBNb = x.template segment<3>(3); // angular velocity [3-5]
    Eigen::Vector3<Scalar> thetaBN = x.template segment<3>(9);  // orientation [9-11]
    return TK(thetaBN) * omegaBNb;
}

// Updated Jacobian function
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    // Jacobian J = df/dx
    J.resize(f.size(), x.size());
    J.setZero();

    // Extract state components
    Eigen::Vector3d vBNb = x.segment<3>(0);       
    Eigen::Vector3d omegaBNb = x.segment<3>(3);   
    Eigen::Vector3d thetaBN = x.segment<3>(9);    

    // Existing terms - derivatives w.r.t. velocities
    Eigen::Matrix3d Rnb = rpy2rot(thetaBN);
    J.block<3,3>(6, 0) = Rnb;  // ∂(dr/dt)/∂v = Rnb

    Eigen::Matrix3d TKmat = TK(thetaBN);
    J.block<3,3>(9, 3) = TKmat;  // ∂(dθ/dt)/∂ω = TK

    // NEW: Missing cross-derivatives using autodiff
    
    // Compute ∂(dr/dt)/∂θ = ∂(Rnb*v)/∂θ
    {
        Eigen::Matrix3d J_pos_theta;
        Eigen::Vector3<autodiff::dual> pos_result;
        Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
        
        // Get Jacobian of position dynamics w.r.t. entire state vector
        Eigen::MatrixXd J_pos_full = jacobian(positionDynamics<autodiff::dual>, wrt(x_dual), at(x_dual), pos_result);
        
        // Extract the part we need: ∂(dr/dt)/∂θ [rows 0-2, cols 9-11]
        J.block<3,3>(6, 9) = J_pos_full.block<3,3>(0, 9);
    }
    
    // Compute ∂(dθ/dt)/∂θ = ∂(TK*ω)/∂θ  
    {
        Eigen::Matrix3d J_orient_theta;
        Eigen::Vector3<autodiff::dual> orient_result;
        Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
        
        // Get Jacobian of orientation dynamics w.r.t. entire state vector
        Eigen::MatrixXd J_orient_full = jacobian(orientationDynamics<autodiff::dual>, wrt(x_dual), at(x_dual), orient_result);
        
        // Extract the part we need: ∂(dθ/dt)/∂θ [rows 0-2, cols 9-11]
        J.block<3,3>(9, 9) = J_orient_full.block<3,3>(0, 9);
    }
    std::cout << "Position cross-derivative norm: " << J.block<3,3>(6, 9).norm() << std::endl;
    std::cout << "Orientation cross-derivative norm: " << J.block<3,3>(9, 9).norm() << std::endl;
    return f;
}

Eigen::VectorXd SystemSLAM::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemSLAM::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ;
    
    // Process noise only on velocity states (constant velocity assumption with noise)
    // Dimension 6: [vBNb(3), omegaBNb(3)]
    SQ = Eigen::MatrixXd::Zero(6, 6);
    
    // Translational velocity noise standard deviation (m/s/sqrt(s))
    double sigma_v = 0.2;  // 20 cm/s per sqrt(second) - increased for better tracking
    
    // Angular velocity noise standard deviation (rad/s/sqrt(s))  
    double sigma_omega = 0.1;  // ~6 degrees/s per sqrt(second) - increased for better tracking
    
    // Diagonal noise model (uncorrelated velocity components)
    for (int i = 0; i < 3; ++i) {
        SQ(i, i) = sigma_v;        // Translational velocity noise
        SQ(i+3, i+3) = sigma_omega; // Angular velocity noise
    }

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemSLAM::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    
    // Process noise is injected on velocity states (indices 0-5)
    // vBNb: indices 0, 1, 2
    // omegaBNb: indices 3, 4, 5
    for (int i = 0; i < 6; ++i) {
        idxQ.push_back(i);
    }
    
    return idxQ;
}

cv::Mat & SystemSLAM::view()
{
    return view_;
};

const cv::Mat & SystemSLAM::view() const
{
    return view_;
};

GaussianInfo<double> SystemSLAM::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemSLAM::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemSLAM::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemSLAM::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d SystemSLAM::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemSLAM::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemSLAM::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}
