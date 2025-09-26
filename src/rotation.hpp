#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(1,1) = cos(x);   // cos(φ)
    R(1,2) = -sin(x);  // -sin(φ)
    R(2,1) = sin(x);   // sin(φ)
    R(2,2) = cos(x);   // cos(φ)
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(0,0) = cos(x);   // cos(θ)
    R(0,2) = sin(x);   // sin(θ)
    R(2,0) = -sin(x);  // -sin(θ)
    R(2,2) = cos(x);   // cos(θ)
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(0,0) = cos(x);   // cos(ψ)
    R(0,1) = -sin(x);  // -sin(ψ)
    R(1,0) = sin(x);   // sin(ψ)
    R(1,1) = cos(x);   // cos(ψ)
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> rpy2rot(const Eigen::MatrixBase<Derived> & Theta)
{
    using Scalar = typename Derived::Scalar;
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    Scalar phi = Theta(0);   // φ (roll)
    Scalar theta = Theta(1); // θ (pitch)  
    Scalar psi = Theta(2);   // ψ (yaw)

    R = rotz(psi) * roty(theta) * rotx(phi);
    return R;
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> rot2rpy(const Eigen::MatrixBase<Derived> & R)
{
    using Scalar = typename Derived::Scalar;
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta;
    // Extract angles from R = Rz(ψ)Ry(θ)Rx(φ)
    Scalar phi = atan2(R(2,1), R(2,2));
    Scalar theta = atan2(-R(2,0), hypot(R(2,1), R(2,2)));
    Scalar psi = atan2(R(1,0), R(0,0));

    Theta << phi, theta, psi;
    return Theta;
}

#endif
