#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  use_laser_ = true;
  use_radar_ = true;
  
  x_ = VectorXd(5);
  P_ = MatrixXd(5, 5);
  std_a_ = 1.4;
  std_yawdd_ = 0.3;
  std_laspx_ = 0.15;
  std_laspy_ = 0.15;
  std_radr_ = 0.4;
  std_radphi_ = 0.03;
  std_radrd_ = 0.3;
  n_x_ = 5;
  n_aug_ = 7;
  cycle = 0;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.5/(n_aug_+lambda_));
  weights_(0) = lambda_/(lambda_+n_aug_);
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if(!is_initialized_) {
    x_.fill(0.0);
    P_.fill(0.0);
    time_us_ = meas_package.timestamp_;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double ro = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      double ro_dot = meas_package.raw_measurements_[2];

      double px = ro * sin(theta);
      double py = ro * cos(theta);

      double v = ro_dot;

      double psi = theta;

      double psi_dot = 0.0;

      x_ << px, py, v, psi, psi_dot;

      double spatial_uncertainty = max(std_radr_, std_radphi_ * ro);

      P_(0,0) = spatial_uncertainty * spatial_uncertainty;
      P_(1,1) = spatial_uncertainty * spatial_uncertainty;
      P_(2,2) = pow(6.5,2);
      P_(3,3) = pow(M_PI,2);
      P_(4,4) = pow(0.2,2);
      
    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {

      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];

      x_ << px, py, 0, 0, 0;

      P_(0,0) = pow(std_laspx_,2);
      P_(1,1) = pow(std_laspy_,2);
      P_(2,2) = pow(6.5,2);
      P_(3,3) = pow(M_PI,2);
      P_(4,4) = pow(0.2,2);

    } else {
      throw "Unrecognized sensor type";
    }

    is_initialized_ = true;
    cycle++;
    return;
  }
  double delta_t = (meas_package.timestamp_ - time_us_) * 0.000001;
  if(false && cycle == 1) { 
    double old_px = x_(0);
    double old_py = x_(1);
    double new_px = x_(0);
    double new_py = x_(1);
    if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
      new_px = meas_package.raw_measurements_(0);
      new_py = meas_package.raw_measurements_(1);
    }
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double ro = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      new_px = ro * sin(theta);
      new_py = ro * cos(theta);
    }
    double dx = new_px - old_px;
    double dy = new_py - old_py;
    if(fabs(dx) > 0.01 || fabs(dy) > 0.01) {
      x_(2) = sqrt(dy*dy + dx*dx) / delta_t; 
      x_(3) = atan2(dy,dx); 
      P_(2,2) = 0.2; 
      P_(3,3) = 1.0; 
      P_(4,4) = 0.1; 
    }
  }
  Prediction(delta_t);
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    if(use_radar_) {
      UpdateRadar(meas_package);
    }
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    if(use_laser_) {
      UpdateLidar(meas_package);
    }
  } else {
    throw "Unrecognized sensor type";
  }
  time_us_ = meas_package.timestamp_;
  cycle++;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  P_aug.fill(0.0);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  x_aug.head(n_x_) = x_;
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_ * std_yawdd_;
  MatrixXd A_aug = P_aug.llt().matrixL();
  MatrixXd B_aug = A_aug * sqrt(lambda_+n_aug_);
  
  for(int i = 0; i < n_aug_; i++) {
    if(B_aug(3,i) > M_PI/4) B_aug(3,i) = M_PI/4;
    if(B_aug(3,i) < -M_PI/4) B_aug(3,i) = -M_PI/4;
    if(B_aug(4,i) > 0.3) B_aug(4,i) = 0.3;
    if(B_aug(4,i) < -M_PI/4) B_aug(4,i) = -0.3;
  }
  Xsig_aug.block(0,0,n_aug_,1) = x_aug;
  Xsig_aug.block(0,1,n_aug_,n_aug_) = B_aug.colwise() + x_aug;
  Xsig_aug.block(0,n_aug_+1,n_aug_,n_aug_) = (B_aug.colwise() - x_aug) * -1.0;

  for(int i=0; i < 2*n_aug_+1; i++) {
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    double px_pred, py_pred;
    if(fabs(v < 0.001)) {
      px_pred = px;
      py_pred = py;
    } else if(fabs(yawd) > 0.001) {
      px_pred = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_pred = py + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    } else {
      px_pred = px + v*delta_t*cos(yaw);
      py_pred = py + v*delta_t*sin(yaw);
    }
    double v_pred = v;
    double yaw_pred = yaw + yawd*delta_t;
    double yawd_pred = yawd;

    px_pred += 0.5*nu_a*delta_t*delta_t*cos(yaw);
    py_pred += 0.5*nu_a*delta_t*delta_t*sin(yaw);
    v_pred += nu_a*delta_t;
    yaw_pred += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_pred += nu_yawdd*delta_t;

    while(yaw_pred > M_PI) yaw_pred -= 2. * M_PI;
    while(yaw_pred < -M_PI) yaw_pred += 2. * M_PI;
    if(yawd_pred > 2.0) yawd_pred = 2.0;
    if(yawd_pred < -2.0) yawd_pred = -2.0;

    Xsig_pred_(0,i) = px_pred;
    Xsig_pred_(1,i) = py_pred;
    Xsig_pred_(2,i) = v_pred;
    Xsig_pred_(3,i) = yaw_pred;
    Xsig_pred_(4,i) = yawd_pred;
  }

  double yaw_offset = - Xsig_pred_(3,0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Xsig_pred_(3,i) = Xsig_pred_(3,i) + yaw_offset;
    while(Xsig_pred_(3,i) > M_PI) Xsig_pred_(3,i) -= 2. * M_PI;
    while(Xsig_pred_(3,i) < -M_PI) Xsig_pred_(3,i) += 2. * M_PI;
  }

  x_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  if(x_(2) > 10.0) x_(2) = 10.0;
  if(x_(4) > 2.0) x_(4) = 2.0;
  if(x_(4) < -2.0) x_(4) = -2.0;

  P_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    P_ += weights_(i) * diff * diff.transpose();
  }

  x_(3) = x_(3) - yaw_offset;
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Xsig_pred_(3,i) = Xsig_pred_(3,i) - yaw_offset;
    while(Xsig_pred_(3,i) > M_PI) Xsig_pred_(3,i) -= 2. * M_PI;
    while(Xsig_pred_(3,i) < -M_PI) Xsig_pred_(3,i) += 2. * M_PI;
  }

  if(x_(2) < -0.01) {
    x_(2) = x_(2) * -1.0;
    x_(3) = x_(3) + M_PI;
  }

  while(x_(3) > M_PI) x_(3) -= 2. * M_PI;
  while(x_(3) < -M_PI) x_(3) += 2. * M_PI;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z,2*n_aug_+1);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S += weights_(i) * diff * diff.transpose();
  }
  S(0,0) += std_laspx_*std_laspx_;
  S(1,1) += std_laspy_*std_laspy_;

  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  VectorXd x_diff = K * z_diff;
  if(cycle > 10) {
    if(x_diff(3) > 0.5) x_diff(3) = 0.5;
    if(x_diff(3) < -0.5) x_diff(3) = -0.5;
  }
  x_ += x_diff;
  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z,2*n_aug_+1);

  for(int i = 0; i < 2*n_aug_+1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double rho = sqrt(px*px + py*py);
    Zsig(0,i) = rho;
    Zsig(1,i) = atan2(py,px);
    Zsig(2,i) = v*(px*cos(yaw) + py*sin(yaw)) / rho;
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    while(diff(1) > M_PI) diff(1) -= 2. * M_PI;
    while(diff(1) < -M_PI) diff(1) += 2. * M_PI;
    S += weights_(i) * diff * diff.transpose();
  }
  S(0,0) += std_radr_*std_radr_;
  S(1,1) += std_radphi_*std_radphi_;
  S(2,2) += std_radrd_*std_radrd_;

  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  VectorXd x_diff = K * z_diff;
  x_ += x_diff;
  P_ -= K * S * K.transpose();
}