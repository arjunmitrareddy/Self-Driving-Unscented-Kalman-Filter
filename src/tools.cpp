#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  if (estimations.size() == 0) {
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    throw "Estimation & Ground Truth Size Does Not Match";
  }
  for(int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    VectorXd residual_squared = residual.array() * residual.array();
    rmse += residual_squared;
    if(true && i > 5 && (residual_squared(0) > 2 || residual_squared(1) > 2 || residual_squared(2) > 30 || residual_squared(3) > 30)) {
      throw "Residual too high";
    }
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}