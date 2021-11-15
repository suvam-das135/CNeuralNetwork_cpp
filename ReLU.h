#pragma once
#include<algorithm>
#include <Eigen/Core>
#include "Config.h"
using namespace std;

class ReLU
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void activate(const Matrix& Z, Matrix& A) {
		
	}

	static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G) {

	}
};