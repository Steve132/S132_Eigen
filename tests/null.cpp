#include<Eigen/Dense>
#include "NullSolver.hpp"
#include<iostream>

int main()
{

	Eigen::RowVector2d z{2.0,1.0};
	S132_Eigen::NullVectorSolver<Eigen::RowVector2d,Eigen::FullPivLU> nvec1(z);
	std::cout << nvec1.solve() << std::endl;
	S132_Eigen::NullVectorSolver<Eigen::RowVector2d,Eigen::HouseholderQR> nvec2(z);
	std::cout << nvec2.solve() << std::endl;
	return 0;
}
