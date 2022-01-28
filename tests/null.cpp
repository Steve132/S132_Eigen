#include<Eigen/Dense>
#include "NullSolver.hpp"

int main()
{

	Eigen::RowVector2d z{2.0,1.0};
	S132_Eigen::detail::NullVectorSolverLU<Eigen::RowVector2d,Eigen::FullPivLU> nvec(z);
	return 0;
}
