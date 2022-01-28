#ifndef NULLSOLVER_HPP
#define NULLSOLVER_HPP

namespace S132_Eigen
{

template<class DataMatrixType,
		 template<class ...Args> class SolverTemplate
>
struct NullVectorSolver{
};


namespace detail{

	template<class MatrixType>
	struct RowOf{
		using Result=decltype(MatrixType{}.row(0));
		using Vec=typename Result::PlainMatrix;
		using VecT=decltype(Vec{}.adjoint());

		static auto default_ones(const Eigen::EigenBase<MatrixType>& A)
		{
			return Vec::Ones(A.cols());
		}
	};

	template<class InputType>
	auto GetCov(const Eigen::EigenBase<InputType>& A,
		const Eigen::EigenBase<typename RowOf<InputType>::Vec>& cvec)
	{
		return A.adjoint()*A+cvec.adjoint()*cvec;
	}

	template<class DataMatrixType>
	struct NullVectorSolverCovBase{
		using CovMatrixType=Eigen::Matrix<
			typename DataMatrixType::Scalar,
			DataMatrixType::ColsAtCompileTime,
			DataMatrixType::ColsAtCompileTime,
			DataMatrixType::Flags,
			DataMatrixType::MaxColsAtCompileTime,
			DataMatrixType::MaxColsAtCompileTime
		>;
	};

	template<
		class DataMatrixType,
		template<class MT>
			class LUType
	>
	struct NullVectorSolverCov:
			private LUType<typename NullVectorSolverCovBase<DataMatrixType>::CovMatrixType>
	{
	public:
		using UnitaryConstraintType=typename RowOf<DataMatrixType>::Vec;
		using ResultType=typename RowOf<DataMatrixType>::VecT;
	protected:
		using Base=LUType<typename NullVectorSolverCovBase<DataMatrixType>::CovMatrixType>;
		using CovMatrixType=typename NullVectorSolverCovBase<DataMatrixType>::CovMatrixType;

		UnitaryConstraintType uconst;
		LUType<CovMatrixType> lu;
	public:
		NullVectorSolverLU(const Eigen::EigenBase<DataMatrixType>& inp,
			const Eigen::EigenBase<UnitaryConstraintType>& unitary_constraint){
			compute(inp,unitary_constraint);
		}
		NullVectorSolverLU(const Eigen::EigenBase<DataMatrixType>& inp){
			compute(inp);
		}
		auto solve() const{
			return lu.solve(uconst.adjoint());
		}
		void compute(const Eigen::EigenBase<DataMatrixType>& inp,
					 const Eigen::EigenBase<UnitaryConstraintType>& unitary_constraint)
		{}
		void compute(const Eigen::EigenBase<S132_Eigen::detail::DataMatrixType> &inp)
		{
			compute(inp,RowOf<DataMatrixType>::default_ones(inp));
		}
	};
}

}

#endif
