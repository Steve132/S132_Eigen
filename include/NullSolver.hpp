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
	struct GeomOf{
		using _RVec=decltype(MatrixType{}.row(0));
		using Vec=typename _RVec::PlainMatrix;

		using _RVecT=decltype(Vec{}.adjoint());
		using VecT=typename _RVecT::PlainMatrix;

		using _RCovMatrixType=decltype(VecT{}*Vec{});
		using CovMatrixType=typename _RCovMatrixType::PlainMatrix;
		static auto default_ones(const Eigen::MatrixBase<MatrixType>& A)
		{
			return Vec::Ones(A.cols());
		}
	};

	template<class InputType,class Input2>
	auto GetCov(const Eigen::MatrixBase<InputType>& A,
		const Eigen::MatrixBase<Input2>& cvec)
	{
		return A.adjoint()*A+cvec.adjoint()*cvec;
	}

	template<
		class DataMatrixType,
		template<class MT>
			class SolverType
	>
	struct NullVectorSolverCov:
			public SolverType<typename GeomOf<DataMatrixType>::CovMatrixType>
	{
	public:
		using UnitaryConstraintType=typename GeomOf<DataMatrixType>::Vec;
		using ResultType=typename GeomOf<DataMatrixType>::VecT;
	protected:
		using CovMatrixType=typename GeomOf<DataMatrixType>::CovMatrixType;
		using Base=SolverType<CovMatrixType>;

		UnitaryConstraintType uconst;
		//SolverType<CovMatrixType> solver;
	public:

		template<class InputMatrix1,class InputMatrix2>
		NullVectorSolverCov(
				const Eigen::MatrixBase<InputMatrix1>& inp,
				const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			compute(inp,unitary_constraint);
		}
		template<class InputMatrix1,class InputMatrix2>
		void compute(const Eigen::MatrixBase<InputMatrix1>& inp,
					 const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			Base::compute(GetCov(inp,unitary_constraint));
		}
		template<class InputMatrix>
		void compute(const Eigen::MatrixBase<InputMatrix> &inp)
		{
			compute(inp,GeomOf<DataMatrixType>::default_ones(inp));
		}
		template<class InputMatrix>
		NullVectorSolverCov(const Eigen::MatrixBase<InputMatrix>& inp)
		{
			compute(inp);
		}

		auto solve() {
			return Base::solve(uconst.adjoint());
		}
	};
}

}

#endif
