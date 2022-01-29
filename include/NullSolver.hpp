#ifndef NULLSOLVER_HPP
#define NULLSOLVER_HPP

namespace S132_Eigen
{

template<class DataMatrixType,
		 template<class MT,auto ...Args> class SolverTemplate,
		 auto ...TArgs
>
struct NullVectorSolver{
};


namespace detail{

	template<class MatrixType>
	struct GeomOf{
		using _RVec=decltype(MatrixType{}.template topRows<1>());
		using Vec=typename _RVec::PlainMatrix;

		using _RVecT=decltype(Vec{}.adjoint());
		using VecT=typename _RVecT::PlainMatrix;

		using _RCovMatrixType=decltype(VecT{}*Vec{});
		using CovMatrixType=typename _RCovMatrixType::PlainMatrix;

		using QRMatrixType=Eigen::Matrix<typename MatrixType::Scalar,
			MatrixType::RowsAtCompileTime==Eigen::Dynamic ? Eigen::Dynamic : (MatrixType::RowsAtCompileTime+1),
			MatrixType::ColsAtCompileTime,
			MatrixType::Options,
			MatrixType::MaxRowsAtCompileTime==Eigen::Dynamic ? Eigen::Dynamic : (MatrixType::MaxRowsAtCompileTime+1),
			MatrixType::MaxColsAtCompileTime==Eigen::Dynamic ? Eigen::Dynamic : (MatrixType::MaxColsAtCompileTime+1)
		>;
		using _RColType=decltype(MatrixType{}.template leftCols<1>());
		using ColType=typename _RColType::PlainMatrix;
		using _RQRColType=decltype(QRMatrixType{}.template leftCols<1>());
		using QRColType=typename _RQRColType::PlainMatrix;

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
	template<class InputType,class Input2>
	auto GetStack(const Eigen::MatrixBase<InputType>& A,
		const Eigen::MatrixBase<Input2>& cvec)
	{
		return (typename GeomOf<InputType>::QRMatrixType::Zero(A.rows()+1,A.cols()) << A,cvec);
	}
	template<class InputType>
	auto GetQrRHS(size_t rows)
	{
		return (typename GeomOf<InputType>::QRColType::Zero(rows) <<
			typename GeomOf<InputType>::ColType::Zero(rows+1),1.0
		);
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


	template<
		class DataMatrixType,
		template<class MT>
			class SolverType
	>
	struct NullVectorSolverQR:
			public SolverType<typename GeomOf<DataMatrixType>::QRMatrixType>
	{
	public:
		using UnitaryConstraintType=typename GeomOf<DataMatrixType>::Vec;
		using ResultType=typename GeomOf<DataMatrixType>::VecT;
	protected:
		using QRMatrixType=typename GeomOf<DataMatrixType>::QRMatrixType;
		using Base=SolverType<QRMatrixType>;
		using QRColType=typename GeomOf<DataMatrixType>::QRColType;

		UnitaryConstraintType uconst;
		size_t rows;
	public:

		template<class InputMatrix1,class InputMatrix2>
		NullVectorSolverQR(
				const Eigen::MatrixBase<InputMatrix1>& inp,
				const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			compute(inp,unitary_constraint);
		}
		template<class InputMatrix1,class InputMatrix2>
		void compute(const Eigen::MatrixBase<InputMatrix1>& inp,
					 const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			rows=inp.rows();
			Base::compute(GetStack(inp,unitary_constraint));
		}
		template<class InputMatrix>
		void compute(const Eigen::MatrixBase<InputMatrix> &inp)
		{
			compute(inp,GeomOf<DataMatrixType>::default_ones(inp));
		}
		template<class InputMatrix>
		NullVectorSolverQR(const Eigen::MatrixBase<InputMatrix>& inp)
		{
			compute(inp);
		}

		auto solve() {
			return Base::solve(GetQrRHS<DataMatrixType>(rows));
		}
	};

	template<
		class DataMatrixType,
		template<class MT>
			class SolverType
	>
	struct NullVectorSolverSVD:
			public SolverType<DataMatrixType>
	{
	public:
		using UnitaryConstraintType=typename GeomOf<DataMatrixType>::Vec;
		using ResultType=typename GeomOf<DataMatrixType>::VecT;
	protected:
		using Base=SolverType<DataMatrixType>;

		UnitaryConstraintType uconst;
	public:

		template<class InputMatrix1,class InputMatrix2>
		NullVectorSolverSVD(
				const Eigen::MatrixBase<InputMatrix1>& inp,
				const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			compute(inp,unitary_constraint);
		}
		template<class InputMatrix1,class InputMatrix2>
		void compute(const Eigen::MatrixBase<InputMatrix1>& inp,
					 const Eigen::MatrixBase<InputMatrix2>& unitary_constraint)
		{
			Base::compute(inp,Eigen::ComputeThinU);
		}
		template<class InputMatrix>
		void compute(const Eigen::MatrixBase<InputMatrix> &inp)
		{
			compute(inp,GeomOf<DataMatrixType>::default_ones(inp));
		}
		template<class InputMatrix>
		NullVectorSolverSVD(const Eigen::MatrixBase<InputMatrix>& inp)
		{
			compute(inp);
		}

		auto solve() {
			return Base::matrixV().template rightCols<1>();
		}
	};

	template<
		template<class MT,auto ...Args> class SolverTemplate,
		auto ...InpArgs>
	struct SolverWrapper
	{
		template<class DT>
		using SolverType=SolverTemplate<DT,InpArgs...>;
	};
	template<class DT>
	struct OptimalUpLo {
		static const int value = (DT::Options & Eigen::RowMajor) ? Eigen::Upper : Eigen::Lower;
	};
}

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::PartialPivLU>:
	public detail::NullVectorSolverCov<DataMatrixType,Eigen::PartialPivLU>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::FullPivLU>:
	public detail::NullVectorSolverCov<DataMatrixType,Eigen::FullPivLU>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::LLT>:
	public detail::NullVectorSolverCov<DataMatrixType,
		detail::SolverWrapper<Eigen::LLT,
			Eigen::Lower //Todo: uplo
		>::SolverType
	>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::LDLT>:
	public detail::NullVectorSolverCov<DataMatrixType,
		detail::SolverWrapper<Eigen::LDLT,
			Eigen::Lower
		>::SolverType
	>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::HouseholderQR>:
	public detail::NullVectorSolverQR<DataMatrixType,Eigen::HouseholderQR>
{};
template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::ColPivHouseholderQR>:
	public detail::NullVectorSolverQR<DataMatrixType,Eigen::ColPivHouseholderQR>
{};
template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::FullPivHouseholderQR>:
	public detail::NullVectorSolverQR<DataMatrixType,Eigen::FullPivHouseholderQR>
{};
template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::CompleteOrthogonalDecomposition>:
	public detail::NullVectorSolverQR<DataMatrixType,Eigen::CompleteOrthogonalDecomposition>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::BDCSVD>:
	public detail::NullVectorSolverSVD<DataMatrixType,
		Eigen::BDCSVD
	>
{};

template<
	class DataMatrixType
>
struct NullVectorSolver<DataMatrixType,
		Eigen::JacobiSVD>:
	public detail::NullVectorSolverSVD<DataMatrixType,
			detail::SolverWrapper<Eigen::JacobiSVD,
				Eigen::ColPivHouseholderQRPreconditioner
			>::SolverType
		>
{};
}

#endif
