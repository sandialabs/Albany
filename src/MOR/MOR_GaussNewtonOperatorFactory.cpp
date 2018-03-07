//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_GaussNewtonOperatorFactory.hpp"

#include "MOR_BasisOps.hpp"

#include "Epetra_Operator.h"
#include "Epetra_Map.h"

#include "Epetra_CrsMatrix.h"  //JF
#include "Epetra_Vector.h"  //JF
#include "Amesos.h"  //JF

#include "EpetraExt_MultiVectorOut.h"

// Set invJacPrec to true only if you REALLY want to enable preconditioning
//   with the inverse Jacobian.  It's a memory hog and causes issues for large
//   problems (i.e. PCAP), so it's commented out for now.
#define invJacPrec false // ALSO MOR_ReducedOrderModelEvaluator.cpp

namespace MOR {

using ::Teuchos::RCP;


template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::parOut(std::string text) const
{
	if (reducedBasis_->Comm().MyPID() == 0)
		std::cout << text << std::endl;
}

template <typename Derived>
GaussNewtonOperatorFactoryBase<Derived>::GaussNewtonOperatorFactoryBase(const RCP<const Epetra_MultiVector> &reducedBasis, bool PsiEqualsPhi, bool runWithQR) :
reducedBasis_(reducedBasis),
jacobianFactory_(reducedBasis_),
PsiEqualsPhi_(PsiEqualsPhi),
runWithQR_(runWithQR)
{
	// Nothing to do

	//Create scaling vector
	scaling_ = Teuchos::rcp(new Epetra_MultiVector(reducedBasis->Map(), 1, true));
	//set initial scaling to 1 in case used before computed
	scaling_->PutScalar(1.0);

#if invJacPrec
	preconditioner_ = Teuchos::rcp(new Epetra_MultiVector(reducedBasis->Map(), reducedBasis->GlobalLength(), true));
	int num_rows = preconditioner_->MyLength();
	int num_vecs = preconditioner_->NumVectors();
	if (reducedBasis_->Comm().MyPID() == 0)
		std::cout << "preconditioner has " << num_rows << " rows and " << num_vecs << " columns" std::endl;
	//set initial preconditioner to identity matrix in case used before computed
	for (int ind_rows = 0; ind_rows < num_rows; ind_rows++)
		preconditioner_->ReplaceMyValue(ind_rows, ind_rows, 1.0);
	//preconditioner_->ReplaceGlobalValue(ind_rows, ind_rows, 1.0);
#endif //invJacPrec

	if (PsiEqualsPhi_)
		leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.rightProjector()));
	else //PsiEqualsPhi
		leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.premultipliedRightProjector()));

	if (runWithQR_)
	{
		// QR Initialization
		int num_vecs_tot;
		num_vecs_tot = jacobianFactory_.premultipliedRightProjector()->NumVectors();

		// Create a view of the Psi such that when we edit jacphi_int_,
		// we modify Psi as well (and vice versa).
		jacphi_int_ = Teuchos::rcp(new Epetra_MultiVector(View,*jacobianFactory_.premultipliedRightProjector(),0,num_vecs_tot));

		Q_ = Teuchos::rcp(new Epetra_MultiVector(*jacphi_int_));
		R_ = Teuchos::rcp(new Teuchos::SerialDenseMatrix<int,double> (num_vecs_tot,num_vecs_tot));

		tsqr_params_ = Teuchos::rcp(new Teuchos::ParameterList());

		////////////////  FROM THE TRILINOS CODE...  //////////////////////
		// TSQR (Tall Skinny QR factorization) is an orthogonalization
		// kernel that is as accurate as Householder QR, yet requires only
		// \f$2 \log P\f$ messages between $P$ MPI processes, independently
		// of the number of columns in the multivector.
		tsqr_adaptor_ = Teuchos::rcp(new Epetra::TsqrAdaptor(tsqr_params_));
	} //runWithQR
}

template <typename Derived>
bool GaussNewtonOperatorFactoryBase<Derived>::fullJacobianRequired(bool residualRequested, bool jacobianRequested) const {
	return residualRequested || jacobianRequested;
}

template <typename Derived>
const Epetra_MultiVector &GaussNewtonOperatorFactoryBase<Derived>::leftProjection(
		const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
	//parOut("    Computes psi^T*res");
	int err = reduce(*this->getMyLeftBasis(), fullVec, result);
	TEUCHOS_TEST_FOR_EXCEPT(err != 0);
	return result;
}

template <typename Derived>
const Epetra_MultiVector &GaussNewtonOperatorFactoryBase<Derived>::leftProjection_ProjectedSol(
		const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
	//parOut("    Computes phi^T*res");
	const int err = reduce(*this->getReducedBasis(), fullVec, result);
	TEUCHOS_TEST_FOR_EXCEPT(err != 0);
	return result;
}

template <typename Derived>
RCP<Epetra_CrsMatrix> GaussNewtonOperatorFactoryBase<Derived>::reducedJacobianNew() {
	return jacobianFactory_.reducedMatrixNew();
}

template <typename Derived>
const Epetra_CrsMatrix &GaussNewtonOperatorFactoryBase<Derived>::reducedJacobian(Epetra_CrsMatrix &result) const {
	return jacobianFactory_.reducedMatrix(*this->getMyLeftBasis(), result);
}

template <typename Derived>
const Epetra_CrsMatrix &GaussNewtonOperatorFactoryBase<Derived>::reducedJacobian_ProjectedSol(Epetra_CrsMatrix &result) const {

	Epetra_Vector ones = Epetra_Vector(result.RowMap(), true);
	ones.PutScalar(1.0);

	result.PutScalar(0.0);
	int err = result.ReplaceDiagonalValues(ones);
	TEUCHOS_TEST_FOR_EXCEPTION(err!=0, std::runtime_error, "Error in replacing diagonal values.\n");
	return result;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::fullJacobianIs(const Epetra_Operator &op) {

	jacobianFactory_.fullJacobianIs(op);
	if (PsiEqualsPhi_)
		leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.rightProjector()));
	else //PsiEqualsPhi
		leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.premultipliedRightProjector()));

	if (runWithQR_)
	{
		Teuchos::RCP<Epetra_MultiVector> A = Teuchos::rcp(new Epetra_MultiVector(*jacphi_int_));

		//EpetraExt::MultiVectorToMatrixMarketFile("A.mm", *A); (if you want to output the data)

		// NOTE: this is NOT the same as [Q,R] = qr(A,0) in Matlab, even in serial
		// (even though the documentation in Trilinos might lead you to think it is)
		tsqr_adaptor_->factorExplicit(*A,*Q_,*R_);

		leftbasis_->Scale(1.0, *Q_);

		/* (if you want to output the data)
		EpetraExt::MultiVectorToMatrixMarketFile("Q.mm", *Q_);
		if (reducedBasis_->Comm().MyPID() == 0)
			std::cout << "R = " << R_ << std::endl;
		int num_vecs_tot = jacobianFactory_.premultipliedRightProjector()->NumVectors();
		for (int i=0; i<num_vecs_tot; i++){
			for (int j=0; j<num_vecs_tot-1; j++){
				if (reducedBasis_->Comm().MyPID() == 0)
					std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*R_)(i,j) << ", ";
			}
			if (reducedBasis_->Comm().MyPID() == 0)
				std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*R_)(i,num_vecs_tot-1) << std::endl;
		}
		*/
	} //runWithQR
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getPremultipliedReducedBasis() const {
	return jacobianFactory_.premultipliedRightProjector();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getReducedBasis() const {
	return reducedBasis_;
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getScaling() const
{
	return scaling_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setScaling(Epetra_CrsMatrix &jacobian) const
{
	parOut("Computing Scaling Factors");
	int num_rows = scaling_->MyLength();
	int num_vecs = scaling_->NumVectors();
	TEUCHOS_ASSERT(num_rows == jacobian.NumMyRows());
	double** data_array = NULL;
	scaling_->ExtractView(&data_array);

	for (int ind_rows = 0; ind_rows < num_rows; ind_rows++)
	{
		int value_count;
		double* row_values;
		int* value_indices;
		const int err = jacobian.ExtractMyRowView(ind_rows,value_count,row_values,value_indices);
		TEUCHOS_ASSERT(err == 0);

		//Scale by row sum
		/*double sum = 0.0;
    for (int ind_vals = 0; ind_vals < value_count; ind_vals++)
    {
      sum += std::abs(row_values[ind_vals]);
      //sum += row_values[ind_vals];
    }
    data_array[0][ind_rows] = 1.0/sum;*/

		//Scale by diagonal entry
		data_array[0][ind_rows] = 1.0;
		for (int ind_vals = 0; ind_vals < value_count; ind_vals++)
		{
			//if (reducedBasis_->Comm().MyPID() == 0)
			//	printf(" %d : %d < %d : %d\n",ind_rows,ind_vals,value_count,value_indices[ind_vals]);
			if (value_indices[ind_vals] == ind_rows)
			{
				data_array[0][ind_rows] = 1.0/row_values[ind_vals];
			}
		}
	}

}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyScaling(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Scaling Factors");
	int num_rows = vector.MyLength();
	int num_vecs = vector.NumVectors();
	TEUCHOS_ASSERT(num_rows == scaling_->MyLength());
	double** data_array = NULL;

	vector.ExtractView(&data_array);

	for (int ind_vecs = 0; ind_vecs < num_vecs; ind_vecs++)
	{
		for (int ind_rows = 0; ind_rows < num_rows; ind_rows++)
		{
			data_array[ind_vecs][ind_rows] *= (*scaling_)[0][ind_rows];
		}
	}

}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getPreconditioner() const
{
	return preconditioner_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setPreconditionerDirectly(Epetra_MultiVector &vector) const
{
	preconditioner_->Update(1.0,vector,0.0);
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setPreconditioner(Epetra_CrsMatrix &jacobian) const
{
	parOut("Computing Preconditioner");

	preconditioner_->PutScalar(0.0);
	Teuchos::RCP<Epetra_MultiVector> identity = Teuchos::rcp(new Epetra_MultiVector(*preconditioner_));

	int num_rows = identity->GlobalLength();
	int num_vecs = identity->NumVectors();
	TEUCHOS_ASSERT(num_rows == num_vecs);

	identity->PutScalar(0.0);
	int *grows = identity->Map().MyGlobalElements();
	for (int local_row = 0; local_row < identity->MyLength(); local_row++)
	{
		int global_row = grows[local_row];
		identity->ReplaceMyValue(local_row, global_row, 1.0);
	}

	{
		parOut("  start");
		Epetra_CrsMatrix* aaa = &jacobian;
		Epetra_MultiVector* xxx = new Epetra_MultiVector(View,*preconditioner_,0,preconditioner_->NumVectors());
		Epetra_MultiVector* bbb = new Epetra_MultiVector(View,*identity,0,identity->NumVectors());
		Epetra_LinearProblem problem(aaa,xxx,bbb);
		Amesos_BaseSolver* solver;
		Amesos factory;
		std::string solvertype = "Klu";
		solver = factory.Create(solvertype, problem);
		TEUCHOS_TEST_FOR_EXCEPTION(solver!=0, std::runtime_error, "Solver type" + solvertype + "is not available.\n");
		Teuchos::ParameterList list;
		list.set("PrintTiming",true);
		list.set("PrintStatus",true);
		solver->SetParameters(list);
		int ierr;
		ierr = solver->SymbolicFactorization();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling SymbolicFactorization.\n");
		ierr = solver->NumericFactorization();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling NumericFactorization.\n");
		ierr = solver->Solve();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling Solve.\n");
		delete solver;
		parOut("  finish");
	}
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditioner(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Preconditioner");
	Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

	int err = temp2->Multiply('N','N',1.0,*preconditioner_,*temp,0.0);
	std::string errstr = "The multiplication failed in applyPreconditioner. We're trying to multiply a preconditioner of size "
										 + std::to_string(preconditioner_->GlobalLength()) + "x" + std::to_string(preconditioner_->NumVectors())
										 + " with a matrix of size "
										 + std::to_string(vector.GlobalLength()) + "x" + std::to_string(vector.NumVectors())
										 + "\nNOTE: this function shouldn't be used in parallel... use ReducedOrderModelEvaluator::multiplyInPlace instead!!\n";
	TEUCHOS_TEST_FOR_EXCEPTION(err!=0, std::runtime_error, errstr);
	//temp2->Multiply('T','N',1.0,*preconditioner_,*temp,0.0);

}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditionerTwice(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Preconditioner, TWICE");
	Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp3 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

	int err = temp2->Multiply('N','N',1.0,*preconditioner_,*temp,0.0);
	TEUCHOS_TEST_FOR_EXCEPTION(err!=0, std::runtime_error, "The first multiplication failed in applyPreconditioner\n");
	err = temp3->Multiply('T','N',1.0,*preconditioner_,*temp2,0.0);
	TEUCHOS_TEST_FOR_EXCEPTION(err!=0, std::runtime_error, "The second multiplication failed in applyPreconditioner\n");

}

template <typename Derived>
RCP<Ifpack_Preconditioner> GaussNewtonOperatorFactoryBase<Derived>::getPreconditionerIfpack() const
{
	return preconditioner_ifpack_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setPreconditionerIfpack(Epetra_CrsMatrix &jacobian, std::string ifpackType) const
{
	parOut("Computing Preconditioner (Ifpack)");

	{
		parOut("  start");
		Epetra_CrsMatrix* aaa = &jacobian;
		Ifpack PrecFactory;
		std::string PrecType;
		Teuchos::ParameterList List;
		if (ifpackType.compare("Jacobi") == 0)
		{
			PrecType = "point relaxation";
			List.set("relaxation: type","Jacobi");  //default
		}
		else if (ifpackType.compare("GaussSeidel") == 0)
		{
			PrecType = "point relaxation";
			List.set("relaxation: type","Gauss-Seidel");
		}
		else if (ifpackType.compare("SymmetricGaussSeidel") == 0)
		{
			PrecType = "point relaxation";
			List.set("relaxation: type","symmetric Gauss-Seidel");
		}
		else if (ifpackType.compare("ILU0") == 0)
		{
			PrecType = "ILU";
			List.set("fact: level-of-fill",0);
		}
		else if (ifpackType.compare("ILU1") == 0)
		{
			PrecType = "ILU";
			List.set("fact: level-of-fill",1);
		}
		else if (ifpackType.compare("ILU2") == 0)
		{
			PrecType = "ILU";
			List.set("fact: level-of-fill",2);
		}
		else if (ifpackType.compare("IC0") == 0)
		{
			PrecType = "IC";
			List.set("fact: ict level-of-fill",0.0);
		}
		else if (ifpackType.compare("IC1") == 0)
		{
			PrecType = "IC";
			List.set("fact: ict level-of-fill",1.0);
		}
		else if (ifpackType.compare("IC2") == 0)
		{
			PrecType = "IC";
			List.set("fact: ict level-of-fill",2.0);
		}
		else if (ifpackType.compare("Amesos") == 0)
		{
			PrecType = "Amesos";
			//List.set("amesos: solver type","Amesos_Dscpack"); //Amesos_Klu
		}
		else if (ifpackType.compare("Identity") == 0)
		{
			PrecType = "Chebyshev";
		}

		int OverlapLevel = 0; // must be >= 0. If Comm.NumProc() == 1, it is ignored.
		preconditioner_ifpack_ = Teuchos::rcp(PrecFactory.Create(PrecType, aaa, OverlapLevel));
		TEUCHOS_ASSERT(preconditioner_ifpack_ != Teuchos::null);

		int err;
		err = preconditioner_ifpack_->SetParameters(List);
		TEUCHOS_ASSERT(err == 0);
		err = preconditioner_ifpack_->Initialize();
		TEUCHOS_ASSERT(err == 0);
		err = preconditioner_ifpack_->Compute();
		TEUCHOS_ASSERT(err == 0);

		TEUCHOS_ASSERT(preconditioner_ifpack_->IsInitialized() == true);
		TEUCHOS_ASSERT(preconditioner_ifpack_->IsComputed() == true);

		/*
		if (reducedBasis_->Comm().MyPID() == 0)
		{
			std::cout << preconditioner_ifpack_->NumInitialize() << std::endl;
			std::cout << preconditioner_ifpack_->NumCompute() << std::endl;
			std::cout << preconditioner_ifpack_->NumApplyInverse() << std::endl;
			std::cout << *preconditioner_ifpack_ << std::endl;
		}
		*/
		parOut("  finish");
	}
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditionerIfpack(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Preconditioner");
	Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

	TEUCHOS_ASSERT(preconditioner_ifpack_ != Teuchos::null);
	TEUCHOS_ASSERT(preconditioner_ifpack_->IsInitialized() == true);
	TEUCHOS_ASSERT(preconditioner_ifpack_->IsComputed() == true);
	/*
	if (reducedBasis_->Comm().MyPID() == 0)
	{
		std::cout << preconditioner_ifpack_->NumInitialize() << std::endl;
		std::cout << preconditioner_ifpack_->NumCompute() << std::endl;
		std::cout << preconditioner_ifpack_->NumApplyInverse() << std::endl;
		std::cout << preconditioner_ifpack_ << std::endl;
		std::cout << *preconditioner_ifpack_ << std::endl;
	}
	*/
	int err = preconditioner_ifpack_->ApplyInverse(*temp,*temp2);

	std::string errstr = "The preconditioner application failed in applyPreconditionerIfpack (error code " + std::to_string(err) + ")\n";
	TEUCHOS_TEST_FOR_EXCEPTION(err!=0, std::runtime_error, errstr);
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditionerIfpackTwice(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Preconditioner");
	Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(vector));
	Teuchos::RCP<Epetra_MultiVector> temp3 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

	TEUCHOS_ASSERT(preconditioner_ifpack_ != Teuchos::null);
	TEUCHOS_ASSERT(preconditioner_ifpack_->IsInitialized() == true);
	TEUCHOS_ASSERT(preconditioner_ifpack_->IsComputed() == true);

	/*
	if (reducedBasis_->Comm().MyPID() == 0)
	{
		std::cout << preconditioner_ifpack_->NumInitialize() << std::endl;
		std::cout << preconditioner_ifpack_->NumCompute() << std::endl;
		std::cout << preconditioner_ifpack_->NumApplyInverse() << std::endl;
		std::cout << preconditioner_ifpack_ << std::endl;
		std::cout << *preconditioner_ifpack_ << std::endl;
	}
	*/

	preconditioner_ifpack_->ApplyInverse(*temp,*temp2);
	int err = preconditioner_ifpack_->SetUseTranspose(true);
	if (err==-1)
		parOut("This Ifpack implementation doesn't support using a transposed preconditioner!");
	TEUCHOS_ASSERT(err == 0);
	preconditioner_ifpack_->ApplyInverse(*temp2,*temp3);
}

template <typename Derived>
RCP<const Epetra_CrsMatrix> GaussNewtonOperatorFactoryBase<Derived>::getJacobian() const
{
	return jacobian_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setJacobian(Epetra_CrsMatrix &jacobian) const
{
	parOut("Copying Jacobian for Projected Solution");

	jacobian_ = Teuchos::rcp(new Epetra_CrsMatrix(jacobian));

}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyJacobian(const Epetra_MultiVector &vector) const
{
	//NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
	//       but the values of the MultiVector are modified.
	parOut("Applying Jacobian");
	{
		parOut("  start");
		Epetra_CrsMatrix* aaa = jacobian_.getRawPtr();
		Epetra_MultiVector* bbb = new Epetra_MultiVector(vector);
		Epetra_MultiVector* xxx = new Epetra_MultiVector(View, vector, 0, vector.NumVectors());
		Epetra_LinearProblem problem(aaa,xxx,bbb);
		Amesos_BaseSolver* solver;
		Amesos factory;
		std::string solvertype = "Klu";
		//std::string solvertype = "Superludist";
		solver = factory.Create(solvertype, problem);
		TEUCHOS_TEST_FOR_EXCEPTION(solver == 0, std::runtime_error, "Specified solver is not available\n");
		Teuchos::ParameterList list;
		list.set("PrintTiming",true);
		list.set("PrintStatus",true);
		solver->SetParameters(list);
		int ierr;
		ierr = solver->SymbolicFactorization();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling SymbolicFactorization.\n");
		ierr = solver->NumericFactorization();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling NumericFactorization.\n");
		ierr = solver->Solve();
		TEUCHOS_TEST_FOR_EXCEPTION(ierr!=0, std::runtime_error, "Error when calling Solve.\n");
		delete solver;
		parOut("  finish");
		delete xxx;
		delete bbb;
	}
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getMyLeftBasis() const {
	return static_cast<const Derived *>(this)->leftProjectorBasis();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasis() const {
	return leftbasis_;
}

GaussNewtonOperatorFactory::GaussNewtonOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis, bool PsiEqualsPhi, bool runWithQR) :
		  GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory>(reducedBasis, PsiEqualsPhi, runWithQR)
		  {
	// Nothing to do
		  }

RCP<const Epetra_MultiVector> GaussNewtonOperatorFactory::leftProjectorBasis() const {
	return this->getLeftBasis();
}

GaussNewtonMetricOperatorFactory::GaussNewtonMetricOperatorFactory(
		const RCP<const Epetra_MultiVector> &reducedBasis,
		const Teuchos::RCP<const Epetra_Operator> &metric,
		bool PsiEqualsPhi,
		bool runWithQR) :
		GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>(reducedBasis, PsiEqualsPhi, runWithQR),
		metric_(metric),
		premultipliedLeftProjector_(new Epetra_MultiVector(metric->OperatorDomainMap(), reducedBasis->NumVectors(), false))
		{
	this->updatePremultipliedLeftProjector();
		}

RCP<const Epetra_MultiVector> GaussNewtonMetricOperatorFactory::leftProjectorBasis() const {
	return this->premultipliedLeftProjector_;
}

void GaussNewtonMetricOperatorFactory::fullJacobianIs(const Epetra_Operator &op) {
	this->GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>::fullJacobianIs(op);
	this->updatePremultipliedLeftProjector();
}

void GaussNewtonMetricOperatorFactory::updatePremultipliedLeftProjector() {
	int err = 0;
	err = metric_->Apply(*this->getLeftBasis(), *premultipliedLeftProjector_);
	TEUCHOS_TEST_FOR_EXCEPT(err != 0);
}

} // namespace MOR
