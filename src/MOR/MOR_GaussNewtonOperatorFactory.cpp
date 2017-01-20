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

namespace MOR {

using ::Teuchos::RCP;

template <typename Derived>
GaussNewtonOperatorFactoryBase<Derived>::GaussNewtonOperatorFactoryBase(const RCP<const Epetra_MultiVector> &reducedBasis, int numDBCModes) :
  reducedBasis_(reducedBasis),
  jacobianFactory_(reducedBasis_)
  ,num_dbc_modes_(numDBCModes)
{
  // Nothing to do

  //Create scaling vector
  scaling_ = Teuchos::rcp(new Epetra_MultiVector(reducedBasis->Map(), 1, true));
  //set initial scaling to 1 in case used before computed
  scaling_->PutScalar(1.0);

/*  preconditioner_ = Teuchos::rcp(new Epetra_MultiVector(reducedBasis->Map(), reducedBasis->GlobalLength(), true));
  int num_rows = preconditioner_->MyLength();
  int num_vecs = preconditioner_->NumVectors();
  printf("preconditioner has %d rows and %d columns\n",num_rows,num_vecs);
  //set initial preconditioner to identity matrix in case used before computed
  for (int ind_rows = 0; ind_rows < num_rows; ind_rows++)
    preconditioner_->ReplaceMyValue(ind_rows, ind_rows, 1.0);
    //preconditioner_->ReplaceGlobalValue(ind_rows, ind_rows, 1.0);

  leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.premultipliedRightProjector()));*/
}

template <typename Derived>
bool GaussNewtonOperatorFactoryBase<Derived>::fullJacobianRequired(bool residualRequested, bool jacobianRequested) const {
  return residualRequested || jacobianRequested;
}

template <typename Derived>
const Epetra_MultiVector &GaussNewtonOperatorFactoryBase<Derived>::leftProjection(
    const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
//printf("    Computes psi^T*res\n");
  int err = 0;
  if (num_dbc_modes_ == 0)
    err = reduce(*this->getLeftBasis(), fullVec, result);
  else
    err = reduce(*this->getLeftBasisCopy(), fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

template <typename Derived>
const Epetra_MultiVector &GaussNewtonOperatorFactoryBase<Derived>::leftProjection_ProjectedSol(
    const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
//printf("    Computes phi^T*res\n");
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
  if (num_dbc_modes_ == 0)
    return jacobianFactory_.reducedMatrix(*this->getLeftBasis(), result);
  else
    return jacobianFactory_.reducedMatrix(*this->getLeftBasisCopy(), result);
}

template <typename Derived>
const Epetra_CrsMatrix &GaussNewtonOperatorFactoryBase<Derived>::reducedJacobian_ProjectedSol(Epetra_CrsMatrix &result) const {

  Epetra_Vector ones = Epetra_Vector(result.RowMap(), true);
  ones.PutScalar(1.0);

  result.PutScalar(0.0);
  int err = 0;
  err = result.ReplaceDiagonalValues(ones);
  if (err != 0)
    printf("error in replacing diagonal values\n");
  return result;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::fullJacobianIs(const Epetra_Operator &op) {

  jacobianFactory_.fullJacobianIs(op);
  leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.premultipliedRightProjector()));

  //printf("using %d DBC modes\n",num_dbc_modes_);
  if (num_dbc_modes_ > 0)
  {
    Epetra_MultiVector* psi_dbc = new Epetra_MultiVector(View,*leftbasis_,0,num_dbc_modes_);
  //psi_dbc->Print(std::cout);
    Epetra_MultiVector* phi_dbc = new Epetra_MultiVector(View,*jacobianFactory_.rightProjector(),0,num_dbc_modes_);
    psi_dbc->Scale(1.0, *phi_dbc);
  //psi_dbc->Print(std::cout);
    delete psi_dbc;
    delete phi_dbc;
  }
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
  printf("Computing Scaling Factors\n");
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
      //printf(" %d : %d < %d : %d\n",ind_rows,ind_vals,value_count,value_indices[ind_vals]);
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
  printf("Applying Scaling Factors\n");
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
void GaussNewtonOperatorFactoryBase<Derived>::setPreconditioner(Epetra_CrsMatrix &jacobian) const
{
  printf("Computing Preconditioner\n");

  preconditioner_->PutScalar(0.0);
  Teuchos::RCP<Epetra_MultiVector> identity = Teuchos::rcp(new Epetra_MultiVector(*preconditioner_));


  int num_rows = identity->MyLength();
  int num_vecs = identity->NumVectors();
  TEUCHOS_ASSERT(num_rows == num_vecs);

  identity->PutScalar(0.0);
  for (int ind_rows = 0; ind_rows < num_rows; ind_rows++)
    identity->ReplaceGlobalValue(ind_rows, ind_rows, 1.0);
  //identity->Print(std::cout);

  {
    printf("  start\n");
    Epetra_CrsMatrix* aaa = &jacobian;
    Epetra_MultiVector* xxx = new Epetra_MultiVector(View,*preconditioner_,0,preconditioner_->NumVectors());
    Epetra_MultiVector* bbb = new Epetra_MultiVector(View,*identity,0,identity->NumVectors());
    Epetra_LinearProblem problem(aaa,xxx,bbb);
    Amesos_BaseSolver* solver;
    Amesos factory;
    std::string solvertype = "Klu";
    solver = factory.Create(solvertype, problem);
    if (solver == 0)
      std::cerr << "Specified solver is not available\n";
    Teuchos::ParameterList list;
    list.set("PrintTiming",true);
    list.set("PrintStatus",true);
    solver->SetParameters(list);
    int ierr;
    ierr = solver->SymbolicFactorization();
    if (ierr > 0) std::cerr << "Error when calling SymbolicFactorization.\n";
    ierr = solver->NumericFactorization();
    if (ierr > 0) std::cerr << "Error when calling NumericFactorization.\n";
    ierr = solver->Solve();
    if (ierr > 0) std::cerr << "Error when calling Solve.\n";
    delete solver;
    printf("  finish\n");
  }
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditioner(const Epetra_MultiVector &vector) const
{
  //NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
  //       but the values of the MultiVector are modified.
  printf("Applying Preconditioner\n");
  Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
  Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

  temp2->Multiply('N','N',1.0,*preconditioner_,*temp,0.0);
  //temp2->Multiply('T','N',1.0,*preconditioner_,*temp,0.0);

}

template <typename Derived>
RCP<Ifpack_Preconditioner> GaussNewtonOperatorFactoryBase<Derived>::getPreconditionerIfpack() const
{
  return preconditioner_ifpack_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setPreconditionerIfpack(Epetra_CrsMatrix &jacobian, std::string ifpackType) const
{
  printf("Computing Preconditioner\n");

  {
    printf("  start\n");
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

    preconditioner_ifpack_ = Teuchos::rcp(PrecFactory.Create(PrecType,aaa));
    TEUCHOS_ASSERT(preconditioner_ifpack_ != Teuchos::null);

    int err;
    err = preconditioner_ifpack_->SetParameters(List);
    std::cout << err << std::endl;
    TEUCHOS_ASSERT(err == 0);
    err = preconditioner_ifpack_->Initialize();
    std::cout << err << std::endl;
    TEUCHOS_ASSERT(err == 0);
    err = preconditioner_ifpack_->Compute();
    std::cout << err << std::endl;
    TEUCHOS_ASSERT(err == 0);

    TEUCHOS_ASSERT(preconditioner_ifpack_->IsInitialized() == true);
    TEUCHOS_ASSERT(preconditioner_ifpack_->IsComputed() == true);

    std::cout << preconditioner_ifpack_->NumInitialize() << std::endl;
    std::cout << preconditioner_ifpack_->NumCompute() << std::endl;
    std::cout << preconditioner_ifpack_->NumApplyInverse() << std::endl;
    std::cout << *preconditioner_ifpack_ << std::endl;
    //preconditioner_ifpack_->Print(std::cout);
    printf("  finish\n");
  }
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyPreconditionerIfpack(const Epetra_MultiVector &vector) const
{
  //NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
  //       but the values of the MultiVector are modified.
  printf("Applying Preconditioner\n");
  Teuchos::RCP<Epetra_MultiVector> temp = Teuchos::rcp(new Epetra_MultiVector(vector));
  Teuchos::RCP<Epetra_MultiVector> temp2 = Teuchos::rcp(new Epetra_MultiVector(View, vector, 0, vector.NumVectors()));

  TEUCHOS_ASSERT(preconditioner_ifpack_ != Teuchos::null);
  TEUCHOS_ASSERT(preconditioner_ifpack_->IsInitialized() == true);
  TEUCHOS_ASSERT(preconditioner_ifpack_->IsComputed() == true);

  std::cout << preconditioner_ifpack_->NumInitialize() << std::endl;
  std::cout << preconditioner_ifpack_->NumCompute() << std::endl;
  std::cout << preconditioner_ifpack_->NumApplyInverse() << std::endl;
  std::cout << preconditioner_ifpack_ << std::endl;
  std::cout << *preconditioner_ifpack_ << std::endl;

  preconditioner_ifpack_->ApplyInverse(*temp,*temp2);
}

template <typename Derived>
RCP<const Epetra_CrsMatrix> GaussNewtonOperatorFactoryBase<Derived>::getJacobian() const
{
  return jacobian_;
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::setJacobian(Epetra_CrsMatrix &jacobian) const
{
  printf("Copying Jacobian for Projected Solution\n");

  jacobian_ = Teuchos::rcp(new Epetra_CrsMatrix(jacobian));

}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::applyJacobian(const Epetra_MultiVector &vector) const
{
  //NOTE:  vector is "const" in the sense that this function does not change what is pointed to,
  //       but the values of the MultiVector are modified.
  printf("Applying Jacobian\n");
  {
    printf("  start\n");
    Epetra_CrsMatrix* aaa = jacobian_.getRawPtr();
    Epetra_MultiVector* bbb = new Epetra_MultiVector(vector);
    Epetra_MultiVector* xxx = new Epetra_MultiVector(View, vector, 0, vector.NumVectors());
    Epetra_LinearProblem problem(aaa,xxx,bbb);
    Amesos_BaseSolver* solver;
    Amesos factory;
    std::string solvertype = "Klu";
    solver = factory.Create(solvertype, problem);
    if (solver == 0)
      std::cerr << "Specified solver is not available\n";
    Teuchos::ParameterList list;
    list.set("PrintTiming",true);
    list.set("PrintStatus",true);
    solver->SetParameters(list);
    int ierr;
    ierr = solver->SymbolicFactorization();
    if (ierr > 0) std::cerr << "Error when calling SymbolicFactorization.\n";
    ierr = solver->NumericFactorization();
    if (ierr > 0) std::cerr << "Error when calling NumericFactorization.\n";
    ierr = solver->Solve();
    if (ierr > 0) std::cerr << "Error when calling Solve.\n";
    delete solver;
    printf("  finish\n");
    delete xxx;
    delete bbb;
  }
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasis() const {
  return static_cast<const Derived *>(this)->leftProjectorBasis();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasisCopy() const {
  return leftbasis_;
}

GaussNewtonOperatorFactory::GaussNewtonOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis, int numDBCModes) :
  GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory>(reducedBasis, numDBCModes)
{
  // Nothing to do
}

RCP<const Epetra_MultiVector> GaussNewtonOperatorFactory::leftProjectorBasis() const {
  return this->getPremultipliedReducedBasis();
}

GaussNewtonMetricOperatorFactory::GaussNewtonMetricOperatorFactory(
    const RCP<const Epetra_MultiVector> &reducedBasis,
    const Teuchos::RCP<const Epetra_Operator> &metric) :
  GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>(reducedBasis, num_dbc_modes_),
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
  const int err = metric_->Apply(*this->getPremultipliedReducedBasis(), *premultipliedLeftProjector_);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
}

} // namespace MOR
