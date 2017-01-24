#include "ATO_XFEM_Preconditioner.hpp"

#include <string>

namespace ATO {
namespace XFEM {

using namespace Teuchos;

/*******************************************************************************/
Preconditioner::Preconditioner(const RCP<const Teuchos::ParameterList> params) {}
/*******************************************************************************/

/*******************************************************************************/
int Preconditioner::BuildPreconditioner(
         const RCP<Epetra_CrsMatrix>& jac, 
         const RCP<Albany::AbstractDiscretization>& disc,
         const Albany::StateManager& stateMgr)
/*******************************************************************************/
{
  domainMap_ = rcpFromRef(jac->DomainMap());
  rangeMap_  = rcpFromRef(jac->RangeMap());

  // initially this is a diagonal matrix
  int numEntriesPerRow = 1;
  // and it is static
  bool staticProfile = false;
  operator_ = rcp(new Epetra_CrsMatrix(Epetra_DataAccess::Copy, jac->RowMap(), jac->ColMap(), numEntriesPerRow, staticProfile));
  int nRows = operator_->NumMyRows();
  double zero = 0.0;
  int err;
  for(int iRow=0; iRow<nRows; ++iRow)
    err = operator_->InsertMyValues(iRow,numEntriesPerRow,&zero,&iRow);
  err = operator_->FillComplete();
 
  // for now, just do inverse row sum.  Later, we'll hook in Cogent.
  Epetra_Vector invRowSums(jac->RowMap());
  jac->InvRowSums(invRowSums);
  err = operator_->ReplaceDiagonalValues(invRowSums);

  invOperator_ = rcp(new Epetra_CrsMatrix(*operator_));
  invOperator_->FillComplete();
  Epetra_Vector rowSums(invRowSums);
  rowSums.Reciprocal(invRowSums);
  invOperator_->ReplaceDiagonalValues(rowSums);
  return 0; 
}


/*******************************************************************************/
int Preconditioner::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
/*******************************************************************************/
{ 
//  operator_->Multiply(/*transpose=*/false,X,Y);
  invOperator_->Multiply(/*transpose=*/false,X,Y);
  return 0;
}


/*******************************************************************************/
int Preconditioner::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
/*******************************************************************************/
{
//  invOperator_->Multiply(/*transpose=*/false,X,Y);
  operator_->Multiply(/*transpose=*/false,X,Y);
  return 0;
}


/*******************************************************************************/
double Preconditioner::NormInf() const
/*******************************************************************************/
{
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                       "ATO::XFEM::Preconditioner::NormInf not implemated");
    return 1.0;
}

} // end namespace XFEM
} // end namespace ATO

