#include "ATOT_XFEM_Preconditioner.hpp"
#include "Albany_Utils.hpp"
#include <string>


namespace ATOT {
namespace XFEM {

using namespace Teuchos;

/*******************************************************************************/
Preconditioner::Preconditioner(const RCP<const Teuchos::ParameterList> params) {}
/*******************************************************************************/

/*******************************************************************************/
int Preconditioner::BuildPreconditioner(
         const RCP<Tpetra_CrsMatrix>& jacT, 
         const RCP<Albany::AbstractDiscretization>& disc,
         const Albany::StateManager& stateMgr)
/*******************************************************************************/
{
  domainMap_ = jacT->getDomainMap();
  rangeMap_  = jacT->getRangeMap(); 

  // initially this is a diagonal matrix
  int numEntriesPerRow = 1;
  
  operator_ = rcp(new Tpetra_CrsMatrix(jacT->getRowMap(), jacT->getColMap(), numEntriesPerRow, Tpetra::ProfileType::StaticProfile));
  
  auto nRows = operator_->getNodeNumRows();
  double zero = 0.0;
  //IKT: err is not set anywhere b/c fillComplete and insertLocalValues routines are 
  //void, not int, in Tpetra.
  int err;
  for(int iRow=0; iRow<nRows; ++iRow)
    operator_->insertLocalValues(iRow,numEntriesPerRow,&zero,&iRow);
  operator_->fillComplete();
  
  // for now, just do inverse row sum.  Later, we'll hook in Cogent.
  RCP<Tpetra_Vector> invRowSums = Albany::InvRowSum(jacT);
  Albany::ReplaceDiagonalEntries(operator_, invRowSums);

  invOperator_ = rcp(new Tpetra_CrsMatrix(jacT->getRowMap(), jacT->getColMap(), numEntriesPerRow, Tpetra::ProfileType::StaticProfile));
  invOperator_->fillComplete();
  
  RCP<Tpetra_Vector> rowSums = rcp(new Tpetra_Vector(*invRowSums));
  rowSums->reciprocal(*invRowSums);
  Albany::ReplaceDiagonalEntries(invOperator_, rowSums);
  return 0; 
}


/*******************************************************************************/
void Preconditioner::apply(
      Tpetra_MultiVector const & X,
      Tpetra_MultiVector & Y,
      Teuchos::ETransp mode, ST alpha, ST beta) const
/*******************************************************************************/
{ 
  invOperator_->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0); 
}

} // end namespace XFEM
} // end namespace ATOT

