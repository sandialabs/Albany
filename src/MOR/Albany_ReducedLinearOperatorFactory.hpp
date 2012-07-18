/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#ifndef ALBANY_REDUCEDLINEAROPERATORFACTORY_HPP
#define ALBANY_REDUCEDLINEAROPERATORFACTORY_HPP

#include "Epetra_CrsGraph.h"

class Epetra_Operator;
class Epetra_CrsMatrix;
class Epetra_MultiVector;

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedLinearOperatorFactory {
public:
  explicit ReducedLinearOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &projector);

  ReducedLinearOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &rightProjector,
                               const Teuchos::RCP<const Epetra_MultiVector> &leftProjector);

  Teuchos::RCP<Epetra_CrsMatrix> reducedOperatorNew() const;
  Teuchos::RCP<Epetra_CrsMatrix> reducedOperatorNew(const Epetra_Operator &fullOperator) const;

  void reducedOperatorInit(const Epetra_Operator &fullOperator, Epetra_CrsMatrix &result) const;

private:
  Teuchos::RCP<const Epetra_MultiVector> rightProjector_;
  Teuchos::RCP<const Epetra_MultiVector> leftProjector_;

  Epetra_CrsGraph reducedGraph_;

  void init();

  bool isMasterProcess() const;
};

} // namespace Albany

#endif /* ALBANY_REDUCEDLINEAROPERATORFACTORY_HPP */
