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
