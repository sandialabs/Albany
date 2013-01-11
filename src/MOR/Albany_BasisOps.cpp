#include "Albany_BasisOps.hpp"

#include "Epetra_LocalMap.h"

namespace Albany {

Epetra_LocalMap createComponentMap(const Epetra_MultiVector &projector) {
  return Epetra_LocalMap(projector.NumVectors(), 0, projector.Comm());
}

} // end namespace Albany
