//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_PROJECTIONERROR_HPP
#define MOR_PROJECTIONERROR_HPP

#include "Teuchos_RCP.hpp"

#include "Epetra_MultiVector.h"

#include <deque>

namespace MOR {

class ReducedSpace;
class MultiVectorOutputFile;

class ProjectionError {
public:
  ProjectionError(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile);

  const Epetra_Comm &projectionBasisComm() const;

  ~ProjectionError();

  void process(const Epetra_MultiVector &v);

private:
  Teuchos::RCP<ReducedSpace> projectionSpace_;
  Teuchos::RCP<MultiVectorOutputFile> errorFile_;

  std::deque<double> relativeErrorNorms_;

  // Disallow copy & assignment
  ProjectionError(const ProjectionError &);
  ProjectionError &operator=(const ProjectionError &);
};

} // namespace MOR

#endif /* MOR_PROJECTIONERROR_HPP */
