//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ProjectionErrorObserver.hpp"

namespace MOR {

ProjectionErrorObserver::ProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile) :
  projectionError_(projectionSpace, errorFile)
{
   // Nothing to do
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution)
{
  projectionError_.process(solution);
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  projectionError_.process(solution);
}

} // namespace MOR
