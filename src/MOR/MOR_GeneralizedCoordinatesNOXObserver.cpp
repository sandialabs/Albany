//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_GeneralizedCoordinatesNOXObserver.hpp"

namespace MOR {

GeneralizedCoordinatesNOXObserver::GeneralizedCoordinatesNOXObserver(
    const std::string &filename,
    const std::string &stampsFilename) :
  impl_(filename, stampsFilename)
{
  // Nothing to do
}

void
GeneralizedCoordinatesNOXObserver::observeSolution(const Epetra_Vector& solution)
{
  impl_.vectorAdd(solution);
}

void
GeneralizedCoordinatesNOXObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  impl_.stampedVectorAdd(time_or_param_val, solution);
}

} // end namespace MOR
