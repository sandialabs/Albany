//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_GeneralizedCoordinatesNOXObserver.hpp"

namespace MOR {

GeneralizedCoordinatesNOXObserver::GeneralizedCoordinatesNOXObserver(const std::string filename) :
  impl_(filename)
{
  // Nothing to do
}

void
GeneralizedCoordinatesNOXObserver::observeSolution(const Epetra_Vector& solution)
{
  impl_.vectorAdd(solution);
}

void
GeneralizedCoordinatesNOXObserver::observeSolution(const Epetra_Vector& solution, double /*time_or_param_val*/)
{
  // TODO: Handle stamps
  impl_.vectorAdd(solution);
}

} // end namespace MOR
