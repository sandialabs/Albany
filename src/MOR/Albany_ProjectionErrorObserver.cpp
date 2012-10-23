//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProjectionErrorObserver.hpp"

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

ProjectionErrorObserver::ProjectionErrorObserver(const RCP<ParameterList> &params,
                                                 const Teuchos::RCP<NOX::Epetra::Observer>& decoratedObserver,
                                                 const RCP<const Epetra_Map> &decoratedMap) :
  decoratedObserver_(decoratedObserver),
  projectionError_(params, decoratedMap)
{
   // Nothing to do
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution)
{
  decoratedObserver_->observeSolution(solution);
  projectionError_.process(solution);
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  decoratedObserver_->observeSolution(solution, time_or_param_val);
  projectionError_.process(solution);
}

} // end namespace Albany
