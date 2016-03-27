//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_FullStateReconstructor.hpp"

#include "MOR_ReducedSpace.hpp"

namespace MOR {

FullStateReconstructor::FullStateReconstructor(
    const Teuchos::RCP<const ReducedSpace> &reducedSpace,
    const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver) :
  reducedSpace_(reducedSpace),
  decoratedObserver_(decoratedObserver),
  lastFullSolution_(reducedSpace->basisMap(), false)
{
  // Nothing to do
}

void FullStateReconstructor::observeSolution(const Epetra_Vector& solution)
{
  computeLastFullSolution(solution);
  decoratedObserver_->observeSolution(lastFullSolution_);
}

void FullStateReconstructor::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  computeLastFullSolution(solution);
  decoratedObserver_->observeSolution(lastFullSolution_, time_or_param_val);
}

void FullStateReconstructor::computeLastFullSolution(const Epetra_Vector& reducedSolution)
{
  reducedSpace_->expansion(reducedSolution, lastFullSolution_);
}

} // namespace MOR
