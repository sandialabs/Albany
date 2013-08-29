//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroObserver.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_Ptr.hpp"

#include <cstddef>

Albany::PiroObserver::PiroObserver(
    const Teuchos::RCP<Albany::Application> &app) :
  app_(app)
{}

void
Albany::PiroObserver::observeSolution(const Thyra::VectorBase<double> &solution)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > solution_ptr =
    Teuchos::rcpFromRef(solution);
  const Teuchos::RCP<const Epetra_Vector> solution_epetra =
    Thyra::get_Epetra_Vector(*app_->getMap(), solution_ptr);

  // Determine the stamp associated with the snapshot
  const double stamp = app_->getParamLib()->isParameter("Time") ?
    app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time") :
    0.0;

  // If solution == "Steady" or "Continuation",
  // we need to update the solution from the initial guess prior to writing it out,
  // or we will not get the proper state of things like "Stress" in the Exodus file.
  app_->evaluateStateFieldManager(stamp, NULL, *solution_epetra);
  app_->getStateMgr().updateStates();

  // Perform solution output
  const Teuchos::Ptr<const Epetra_Vector> overlappedSolution(
      app_->getAdaptSolMgr()->getOverlapSolution(*solution_epetra));
  app_->getDiscretization()->writeSolution(*overlappedSolution, stamp, /*overlapped =*/ true);
}
