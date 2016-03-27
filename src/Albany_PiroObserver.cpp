//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#include "Albany_PiroObserver.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_ENull.hpp"

Albany::PiroObserver::PiroObserver(
    const Teuchos::RCP<Albany::Application> &app) :
  impl_(app)
{}

void
Albany::PiroObserver::observeSolution(const Thyra::VectorBase<double> &solution)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > solution_ptr =
    Teuchos::rcpFromRef(solution);
  const Teuchos::RCP<const Epetra_Vector> solution_epetra =
    Thyra::get_Epetra_Vector(impl_.getNonOverlappedMap(), solution_ptr);

  // Determine the stamp associated with the snapshot
  const double stamp = impl_.getTimeParamValueOrDefault(0.0);

  impl_.observeSolution(stamp, *solution_epetra, Teuchos::null);
}
