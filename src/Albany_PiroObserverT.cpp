//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroObserverT.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Thyra_TpetraThyraWrappers.hpp"

#include "Teuchos_ScalarTraits.hpp"

#include <cstddef>

Albany::PiroObserverT::PiroObserverT(
    const Teuchos::RCP<Albany::Application> &app) :
  app_(app),
  exodusOutput_(app_->getDiscretization())
{}

void
Albany::PiroObserverT::observeSolution(const Thyra::VectorBase<ST> &solution)
{
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST &stamp)
{
  this->observeSolutionImpl(solution, stamp);
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST &stamp)
{
  this->observeSolutionImpl(solution, solution_dot, stamp);
}

namespace { // anonymous

Teuchos::RCP<const Tpetra_Vector>
tpetraFromThyra(const Thyra::VectorBase<double> &v)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > v_nonowning_rcp =
    Teuchos::rcpFromRef(v);

  typedef Thyra::TpetraOperatorVectorExtraction<ST, int> ConverterT;
  return ConverterT::getConstTpetraVector(v_nonowning_rcp);
}

} // anonymous namespace

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const ST &defaultStamp)
{
  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(solution);

  this->observeTpetraSolutionImpl(
      *solution_tpetra,
      Teuchos::null,
      defaultStamp);
}

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST &defaultStamp)
{
  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(solution);
  const Teuchos::RCP<const Tpetra_Vector> solution_dot_tpetra =
    tpetraFromThyra(solution_dot);

  this->observeTpetraSolutionImpl(
      *solution_tpetra,
      solution_dot_tpetra.ptr(),
      defaultStamp);
}

void
Albany::PiroObserverT::observeTpetraSolutionImpl(
    const Tpetra_Vector &solution,
    Teuchos::Ptr<const Tpetra_Vector> solution_dot,
    const ST &defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = app_->getParamLib()->isParameter("Time") ?
    app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time") :
    defaultStamp;

  // We need to update the solution from the initial guess prior to writing it out,
  // or we will not get the proper state of things like "Stress" in the Exodus file.
  app_->evaluateStateFieldManagerT(stamp, solution_dot, solution);
  app_->getStateMgr().updateStates();

  // Perform Exodus output if the SEACAS package is enabled
#ifdef ALBANY_SEACAS
  const Teuchos::RCP<const Tpetra_Vector> overlappedSolution =
    app_->getOverlapSolutionT(solution);
  exodusOutput_.writeSolutionT(stamp, *overlappedSolution, /*overlapped =*/ true);
#endif /* ALBANY_SEACAS */
}
