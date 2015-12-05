//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "Albany_PiroObserverT.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <cstddef>

Albany::PiroObserverT::PiroObserverT(
    const Teuchos::RCP<Albany::Application> &app) :
  impl_(app)
{}

void
Albany::PiroObserverT::observeSolution(const Thyra::VectorBase<ST> &solution)
{
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dot, stamp);
}

void
Albany::PiroObserverT::observeSolution(
      const Piro::SolnSet<ST> &solution)
{
  this->observeSolutionImpl(solution);
}

namespace { // anonymous

Teuchos::RCP<const Tpetra_Vector>
tpetraFromThyra(const Thyra::VectorBase<double> &v)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > v_nonowning_rcp =
    Teuchos::rcpFromRef(v);

  return ConverterT::getConstTpetraVector(v_nonowning_rcp);
}

} // anonymous namespace

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const ST defaultStamp)
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
    const ST defaultStamp)
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
Albany::PiroObserverT::observeSolutionImpl(
    const Piro::SolnSet<ST> &solution_)
{

  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(*solution_.solution);
  const Teuchos::RCP<const Tpetra_Vector> solution_dot_tpetra =
    tpetraFromThyra(*solution_.solution_dot);
  const Teuchos::RCP<const Tpetra_Vector> solution_dotdot_tpetra =
    tpetraFromThyra(*solution_.solution_dotdot);

  impl_.observeSolutionT(solution_.stamp, *solution_tpetra, 
       solution_dot_tpetra.ptr(), solution_dotdot_tpetra.ptr());
}

void
Albany::PiroObserverT::observeTpetraSolutionImpl(
    const Tpetra_Vector &solution,
    Teuchos::Ptr<const Tpetra_Vector> solution_dot,
    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);

  impl_.observeSolutionT(stamp, solution, solution_dot);
}

