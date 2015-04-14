//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_PiroObserverT.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <cstddef>

LCM::Schwarz_PiroObserverT::Schwarz_PiroObserverT(
    const Teuchos::RCP<Albany::Application> &app, const Teuchos::RCP<SchwarzMultiscale>& cs_model)
{
  //FIXME: remove app as argument 
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  apps_ = cs_model->getApps(); 
  std::cout << "# models seen by Schwarz_PiroObserverT: " << apps_.size() << std::endl; 
  impl_ = Teuchos::rcp(new ObserverImpl(app, apps_));  
}

void
LCM::Schwarz_PiroObserverT::observeSolution(const Thyra::VectorBase<ST> &solution)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
}

void
LCM::Schwarz_PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST stamp)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  this->observeSolutionImpl(solution, stamp);
}

void
LCM::Schwarz_PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST stamp)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  this->observeSolutionImpl(solution, solution_dot, stamp);
}

namespace { // anonymous

Teuchos::RCP<const Tpetra_Vector>
tpetraFromThyra(const Thyra::VectorBase<double> &v)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > v_nonowning_rcp =
    Teuchos::rcpFromRef(v);

  //FIXME: this does not work with product vectors!! 
  return ConverterT::getConstTpetraVector(v_nonowning_rcp);
}

} // anonymous namespace

void
LCM::Schwarz_PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const ST defaultStamp)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(solution);

  this->observeTpetraSolutionImpl(
      *solution_tpetra,
      Teuchos::null,
      defaultStamp);
}

void
LCM::Schwarz_PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST defaultStamp)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
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
LCM::Schwarz_PiroObserverT::observeTpetraSolutionImpl(
    const Tpetra_Vector &solution,
    Teuchos::Ptr<const Tpetra_Vector> solution_dot,
    const ST defaultStamp)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_->getTimeParamValueOrDefault(defaultStamp);

  impl_->observeSolutionT(stamp, solution, solution_dot);
}
