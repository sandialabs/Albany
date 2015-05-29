//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_PiroObserverT.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <cstddef>

//#define OUTPUT_TO_SCREEN 

LCM::Schwarz_PiroObserverT::Schwarz_PiroObserverT(
    const Teuchos::RCP<SchwarzMultiscale>& cs_model)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  apps_ = cs_model->getApps();
  n_models_ = apps_.size();
#ifdef OUTPUT_TO_SCREEN
  std::cout << "# models seen by Schwarz_PiroObserverT: " << n_models_ << '\n';
#endif 
  impl_ = Teuchos::rcp(new ObserverImpl(apps_));
}

void
LCM::Schwarz_PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
}

void
LCM::Schwarz_PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST stamp)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  this->observeSolutionImpl(solution, stamp);
}

void
LCM::Schwarz_PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST stamp)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  this->observeSolutionImpl(solution, solution_dot, stamp);
}

namespace { // anonymous

Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> >
tpetraFromThyra(const Thyra::VectorBase<double> &v, int n_models)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

  const Teuchos::RCP<const Thyra::ProductVectorBase<ST> > v_nonowning_rcp =
      Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(
          Teuchos::rcpFromRef(v));

  //Create a Teuchos array of the vs for each model.
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > vs(n_models);
  for (int m = 0; m < n_models; ++m) {
    //Get each Tpetra vector
    vs[m] = Teuchos::rcp_dynamic_cast<const ThyraVector>(
        v_nonowning_rcp->getVectorBlock(m), true)->getConstTpetraVector();
  }
  return vs;
}

} // anonymous namespace

void
LCM::Schwarz_PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const ST default_stamp)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > solutions_tpetra =
      tpetraFromThyra(solution, n_models_);
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > null_array;
  null_array.resize(n_models_);
  for (int m = 0; m < n_models_; m++)
    null_array[m] = Teuchos::null;

  this->observeTpetraSolutionImpl(
      solutions_tpetra,
      null_array,
      default_stamp);
}

void
LCM::Schwarz_PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST default_stamp)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > solutions_tpetra =
      tpetraFromThyra(solution, n_models_);
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > solutions_dot_tpetra =
      tpetraFromThyra(solution_dot, n_models_);

  this->observeTpetraSolutionImpl(
      solutions_tpetra,
      solutions_dot_tpetra,
      default_stamp);
}

void
LCM::Schwarz_PiroObserverT::observeTpetraSolutionImpl(
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > solutions,
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > solutions_dot,
    const ST default_stamp)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_->getTimeParamValueOrDefault(default_stamp);

  //FIXME: change arguments to take in arrays
  impl_->observeSolutionT(stamp, solutions, solutions_dot);
}
