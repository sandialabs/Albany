//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_ObserverImpl.hpp"

namespace LCM {

//
//
//
ObserverImpl::ObserverImpl(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>& apps)
    : StatelessObserverImpl(apps)
{
  return;
}

//
//
//
ObserverImpl::~ObserverImpl() { return; }

//
//
//
void
ObserverImpl::observeSolutionT(
    double                                            stamp,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> non_overlapped_solution,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>
        non_overlapped_solution_dot)
{
  for (int m = 0; m < this->n_models_; m++) {
    this->apps_[m]->evaluateStateFieldManagerT(
        stamp,
        non_overlapped_solution_dot[m].ptr(),
        Teuchos::null,
        *non_overlapped_solution[m]);

    this->apps_[m]->getStateMgr().updateStates();
  }

  StatelessObserverImpl::observeSolutionT(
      stamp, non_overlapped_solution, non_overlapped_solution_dot);
}

}  // namespace LCM
