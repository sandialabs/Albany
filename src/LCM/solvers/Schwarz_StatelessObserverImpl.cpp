//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_StatelessObserverImpl.hpp"

namespace LCM {

//
//
//
StatelessObserverImpl::StatelessObserverImpl(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>& apps)
    : n_models_(apps.size()), apps_(apps)
{
  sol_out_time_ = Teuchos::TimeMonitor::getNewTimer("Albany: Output to File");
  return;
}

//
//
//
StatelessObserverImpl::~StatelessObserverImpl() { return; }

//
//
//
RealType
StatelessObserverImpl::getTimeParamValueOrDefault(RealType default_value) const
{
  // FIXME, IKT, : We may want to change the logic here at some point.
  // I am assuming all the models have the same parameters,
  // so we only pull the time-label from the 0th model.
  std::string const label("Time");

  bool const have_time = apps_[0]->getParamLib()->isParameter(label);

  return have_time == true ?
             apps_[0]
                 ->getParamLib()
                 ->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
             default_value;
}

//
//
//
void
StatelessObserverImpl::observeSolutionT(
    double                                            stamp,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> non_overlapped_solution,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>
        non_overlapped_solution_dot)
{
  Teuchos::TimeMonitor timer(*sol_out_time_);

  for (int m = 0; m < n_models_; m++) {
    Teuchos::RCP<Tpetra_Vector const> const overlapped_solution =
        apps_[m]->getAdaptSolMgrT()->updateAndReturnOverlapSolutionT(
            *non_overlapped_solution[m]);
    if (non_overlapped_solution_dot[m] != Teuchos::null) {
      const Teuchos::RCP<const Tpetra_Vector> overlapped_solution_dot =
          apps_[m]->getAdaptSolMgrT()->updateAndReturnOverlapSolutionDotT(
              *non_overlapped_solution_dot[m]);
      apps_[m]->getDiscretization()->writeSolutionT(
          *overlapped_solution,
          *overlapped_solution_dot,
          stamp,
          /*overlapped =*/true);
    } else {
      apps_[m]->getDiscretization()->writeSolutionT(
          *overlapped_solution, stamp, true);
    }
  }
  return;
}

}  // namespace LCM
