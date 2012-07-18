/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#ifndef ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP
#define ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Albany_SnapshotCollection.hpp"

class Epetra_Map;

#include "Teuchos_ParameterList.hpp"

namespace Albany {

class RythmosSnapshotCollectionObserver : public Rythmos::IntegrationObserverBase<double> {
public:
  explicit RythmosSnapshotCollectionObserver(const Teuchos::RCP<Teuchos::ParameterList> &params,
                                             const Teuchos::RCP<const Epetra_Map> &stateMap);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  SnapshotCollection snapshotCollector_;
  Teuchos::RCP<const Epetra_Map> stateMap_;

  virtual void observeTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

};

} // namespace Albany

#endif /*ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP*/
