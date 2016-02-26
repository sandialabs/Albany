//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP
#define MOR_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "MOR_SnapshotCollection.hpp"

namespace MOR {

class MultiVectorOutputFile;

class RythmosSnapshotCollectionObserver : public Rythmos::IntegrationObserverBase<double> {
public:
  RythmosSnapshotCollectionObserver(
      int period,
      Teuchos::RCP<MultiVectorOutputFile> snapshotFile);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeIntegration(
      const Rythmos::StepperBase<double> &stepper);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  SnapshotCollection snapshotCollector_;

  virtual void observeTimeStep(const Rythmos::StepperBase<double> &stepper);
};

} // namespace MOR

#endif /*MOR_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP*/
