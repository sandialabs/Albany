//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_RYTHMOSPROJECTIONERROROBSERVER_HPP
#define MOR_RYTHMOSPROJECTIONERROROBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "MOR_ProjectionError.hpp"

namespace MOR {

class ReducedSpace;
class MultiVectorOutputFile;

class RythmosProjectionErrorObserver : public Rythmos::IntegrationObserverBase<double> {
public:
  RythmosProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeIntegration(const Rythmos::StepperBase<double> &stepper);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  ProjectionError projectionError_;

  virtual void observeTimeStep(const Rythmos::StepperBase<double> &stepper);
};

} // namespace MOR

#endif /*MOR_RYTHMOSPROJECTIONERROROBSERVER_HPP*/
