//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GENERALIZEDCOORDINATESRYTHMOSOBSERVERBSERVER_HPP
#define MOR_GENERALIZEDCOORDINATESRYTHMOSOBSERVERBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "MOR_GeneralizedCoordinatesOutput.hpp"

namespace MOR {

class GeneralizedCoordinatesRythmosObserver : public Rythmos::IntegrationObserverBase<double> {
public:
  GeneralizedCoordinatesRythmosObserver(const std::string &filename, const std::string &stampsFilename);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeIntegration(const Rythmos::StepperBase<double> &stepper);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  GeneralizedCoordinatesOutput impl_;

  virtual void observeTimeStep(const Rythmos::StepperBase<double> &stepper);
};

} // end namespace MOR

#endif /*MOR_GENERALIZEDCOORDINATESRYTHMOSOBSERVERBSERVER_HPP*/
