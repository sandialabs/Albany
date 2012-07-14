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

#ifndef ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP
#define ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Teuchos_ParameterList.hpp"

class Epetra_Map;

namespace Albany {

class ReducedSpace;

class RythmosFullStateReconstructor : public Rythmos::IntegrationObserverBase<double> {
public:
  explicit RythmosFullStateReconstructor(const Teuchos::RCP<Teuchos::ParameterList> &params,
                                         const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &decoratedObserver,
                                         const Teuchos::RCP<const Epetra_Map> &decoratedMap);

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
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > decoratedObserver_;
  Teuchos::RCP<const Epetra_Map> decoratedMap_;
  Teuchos::RCP<const ReducedSpace> reducedSpace_;
};

} // namespace Albany

#endif /*ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP*/
