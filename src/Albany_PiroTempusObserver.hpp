//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PIRO_TEMPUS_OBSERVER_HPP
#define ALBANY_PIRO_TEMPUS_OBSERVER_HPP

#include "Albany_PiroObserver.hpp"
#include "Tempus_IntegratorObserverBasic.hpp"

namespace Albany {

class PiroTempusObserver : public PiroObserver,
                           public Tempus::IntegratorObserverBasic<ST>
{
public:
  PiroTempusObserver(const Teuchos::RCP<Application>& app,
                     const Teuchos::RCP<const Thyra_ModelEvaluator>& model);

  // Observe the beginning of each time step in the time loop
  // This is where adapatation happens (if any)
  void observeStartTimeStep(const Tempus::Integrator<ST>& integrator) override;

  // Observe the end of each time step in the time loop
  // This is where saving to mesh happens (if needed)
  void observeEndTimeStep(const Tempus::Integrator<ST>& integrator) override;
protected:

  bool is_first_time_step_ = true;

  Teuchos::RCP<Application> app_;
};

} // namespace Albany

#endif // ALBANY_PIRO_TEMPUS_OBSERVER_HPP
