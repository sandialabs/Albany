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

class ExplicitODEModelEvaluator;

class PiroTempusObserver : public PiroObserver,
                           public Tempus::IntegratorObserverBasic<ST>
{
public:
  PiroTempusObserver(const Teuchos::RCP<Application>& app,
                     const Teuchos::RCP<const Thyra_ModelEvaluator>& model);

  // Observe the end of each time step in the time loop
  void observeEndTimeStep(const Tempus::Integrator<ST>& integrator) override;

  // Observe the end of the time integration
  void observeEndIntegrator(const Tempus::Integrator<ST>& integrator) override;

  // When the model is exposed to Tempus as an explicit ODE, the dofs stored by
  // Tempus are not evolved: they are made consistent with the new state at the end of
  // each step (and, after mesh adaptation, the model is rebuilt on the new dof layout).
  void setExplicitODEModel (const Teuchos::RCP<ExplicitODEModelEvaluator>& model) {
    explicit_ode_model_ = model;
  }
protected:

  Teuchos::RCP<Application> app_;
  Teuchos::RCP<ExplicitODEModelEvaluator> explicit_ode_model_;
};

} // namespace Albany

#endif // ALBANY_PIRO_TEMPUS_OBSERVER_HPP
