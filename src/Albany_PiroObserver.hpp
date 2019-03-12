//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PIRO_OBSERVER_HPP
#define ALBANY_PIRO_OBSERVER_HPP

#include "Albany_ThyraTypes.hpp"
#include "Piro_ObserverBase.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

class Application;
class ObserverImpl;

class PiroObserver : public Piro::ObserverBase<double> {
public:
  explicit PiroObserver(const Teuchos::RCP<Application> &app);

  void observeSolution(const Thyra_Vector& solution);

private:
  Teuchos::RCP<ObserverImpl> impl_;
};

} // namespace Albany

#endif // ALBANY_PIRO_OBSERVER_HPP
