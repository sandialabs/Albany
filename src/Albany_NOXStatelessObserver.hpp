//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATELESS_NOX_OBSERVER_HPP
#define ALBANY_STATELESS_NOX_OBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "Albany_Application.hpp"

namespace Albany {
class StatelessObserverImpl;

class NOXStatelessObserver : public NOX::Epetra::Observer {
public:
  NOXStatelessObserver(const Teuchos::RCP<Application> &app);

  void observeSolution(const Epetra_Vector& solution);
  void observeSolution(const Epetra_Vector& solution, double time_or_param_val);
private:
  Teuchos::RCP<StatelessObserverImpl> impl;
};

} // namespace Albany

#endif // ALBANY_STATELESS_NOX_OBSERVER_HPP
