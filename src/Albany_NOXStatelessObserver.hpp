//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATELESSNOXOBSERVER
#define ALBANY_STATELESSNOXOBSERVER

#include "NOX_Epetra_Observer.H"

#include "Albany_Application.hpp"
#include "Albany_StatelessObserverImpl.hpp"

namespace Albany {

class NOXStatelessObserver : public NOX::Epetra::Observer {
public:
  NOXStatelessObserver(const Teuchos::RCP<Albany::Application> &app);
  void observeSolution(const Epetra_Vector& solution);
  void observeSolution(const Epetra_Vector& solution, double time_or_param_val);
private:
   Albany::StatelessObserverImpl impl;
};

}

#endif // ALBANY_STATELESSNOXOBSERVER
