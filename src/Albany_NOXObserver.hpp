//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NOXOBSERVER
#define ALBANY_NOXOBSERVER


#include "Albany_Application.hpp"
#include "NOX_Epetra_Observer.H"
#include "Teuchos_TimeMonitor.hpp"
#include "Albany_ExodusOutput.hpp" 

class Albany_NOXObserver : public NOX::Epetra::Observer
{
public:
   Albany_NOXObserver (
         const Teuchos::RCP<Albany::Application> &app_);

   ~Albany_NOXObserver ()
   { };

  //! Original version, for steady with no time or param info
  void observeSolution(
    const Epetra_Vector& solution);

  //! Improved version with space for time or parameter value
  void observeSolution(
    const Epetra_Vector& solution, double time_or_param_val);

private:
   Teuchos::RCP<Albany::Application> app;
  
   Albany::ExodusOutput exodusOutput;
};

#endif //ALBANY_NOXOBSERVER
