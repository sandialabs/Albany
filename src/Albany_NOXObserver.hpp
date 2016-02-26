//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_NOXOBSERVER
#define ALBANY_NOXOBSERVER

#include "NOX_Epetra_Observer.H"

#include "Albany_Application.hpp"
#include "Albany_ObserverImpl.hpp"

class Albany_NOXObserver : public NOX::Epetra::Observer
{
public:
   Albany_NOXObserver(
       const Teuchos::RCP<Albany::Application> &app_);

  //! Original version, for steady with no time or param info
  void observeSolution(
    const Epetra_Vector& solution);

  //! Improved version with space for time or parameter value
  void observeSolution(
    const Epetra_Vector& solution, double time_or_param_val);

private:
   Albany::ObserverImpl impl;
};

#endif //ALBANY_NOXOBSERVER
