//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_PIROOBSERVER_HPP
#define ALBANY_PIROOBSERVER_HPP

#include "Piro_ObserverBase.hpp"

#include "Albany_Application.hpp"
#include "Albany_ObserverImpl.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class PiroObserver : public Piro::ObserverBase<double> {
public:
  explicit PiroObserver(const Teuchos::RCP<Application> &app);

  virtual void observeSolution(const Thyra::VectorBase<double> &solution);

private:
  ObserverImpl impl_;
};

} // namespace Albany

#endif /*ALBANY_PIROOBSERVER_HPP*/
