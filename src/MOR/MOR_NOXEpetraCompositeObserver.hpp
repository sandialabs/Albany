//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_NOXEPETRACOMPOSITEOBSERVER_HPP
#define MOR_NOXEPETRACOMPOSITEOBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

namespace MOR {

class NOXEpetraCompositeObserver : public NOX::Epetra::Observer {
public:
  //! Calls observeSolution for all subobservers in the same order they have been added
  virtual void observeSolution(const Epetra_Vector& solution);

  //! Calls observeSolution for all subobservers in the same order they have been added
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

  int observerCount() const;
  void addObserver(const Teuchos::RCP<NOX::Epetra::Observer> &);

private:
  typedef Teuchos::Array<Teuchos::RCP<NOX::Epetra::Observer> > ObserverSequence;
  ObserverSequence observers_;
};

} // end namespace MOR

#endif /* MOR_NOXEPETRACOMPOSITEOBSERVER_HPP */

