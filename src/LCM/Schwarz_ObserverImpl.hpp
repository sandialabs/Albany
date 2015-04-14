//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef SCHWARZ_OBSERVERIMPL_HPP
#define SCHWARZ_OBSERVERIMPL_HPP

#include "Schwarz_StatelessObserverImpl.hpp"

namespace LCM {

class ObserverImpl : public StatelessObserverImpl {
public:
  explicit ObserverImpl(Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > &apps);

  virtual void observeSolutionT(
    double stamp, Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > non_overlapped_solutionT,
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > non_overlapped_solution_dotT);

private:
  ObserverImpl(const ObserverImpl&);
  ObserverImpl& operator=(const ObserverImpl&);
protected:
  int n_models_; 
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > apps_;
};

} // namespace LCM

#endif // SCHWARZ_OBSERVERIMPL_HPP
