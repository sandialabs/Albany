//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_OBSERVERIMPL_HPP
#define ALBANY_OBSERVERIMPL_HPP

#include "Albany_StatelessObserverImpl.hpp"

namespace Albany {

class ObserverImpl : public StatelessObserverImpl {
public:
  explicit ObserverImpl(const Teuchos::RCP<Application>& app);

#if defined(ALBANY_EPETRA)
  virtual void observeSolution(
    double stamp, const Epetra_Vector& nonOverlappedSolution,
    const Teuchos::Ptr<const Epetra_Vector>& nonOverlappedSolutionDot);
#endif

  virtual void observeSolutionT(
    double stamp, const Tpetra_Vector& nonOverlappedSolutionT,
    const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotT,
    const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotDotT = Teuchos::null);

  virtual void observeSolutionT(
    double stamp, const Tpetra_MultiVector& nonOverlappedSolutionT);

private:
  ObserverImpl(const ObserverImpl&);
  ObserverImpl& operator=(const ObserverImpl&);
};

} // namespace Albany

#endif // ALBANY_OBSERVERIMPL_HPP
