//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_OBSERVERIMPL_HPP
#define ALBANY_OBSERVERIMPL_HPP

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Ptr.hpp"

#include "Teuchos_Time.hpp"

namespace Albany {

class ObserverImpl {
public:
  explicit ObserverImpl(const Teuchos::RCP<Application> &app);

  RealType getTimeParamValueOrDefault(RealType defaultValue) const;

  Epetra_Map getNonOverlappedMap() const;

  Teuchos::RCP<const Tpetra_Map> getNonOverlappedMapT() const;

  void observeSolution(
      double stamp,
      const Epetra_Vector &nonOverlappedSolution,
      Teuchos::Ptr<const Epetra_Vector> nonOverlappedSolutionDot);

  void observeSolutionT(
      double stamp,
      const Tpetra_Vector &nonOverlappedSolutionT,
      Teuchos::Ptr<const Tpetra_Vector> nonOverlappedSolutionDotT);

private:
  Teuchos::RCP<Application> app_;

  Teuchos::RCP<Teuchos::Time> solOutTime_;

  // Disallow copy and assignment
  ObserverImpl(const ObserverImpl &);
  ObserverImpl &operator=(const ObserverImpl &);
};

} // namespace Albany

#endif //ALBANY_OBSERVERIMPL_HPP
