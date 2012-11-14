//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_PROJECTIONERROR_HPP
#define ALBANY_PROJECTIONERROR_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"

#include <deque>

namespace Albany {

class ReducedSpace;

class ProjectionError {
public:
  ProjectionError(const Teuchos::RCP<Teuchos::ParameterList> &params,
                  const Teuchos::RCP<const Epetra_Map> &dofMap);

  ~ProjectionError();

  void process(const Epetra_MultiVector &v);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;
  Teuchos::RCP<const Epetra_Map> dofMap_;

  Teuchos::RCP<ReducedSpace> reducedSpace_;

  std::deque<double> relativeErrorNorms_;

  static Teuchos::RCP<Teuchos::ParameterList> fillDefaultParams(const Teuchos::RCP<Teuchos::ParameterList> &params);

  // Disallow copy & assignment
  ProjectionError(const ProjectionError &);
  ProjectionError &operator=(const ProjectionError &);
};

} // end namespace Albany

#endif /* ALBANY_PROJECTIONERROR_HPP */
