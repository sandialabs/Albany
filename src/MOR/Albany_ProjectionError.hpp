/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

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
