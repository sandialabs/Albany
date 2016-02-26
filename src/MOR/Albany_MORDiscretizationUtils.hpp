//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DiscretizationFactory.hpp"

#include "Epetra_Comm.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

Teuchos::RCP<AbstractDiscretization>
discretizationNew(
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams,
    const Teuchos::RCP<const Teuchos_Comm> &comm);

class DiscretizationTransformation {
public:
  virtual void operator()(DiscretizationFactory &) = 0;
  virtual ~DiscretizationTransformation() {}
};

Teuchos::RCP<AbstractDiscretization>
modifiedDiscretizationNew(
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams,
    const Teuchos::RCP<const Teuchos_Comm> &comm,
    DiscretizationTransformation &transformation);

} // namespace Albany
