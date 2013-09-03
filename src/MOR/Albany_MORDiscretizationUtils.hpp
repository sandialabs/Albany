//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_AbstractDiscretization.hpp"

namespace Albany {

void
setupInternalMeshStruct(
    DiscretizationFactory &discFactory,
    const Teuchos::RCP<Teuchos::ParameterList> &problemParams,
    const Teuchos::RCP<const Epetra_Comm> &epetraComm);

Teuchos::RCP<AbstractDiscretization>
createDiscretization(DiscretizationFactory &discFactory);

Teuchos::RCP<AbstractDiscretization>
discretizationNew(
    DiscretizationFactory &discFactory,
    const Teuchos::RCP<Teuchos::ParameterList> &problemParams,
    const Teuchos::RCP<const Epetra_Comm> &epetraComm);

} // namespace Albany
