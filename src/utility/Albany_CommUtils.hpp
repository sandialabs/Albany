//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COMM_UTILS_HPP
#define ALBANY_COMM_UTILS_HPP

// Get Albany configuration macros
#include "Albany_config.h"

#include "Albany_CommTypes.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ConfigDefs.hpp" // For Ordinal

#if defined(ALBANY_EPETRA)
#include "Epetra_Comm.h"
#endif

namespace Albany {

Teuchos::RCP<const Teuchos_Comm> getDefaultComm();

#if defined(ALBANY_EPETRA)
MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec);

Teuchos::RCP<const Epetra_Comm> createEpetraCommFromMpiComm(const MPI_Comm& mc);
Teuchos::RCP<const Epetra_Comm> createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc);

Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec);
Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromEpetraComm(const Epetra_Comm& ec);
#endif

MPI_Comm getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc);

Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromMpiComm(const MPI_Comm& mc);

Teuchos::RCP<const Teuchos_Comm>
createTeuchosCommFromThyraComm(const Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>& tc_in);

Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>
createThyraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc_in);

} // namespace Albany

#endif  // ALBANY_COMM_UTILS_HPP
