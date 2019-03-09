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

#if defined(ALBANY_EPETRA)
#include "Epetra_Comm.h"
#endif

namespace Albany {

#if defined(ALBANY_EPETRA)
Albany_MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec);

Teuchos::RCP<const Epetra_Comm> createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc);
Teuchos::RCP<const Epetra_Comm> createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc);

Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec);
Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromEpetraComm(const Epetra_Comm& ec);
#endif

Albany_MPI_Comm getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc);

Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc);

template<typename OrdinalType>
Teuchos::RCP<const Teuchos_Comm> createTeuchosCommFromTeuchosComm(const Teuchos::RCP<const Teuchos::Comm<OrdinalType>>& tc_in) {
#ifdef HAVE_MPI
  const Teuchos::RCP<const Teuchos::MpiComm<OrdinalType> > mpiCommIn =
    Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<OrdinalType> >(tc_in);
  if (nonnull(mpiCommIn)) {
    return Teuchos::createMpiComm<int>(mpiCommIn->getRawMpiComm());
  }
#endif // HAVE_MPI

  // Assert conversion to Teuchos::SerialComm as a last resort (or throw)
  Teuchos::rcp_dynamic_cast<const Teuchos::SerialComm<OrdinalType> >(tc_in, true);

  return Teuchos::createSerialComm<int>();
}

} // namespace Albany

#endif  // ALBANY_COMM_UTILS_HPP
