//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CommUtils.hpp"

#include "Albany_ThyraUtils.hpp"

// Include the concrete Epetra Comm's, if needed
#if defined(ALBANY_EPETRA)
  #include "Epetra_MpiComm.h"
#endif

namespace Albany
{

#if defined(ALBANY_EPETRA)  
MPI_Comm
getMpiCommFromEpetraComm(const Epetra_Comm& ec)
{
  const Epetra_MpiComm& emc = dynamic_cast<const Epetra_MpiComm&>(ec);
  return emc.Comm();
}

Teuchos::RCP<const Epetra_Comm>
createEpetraCommFromMpiComm(const MPI_Comm& mc)
{
  return Teuchos::rcp(new Epetra_MpiComm(mc));
}

Teuchos::RCP<const Epetra_Comm>
createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc)
{
  auto mpiComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(tc);
  return  createEpetraCommFromMpiComm(*mpiComm->getRawMpiComm()());
}

Teuchos::RCP<const Teuchos_Comm>
createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec)
{
  auto mpiComm = Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(ec);
  return  createTeuchosCommFromMpiComm(mpiComm->Comm());
}

Teuchos::RCP<const Teuchos_Comm>
createTeuchosCommFromEpetraComm(const Epetra_Comm& ec)
{
  const Epetra_MpiComm *mpiComm = dynamic_cast<const Epetra_MpiComm *>(&ec);
  return  createTeuchosCommFromMpiComm(mpiComm->Comm());
}
#endif // defined(ALBANY_EPETRA)

MPI_Comm
getMpiCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc)
{
  auto mpiComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(tc);
  return *mpiComm->getRawMpiComm();
}

Teuchos::RCP<const Teuchos_Comm>
createTeuchosCommFromMpiComm(const MPI_Comm& mc)
{
  // The default tag in the MpiComm is used in Teuchos send/recv operations *only if* the user
  // does not specify a tag for the message. Here, I pick a weird large number, unlikely
  // to ever be hit by a tag used by albany.
  return Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(mc),1984));
}

Teuchos::RCP<const Teuchos_Comm>
getDefaultComm()
{
  return Teuchos::DefaultComm<int>::getComm();
}

Teuchos::RCP<const Teuchos_Comm>
createTeuchosCommFromThyraComm(const Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>& tc_in)
{
  auto mpiCommIn = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<Teuchos::Ordinal> >(tc_in);
  return Teuchos::createMpiComm<int>(mpiCommIn->getRawMpiComm());
}

Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal>>
createThyraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc_in)
{
  auto mpiCommIn = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(tc_in);
  return Teuchos::createMpiComm<Teuchos::Ordinal>(mpiCommIn->getRawMpiComm());
}

template<>
MPI_Datatype get_mpi_type<int> () { return MPI_INT; }
template<>
MPI_Datatype get_mpi_type<long long> () { return MPI_LONG_LONG_INT; }
template<>
MPI_Datatype get_mpi_type<double> () { return MPI_DOUBLE; }

} // namespace Albany
