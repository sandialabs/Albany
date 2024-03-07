//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CommUtils.hpp"

#include "Albany_ThyraUtils.hpp"

namespace Albany
{

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
