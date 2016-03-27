//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TPETRA_GATHERALLV_HPP
#define TPETRA_GATHERALLV_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"
#ifdef HAVE_TPETRA_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#endif /* HAVE_TPETRA_MPI */
#include "Teuchos_DefaultSerialComm.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_TestForException.hpp"

#ifdef HAVE_TPETRA_MPI
#include "mpi.h"
#endif /* HAVE_TPETRA_MPI */

#include <numeric>
#include <algorithm>

namespace Tpetra {

template <typename GO>
int GatherAllV(
    const Teuchos::RCP<const Teuchos::Comm<int> >& commT,
    const GO *myVals, int myCount,
    GO *allVals, int allCount)
{
#ifdef HAVE_TPETRA_MPI
  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get())) {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    const int cpuCount = mpiComm->getSize();
    Teuchos::Array<int> allValCounts(cpuCount);
    const int ierr = MPI_Allgather(
        &myCount, 1, MPI_INT,
        allValCounts.getRawPtr(), 1, MPI_INT,
        rawComm);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount);
    std::partial_sum(allValCounts.begin(), allValCounts.end() - 1, allValDisps.begin() + 1);
    TEUCHOS_ASSERT(allCount == allValCounts.back() + allValDisps.back());

    MPI_Datatype GO_type =
#ifdef ALBANY_64BIT_INT
      MPI_LONG
#else
      MPI_INT
#endif
      ;
    return MPI_Allgatherv(
        const_cast<GO *>(myVals), myCount, GO_type,
        allVals, allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
        rawComm);
  } else
#endif /* HAVE_TPETRA_MPI */
  if (dynamic_cast<const Teuchos::SerialComm<int>*>(commT.get())) {
    TEUCHOS_ASSERT(myCount == allCount);
    std::copy(myVals, myVals + myCount, allVals);
    return 0;
  } else {
    const bool commTypeNotSupported = true;
    TEUCHOS_TEST_FOR_EXCEPT(commTypeNotSupported);
  }
}

} // namespace Tpetra

#endif
