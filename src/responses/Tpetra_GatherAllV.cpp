//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Tpetra_GatherAllV.hpp"

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

int Tpetra::GatherAllV(
    const Teuchos::RCP<const Teuchos::Comm<int> >& commT,
    const int *myVals, int myCount,
    int *allVals, int allCount)
{
#ifdef HAVE_TPETRA_MPI
  if(const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get())) {
    const MPI_Comm rawComm = mpiComm->getRawMpiComm();

    const int cpuCount = mpiComm->NumProc();
    Teuchos::Array<int> allValCounts(cpuCount);
    const int ierr = MPI_Allgather(
        &myCount, 1, MPI_INT,
        allValCounts.getRawPtr(), 1, MPI_INT,
        rawComm);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount);
    std::partial_sum(allValCounts.begin(), allValCounts.end() - 1, allValDisps.begin() + 1);
    TEUCHOS_ASSERT(allCount == allValCounts.back() + allValDisps.back());

    return MPI_Allgatherv(
        const_cast<int *>(myVals), myCount, MPI_INT,
        allVals, allValCounts.getRawPtr(), allValDisps.getRawPtr(), MPI_INT,
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
