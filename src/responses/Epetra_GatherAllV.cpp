//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Epetra_GatherAllV.hpp"

#include "Epetra_ConfigDefs.h"

#include "Epetra_Comm.h"
#ifdef EPETRA_MPI
#include "Epetra_MpiComm.h"
#endif /* EPETRA_MPI */
#include "Epetra_SerialComm.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_TestForException.hpp"

#ifdef EPETRA_MPI
#include "mpi.h"
#endif /* EPETRA_MPI */

#include <numeric>
#include <algorithm>

int Epetra::GatherAllV(
    const Epetra_Comm &comm,
    const int *myVals, int myCount,
    int *allVals, int allCount)
{
#ifdef EPETRA_MPI
  if (const Epetra_MpiComm *mpiComm = dynamic_cast<const Epetra_MpiComm *>(&comm)) {
    const MPI_Comm rawComm = mpiComm->Comm();

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
#endif /* EPETRA_MPI */
  if (dynamic_cast<const Epetra_SerialComm *>(&comm)) {
    TEUCHOS_ASSERT(myCount == allCount);
    std::copy(myVals, myVals + myCount, allVals);
    return 0;
  } else {
    const bool commTypeNotSupported = true;
    TEUCHOS_TEST_FOR_EXCEPT(commTypeNotSupported);
  }
}
