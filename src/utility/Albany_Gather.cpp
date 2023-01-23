#include "Albany_Gather.hpp"

#include "Albany_Macros.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"

#include <mpi.h>
#include <Teuchos_DefaultMpiComm.hpp>

#include "Teuchos_DefaultSerialComm.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Details_MpiTypeTraits.hpp"

namespace Albany {

void gatherAllV(const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::ArrayView<const GO>& myVals,
                Teuchos::Array<GO>& allVals)
{
  const int myCount  = myVals.size();

  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get())) {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    int allCount;
    MPI_Allreduce(&myCount,&allCount,1,MPI_INT,MPI_SUM,rawComm);
    allVals.resize(allCount);

    const int cpuCount = mpiComm->getSize();
    Teuchos::Array<int> allValCounts(cpuCount);
    const int ierr = MPI_Allgather(
        &myCount, 1, MPI_INT,
        allValCounts.getRawPtr(), 1, MPI_INT,
        rawComm);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount,0);
    for (int i=1; i<cpuCount; ++i) {
      allValDisps[i] = allValDisps[i-1] + allValCounts[i-1];
    }
    ALBANY_EXPECT(allCount == allValCounts.back() + allValDisps.back(),"Error! Mismatch in values counts.\n");

    auto GO_type = Teuchos::Details::MpiTypeTraits<GO>::getType();
    MPI_Allgatherv(
        const_cast<GO *>(myVals.getRawPtr()), myCount, GO_type,
        allVals.getRawPtr(), allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
        rawComm);
  } else

  if (dynamic_cast<const Teuchos::SerialComm<int>*>(comm.get())) {
    allVals.resize(myCount);
    std::copy(myVals.getRawPtr(), myVals.getRawPtr() + myCount, allVals.getRawPtr());
  } else {
    const bool commTypeNotSupported = true;
    TEUCHOS_TEST_FOR_EXCEPT(commTypeNotSupported);
  }
}

void gatherV(const Teuchos::RCP<const Teuchos_Comm>& comm,
             const Teuchos::ArrayView<const GO>& myVals,
             Teuchos::Array<GO>& allVals, const LO root_rank)
{
  const int myCount  = myVals.size();

  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get())) {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    int allCount;
    MPI_Allreduce(&myCount,&allCount,1,MPI_INT,MPI_SUM,rawComm);

    int myRank = comm->getRank();
    int myAllCount = (myRank==root_rank ? allCount : 0);
    allVals.resize(myAllCount);

    const int cpuCount = mpiComm->getSize();
    Teuchos::Array<int> allValCounts(cpuCount);
    const int ierr = MPI_Gather(
        &myCount, 1, MPI_INT,
        allValCounts.getRawPtr(), 1, MPI_INT,
        root_rank,rawComm);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);

    Teuchos::Array<int> allValDisps(cpuCount,0);
    for (int i=1; i<cpuCount; ++i) {
      allValDisps[i] = allValDisps[i-1] + allValCounts[i-1];
    }
    ALBANY_EXPECT(myRank!=root_rank || (allCount==allValCounts.back() + allValDisps.back()),"Error! Mismatch in values counts.\n");

    auto GO_type = Teuchos::Details::MpiTypeTraits<GO>::getType();
    MPI_Gatherv(
        const_cast<GO *>(myVals.getRawPtr()), myCount, GO_type,
        allVals.getRawPtr(), allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
        root_rank,rawComm);
  } else

  if (dynamic_cast<const Teuchos::SerialComm<int>*>(comm.get())) {
    allVals.resize(myCount);
    std::copy(myVals.getRawPtr(), myVals.getRawPtr() + myCount, allVals.getRawPtr());
  } else {
    const bool commTypeNotSupported = true;
    TEUCHOS_TEST_FOR_EXCEPT(commTypeNotSupported);
  }
}

} // namespace Albany
