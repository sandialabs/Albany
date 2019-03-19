#include "Albany_GatherAllV.hpp"

#include "Albany_Macros.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"

#ifdef ALBANY_MPI
#include <mpi.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include "Teuchos_DefaultSerialComm.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Details_MpiTypeTraits.hpp"
#include <numeric>

namespace Albany {

void gatherAllV(const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::ArrayView<const GO>& myVals,
                Teuchos::Array<GO>& allVals)
{
  const int myCount  = myVals.size();
  const int allCount = allVals.size();

  TEUCHOS_TEST_FOR_EXCEPTION (allCount<myCount, std::logic_error,
                              "Error! The array allVals must be at least as large as myVals.\n"
                              "       Did you forget to properly size allVals?\n");
#ifdef ALBANY_MPI
  if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get())) {
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
    ALBANY_EXPECT(allCount == allValCounts.back() + allValDisps.back(),"Error! Mismatch in values counts.\n");

    auto GO_type = Teuchos::Details::MpiTypeTraits<GO>::getType();
    MPI_Allgatherv(
        const_cast<GO *>(myVals.getRawPtr()), myCount, GO_type,
        allVals.getRawPtr(), allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
        rawComm);
  } else
#endif
  if (dynamic_cast<const Teuchos::SerialComm<int>*>(comm.get())) {
    TEUCHOS_ASSERT(myCount == allCount);
    std::copy(myVals.getRawPtr(), myVals.getRawPtr() + myCount, allVals.getRawPtr());
  } else {
    const bool commTypeNotSupported = true;
    TEUCHOS_TEST_FOR_EXCEPT(commTypeNotSupported);
  }
}

} // namespace Albany
