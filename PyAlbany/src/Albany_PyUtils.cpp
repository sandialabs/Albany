//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PyUtils.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GitVersion.h"

// Include the concrete Epetra Comm's, if needed
#if defined(ALBANY_EPETRA)
#ifdef ALBANY_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#endif

#ifdef ALBANY_MPI
#include <mpi.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include <cstdlib>
#include <stdexcept>
#include <time.h>

#include "MatrixMarket_Tpetra.hpp"
#include "Teuchos_TestForException.hpp"
#include "Kokkos_Macros.hpp"

// For vtune
#include <sys/types.h>
#include <unistd.h>

// For stack trace
#include <execinfo.h>
#include <cstdarg>
#include <cstdio>

#include "Teuchos_DefaultSerialComm.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Details_MpiTypeTraits.hpp"

namespace PyAlbany
{
  void
  PrintPyHeader(std::ostream &os)
  {
    os << R"(*********************************************************************************)" << std::endl;
    os << R"(**  ______   __  __   ______   __       ______   ______   __   __   __  __     **)" << std::endl;
    os << R"(** /\  == \ /\ \_\ \ /\  __ \ /\ \     /\  == \ /\  __ \ /\ "-.\ \ /\ \_\ \    **)" << std::endl;
    os << R"(** \ \  __/ \ \____ \\ \  __ \\ \ \____\ \  __< \ \  __ \\ \ \-.  \\ \____ \   **)" << std::endl;
    os << R"(**  \ \_\    \/\_____\\ \_\ \_\\ \_____\\ \_____\\ \_\ \_\\ \_\\"\_\\/\_____\  **)" << std::endl;
    os << R"(**   \/_/     \/_____/ \/_/\/_/ \/_____/ \/_____/ \/_/\/_/ \/_/ \/_/ \/_____/  **)" << std::endl;
    os << R"(**                                                                             **)" << std::endl;
    os << R"(*********************************************************************************)" << std::endl;
    os << R"(** Trilinos git commit id - )" << ALBANY_TRILINOS_GIT_COMMIT_ID << std::endl;
    os << R"(** Albany git branch ------ )" << ALBANY_GIT_BRANCH << std::endl;
    os << R"(** Albany git commit id --- )" << ALBANY_GIT_COMMIT_ID << std::endl;
    os << R"(** Albany cxx compiler ---- )" << ALBANY_CXX_COMPILER_ID << " " << ALBANY_CXX_COMPILER_VERSION << std::endl;

#ifdef KOKKOS_COMPILER_CUDA_VERSION
    os << R"(** Albany cuda compiler --- Cuda )" << KOKKOS_COMPILER_CUDA_VERSION << std::endl;
#endif

    // Print start time
    time_t rawtime;
    time(&rawtime);
    struct tm *timeinfo = localtime(&rawtime);
    char buffer[80];
    strftime(buffer, 80, "%F at %T", timeinfo);
    os << R"(** Simulation start time -- )" << buffer << std::endl;
    os << R"(***************************************************************)" << std::endl;
  }

  void
  correctIDs(const Teuchos::RCP<const Teuchos_Comm> &comm,
             const Teuchos::ArrayView<Tpetra_GO> &myVals,
             int indexBase)
  {
    const int myCount = myVals.size();
    Teuchos::Array<Tpetra_GO> allVals;
    const LO root_rank = 0;
    if (const Teuchos::MpiComm<int> *mpiComm = dynamic_cast<const Teuchos::MpiComm<int> *>(comm.get()))
    {
      // First, the input global IDs are gathered on the rank 0.
      MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

      int allCount;
      MPI_Allreduce(&myCount, &allCount, 1, MPI_INT, MPI_SUM, rawComm);

      int myRank = comm->getRank();
      int myAllCount = (myRank == root_rank ? allCount : 0);
      allVals.resize(myAllCount);

      const int cpuCount = mpiComm->getSize();
      Teuchos::Array<int> allValCounts(cpuCount);
      const int ierr = MPI_Gather(
          &myCount, 1, MPI_INT,
          allValCounts.getRawPtr(), 1, MPI_INT,
          root_rank, rawComm);
      TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);

      Teuchos::Array<int> allValDisps(cpuCount, 0);
      for (int i = 1; i < cpuCount; ++i)
      {
        allValDisps[i] = allValDisps[i - 1] + allValCounts[i - 1];
      }
      ALBANY_EXPECT(myRank != root_rank || (allCount == allValCounts.back() + allValDisps.back()), "Error! Mismatch in values counts.\n");

      auto GO_type = Teuchos::Details::MpiTypeTraits<Tpetra_GO>::getType();
      MPI_Gatherv(
          const_cast<Tpetra_GO *>(myVals.getRawPtr()), myCount, GO_type,
          allVals.getRawPtr(), allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
          root_rank, rawComm);

      // Now that all the global IDs are on the rank 0, they are sorted and the indices
      // of the sort are kept.

      std::vector<Tpetra_GO> idx(allVals.size());
      std::vector<Tpetra_GO> inv_idx(allVals.size());
      std::iota(idx.begin(), idx.end(), 0);

      std::stable_sort(idx.begin(), idx.end(),
                       [&allVals](Tpetra_GO i1, Tpetra_GO i2) { return allVals[i1] < allVals[i2]; });

      for (int i = 0; i < allVals.size(); ++i)
        inv_idx[idx[i]] = i;

      // The previous global IDs are replaced by its order in the sorted sequence plus
      // the indexbase.
      for (int i = 0; i < allVals.size(); ++i)
        allVals[i] = inv_idx[i] + indexBase;

      // The new global IDs are then scattered to all the owning processes.
      MPI_Scatterv(
          const_cast<Tpetra_GO *>(allVals.getRawPtr()), allValCounts.getRawPtr(), allValDisps.getRawPtr(), GO_type,
          myVals.getRawPtr(), myCount, GO_type,
          root_rank, rawComm);
    }
  }

  Teuchos::RCP<const PyTrilinosMap> getPyTrilinosMap(Teuchos::RCP<const Tpetra_Map> t_map, bool correctGIDs)
  {
    if (!correctGIDs)
    {
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
      return t_map;
#endif
    }
    PyTrilinosMap::global_ordinal_type globalNumElements = t_map->getGlobalNumElements();
    auto indexBase = t_map->getIndexBase();
    auto comm = t_map->getComm();
    auto nodeNumElements = t_map->getNodeNumElements();
    PyTrilinosMap::global_ordinal_type myIndicesLongLong[nodeNumElements];
    Teuchos::Array<Tpetra_GO> nodes_gids = t_map->getNodeElementList();

    if (correctGIDs)
      correctIDs(comm, nodes_gids, indexBase);

    for (size_t i = 0; i < nodeNumElements; ++i)
      myIndicesLongLong[i] = nodes_gids[i];

    Teuchos::RCP<PyTrilinosMap> out_map = rcp(new PyTrilinosMap(globalNumElements, myIndicesLongLong, nodeNumElements, indexBase, comm));
    return out_map;
  }
} // namespace PyAlbany
