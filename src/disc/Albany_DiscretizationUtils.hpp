//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISCRETIZATION_UTILS_HPP
#define ALBANY_DISCRETIZATION_UTILS_HPP

#include <map>
#include <string>
#include <vector>

#include "Albany_KokkosTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Teuchos_ArrayRCP.hpp"

namespace Albany {

enum class DiscType
{
  BlockedMono = 0,
  Interleaved = 1,
  BlockedDisc = 2
};

using NodeSetList      = std::map<std::string, std::vector<std::vector<int>>>;
using NodeSetGIDsList  = std::map<std::string, std::vector<GO>>;
using NodeSetCoordList = std::map<std::string, std::vector<double*>>;

// Legacy SideStruct and SideSetList for compatability until all problems are converted to new layouts
class SideStruct
{
 public:
  GO       side_GID;       // global id of side in the mesh
  GO       elem_GID;       // global id of element containing side
  int      elem_LID;       // local id of element containing side
  int      elem_ebIndex;   // index of element block that contains element
  unsigned side_local_id;  // local id of side relative to owning element
};
using SideSetList = std::map<std::string, std::vector<SideStruct>>;


// This is a stucture that holds all of the sideset information over all worksets. When running populate mesh,
//   there can be a huge number of worksets with only a handful (1-20) of entries each, which causes a huge
//   performance hit on GPUs since each of these tiny views must be allocated serially on the GPU. These global/local
//   side structs avoid this issue by allocating once after computing the global view extents and then providing
//   local subviews to each workset.
// Memory layout is enforced to be LayoutRight regardless of architecture since Kokkos parallel sections
//   will only ever operate on the last extent of any of these Views and therefore they should be contiguous.
class GlobalSideSetInfo
{
public:
  int num_local_worksets;
  int max_sideset_length;
  Kokkos::View<int*, Kokkos::LayoutRight> sideset_sizes;       // (num_local_worksets)
  Kokkos::View<GO**, Kokkos::LayoutRight>       side_GID;      // (num_local_worksets, max_sideset_length)
  Kokkos::View<GO**, Kokkos::LayoutRight>       elem_GID;      // (num_local_worksets, max_sideset_length)
  Kokkos::View<int**, Kokkos::LayoutRight>      elem_LID;      // (num_local_worksets, max_sideset_length)
  Kokkos::View<int**, Kokkos::LayoutRight>      elem_ebIndex;  // (num_local_worksets, max_sideset_length)
  Kokkos::View<unsigned**, Kokkos::LayoutRight> side_local_id; // (num_local_worksets, max_sideset_length)

  int max_sides;
  Kokkos::View<int**, Kokkos::LayoutRight>      numCellsOnSide;   // (num_local_worksets, max_sides)
  Kokkos::View<int***, Kokkos::LayoutRight>     cellsOnSide;      // (num_local_worksets, max_sides, max_sideset_length)
  Kokkos::View<int***, Kokkos::LayoutRight>     sideSetIdxOnSide; // (num_local_worksets, max_sides, max_sideset_length)
};
using GlobalSideSetList = std::map<std::string, GlobalSideSetInfo>;

class LocalSideSetInfo
{
public:
  int size;
  Kokkos::View<GO*, Kokkos::LayoutRight>       side_GID;      // (size)
  Kokkos::View<GO*, Kokkos::LayoutRight>       elem_GID;      // (size)
  Kokkos::View<int*, Kokkos::LayoutRight>      elem_LID;      // (size)
  Kokkos::View<int*, Kokkos::LayoutRight>      elem_ebIndex;  // (size)
  Kokkos::View<unsigned*, Kokkos::LayoutRight> side_local_id; // (size)

  int numSides;
  Kokkos::View<int*, Kokkos::LayoutRight>      numCellsOnSide;   // (sides)
  Kokkos::View<int**, Kokkos::LayoutRight>     cellsOnSide;      // (numSides, sides)
  Kokkos::View<int**, Kokkos::LayoutRight>     sideSetIdxOnSide; // (numSides, sides)
};
using LocalSideSetInfoList = std::map<std::string, LocalSideSetInfo>;

class wsLid
{
 public:
  int ws;   // workset of element containing side
  int LID;  // local id of element containing side
};

using WsLIDList = std::map<GO, wsLid>;

template <typename T>
using WorksetArray = Teuchos::ArrayRCP<T>;

// LB 8/17/18: I moved these out of AbstractDiscretization, so if one only needs
// these types,
//             he/she can include this small file rather than
//             Albany_AbstractDiscretization.hpp, which has tons of
//             dependencies.
using WorksetConn = Kokkos::View<LO***, Kokkos::LayoutRight, PHX::Device>;
using Conn        = WorksetArray<WorksetConn>;

}  // namespace Albany

#endif  // ALBANY_DISCRETIZATION_UTILS_HPP
