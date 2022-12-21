//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISCRETIZATION_UTILS_HPP
#define ALBANY_DISCRETIZATION_UTILS_HPP

#include "Albany_KokkosTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Intrepid2_Basis.hpp"
#include "Teuchos_ArrayRCP.hpp"

#include <map>
#include <string>
#include <vector>

namespace Albany {

enum class FE_Type {
  HVOL,
  HDIV,
  HCURL,
  HGRAD
};

inline std::string e2str (const FE_Type fe_type)
{
  std::string s;
  switch (fe_type) {
    case FE_Type::HVOL:  s = "HVOL";  break;
    case FE_Type::HDIV:  s = "HDIV";  break;
    case FE_Type::HCURL: s = "HCURL"; break;
    case FE_Type::HGRAD: s = "HGRAD"; break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! Unsupported FE_Type.\n");
  }
  return s;
}

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis (const CellTopologyData& cell_topo,
                   const FE_Type fe_type, const int order);

enum class DiscType
{
  BlockedMono = 0,
  Interleaved = 1,
  BlockedDisc = 2
};

// list[ns_name][inode] = <ielem,elem_pos> (we pick _any_ elem containing that node)
using NodeSetList      = std::map<std::string, std::vector<std::pair<int,int>>>;
// list[ns_name][inode] = node_gid
using NodeSetGIDsList  = std::map<std::string, std::vector<GO>>;
// list[ns_name][inode] = ptr_to_coords
using NodeSetCoordList = std::map<std::string, std::vector<double*>>;

// Legacy SideStruct and SideSetList for compatibility until all problems are converted to new layouts
class SideStruct
{
 public:
  GO    side_GID;       // global id of side in the mesh
  GO    elem_GID;       // global id of element containing side
  int   ws_elem_idx;    // index of element containing side within this workset
  int   elem_ebIndex;   // index of element block that contains element
  int   side_pos;       // position of side relative to owning element
};
using SideSetList = std::map<std::string, std::vector<SideStruct>>;


// This is a structure that holds all of the sideset information over all worksets. When running populate mesh,
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

  // (num_local_worksets)
  Kokkos::View<int*, Kokkos::LayoutRight> sideset_sizes;

  // (num_local_worksets, max_sideset_length)
  Kokkos::View<GO**, Kokkos::LayoutRight>   side_GID;
  Kokkos::View<GO**, Kokkos::LayoutRight>   elem_GID;
  Kokkos::View<int**, Kokkos::LayoutRight>  ws_elem_idx;
  Kokkos::View<int**, Kokkos::LayoutRight>  elem_ebIndex;
  Kokkos::View<int**, Kokkos::LayoutRight>  side_pos;

  int max_sides;
  // (num_local_worksets, max_sides)
  Kokkos::View<int**, Kokkos::LayoutRight>  numCellsOnSide;

  // (num_local_worksets, max_sides, max_sideset_length)
  Kokkos::View<int***, Kokkos::LayoutRight>   cellsOnSide;
  Kokkos::View<int***, Kokkos::LayoutRight>   sideSetIdxOnSide;
};

using GlobalSideSetList = std::map<std::string, GlobalSideSetInfo>;

class LocalSideSetInfo
{
public:
  int size;
  Kokkos::View<GO*, Kokkos::LayoutRight>    side_GID;      // (size)
  Kokkos::View<GO*, Kokkos::LayoutRight>    elem_GID;      // (size)
  Kokkos::View<int*, Kokkos::LayoutRight>   ws_elem_idx;   // (size)
  Kokkos::View<int*, Kokkos::LayoutRight>   elem_ebIndex;  // (size)
  Kokkos::View<int*, Kokkos::LayoutRight>   side_pos;      // (size)

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
