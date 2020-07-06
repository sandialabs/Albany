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

class SideStruct
{
 public:
  GO       side_GID;       // global id of side in the mesh
  GO       elem_GID;       // global id of element containing side
  int      elem_LID;       // local id of element containing side
  int      elem_ebIndex;   // index of element block that contains element
  unsigned side_local_id;  // local id of side relative to owning element
};

class SideStructViews
{
 public:
  Kokkos::View<GO*>       side_GID;
  Kokkos::View<GO*>       elem_GID;
  Kokkos::View<int*>      elem_LID;
  Kokkos::View<int*>      elem_ebIndex;
  Kokkos::View<unsigned*> side_local_id;
  int                     size;
  int                     numSides;
  Kokkos::View<int*>      numCellsOnSide;
  Kokkos::View<int**>     cellsOnSide;
};

using SideSetList = std::map<std::string, std::vector<SideStruct>>;
using SideSetViewList = std::map<std::string, SideStructViews>;

class wsLid
{
 public:
  int ws;   // workset of element containing side
  int LID;  // local id of element containing side
};

using WsLIDList = std::map<GO, wsLid>;

template <typename T>
struct WorksetArray
{
  using type = Teuchos::ArrayRCP<T>;
};

// LB 8/17/18: I moved these out of AbstractDiscretization, so if one only needs
// these types,
//             he/she can include this small file rather than
//             Albany_AbstractDiscretization.hpp, which has tons of
//             dependencies.
using WorksetConn = Kokkos::View<LO***, Kokkos::LayoutRight, PHX::Device>;
using Conn        = WorksetArray<WorksetConn>::type;

}  // namespace Albany

#endif  // ALBANY_DISCRETIZATION_UTILS_HPP
