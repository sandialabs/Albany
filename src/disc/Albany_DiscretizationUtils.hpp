//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISCRETIZATION_UTILS_HPP
#define ALBANY_DISCRETIZATION_UTILS_HPP

#include "Albany_KokkosTypes.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_CommTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Kokkos_DualView.hpp"
#include "Intrepid2_Basis.hpp"
#include "Teuchos_ArrayRCP.hpp"

#include <map>
#include <string>
#include <vector>

namespace Albany {

// Utility function that uses some integer arithmetic to choose a good worksetSize
int computeWorksetSize(const int worksetSizeMax,
                       const int ebSizeMax);

// This is mostly for Omega_h layer, but may be useful in general while doing refactorings
class NotYetImplemented : public std::runtime_error
{
public:
  NotYetImplemented (const std::string& func_name)
    : std::runtime_error(func_name + " not yet implemented!\n") {}
};

enum class FE_Type {
  HVOL,
  HDIV,
  HCURL,
  HGRAD
};

enum class MeshType {
  Structured,   // structured in all directions
  Extruded,     // structured vertically
  Unstructured  // No structure known (e.g., read from file)
};

// When adapting mesh, we'll return an enum stating what kind of adaptation happened
enum class AdaptationType {
  None,
  Movement,
  Topology
};

struct AdaptationData {
  AdaptationType type = AdaptationType::None;

  // Current value of solution and its time deriv.
  // If adapting, the discretization can interpolate these onto
  // the new mesh.
  Teuchos::RCP<const Thyra_Vector> x;
  Teuchos::RCP<const Thyra_Vector> x_dot;
  Teuchos::RCP<const Thyra_Vector> x_dotdot;
  Teuchos::RCP<const Thyra_MultiVector> dxdp;
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
  Kokkos::DualView<int*, Kokkos::LayoutRight, PHX::Device> sideset_sizes;

  // (num_local_worksets, max_sideset_length)
  Kokkos::DualView<GO**, Kokkos::LayoutRight, PHX::Device>   side_GID;
  Kokkos::DualView<GO**, Kokkos::LayoutRight, PHX::Device>   elem_GID;
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>  ws_elem_idx;
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>  elem_ebIndex;
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>  side_pos;

  int max_sides;
  // (num_local_worksets, max_sides)
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>  numCellsOnSide;

  // (num_local_worksets, max_sides, max_sideset_length)
  Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>   cellsOnSide;
  Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>   sideSetIdxOnSide;
};

using GlobalSideSetList = std::map<std::string, GlobalSideSetInfo>;

class LocalSideSetInfo
{
public:
  int size;
  Kokkos::DualView<GO*, Kokkos::LayoutRight, PHX::Device>    side_GID;      // (size)
  Kokkos::DualView<GO*, Kokkos::LayoutRight, PHX::Device>    elem_GID;      // (size)
  Kokkos::DualView<int*, Kokkos::LayoutRight, PHX::Device>   ws_elem_idx;   // (size)
  Kokkos::DualView<int*, Kokkos::LayoutRight, PHX::Device>   elem_ebIndex;  // (size)
  Kokkos::DualView<int*, Kokkos::LayoutRight, PHX::Device>   side_pos;      // (size)

  int numSides;
  Kokkos::DualView<int*, Kokkos::LayoutRight, PHX::Device>      numCellsOnSide;   // (sides)
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>     cellsOnSide;      // (numSides, sides)
  Kokkos::DualView<int**, Kokkos::LayoutRight, PHX::Device>     sideSetIdxOnSide; // (numSides, sides)
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

// ===== Utilities to serially read data from file ====== //

// Fwd-declare
class CombineAndScatterManager;

Teuchos::RCP<Thyra_MultiVector>
readScalarFileSerial (const std::string& fname,
                      const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                      const Teuchos::RCP<const Teuchos_Comm>& comm);

Teuchos::RCP<Thyra_MultiVector>
readLayeredScalarFileSerial (const std::string &fname,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm);

Teuchos::RCP<Thyra_MultiVector>
readLayeredVectorFileSerial (const std::string &fname,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm);

// Calls one of the above, depending on inputs, and takes care of
// redistributing the vector over MPI ranks, as well as set scaling fators (if requested)
Teuchos::RCP<Thyra_MultiVector>
loadField (const std::string& field_name,
           const Teuchos::ParameterList& field_params,
           const CombineAndScatterManager& cas_manager,
           const Teuchos::RCP<const Teuchos_Comm>& comm,
           bool node, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out,
           std::vector<double>& norm_layers_coords);

Teuchos::RCP<Thyra_MultiVector>
fillField (const std::string& field_name,
           const Teuchos::ParameterList& field_params,
           const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
           bool nodal, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out,
           std::vector<double>& norm_layers_coords);

}  // namespace Albany

#endif  // ALBANY_DISCRETIZATION_UTILS_HPP
