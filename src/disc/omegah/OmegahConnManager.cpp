//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "OmegahConnManager.hpp"
#include "Albany_config.h"

#include "Panzer_FieldPattern.hpp"

#include <Omega_h_element.hpp> //topological_singular_name

namespace Albany {

OmegahConnManager::
OmegahConnManager(Omega_h::Mesh in_mesh) : mesh(in_mesh)
{
  //albany does *not* support processes without elements
  TEUCHOS_TEST_FOR_EXCEPTION (!mesh.nelems(), std::runtime_error,
      "Error! Input mesh has no elements!\n");

  m_elem_blocks_names.push_back("omegah_mesh_block");

  //the omegah conn manager will be recreated after each topological adaptation
  // - a change to mesh vertex coordinates (mesh motion) will not require
  //   recreating the conn manager
  initLocalElmIds();
  assert(mesh.has_tag(mesh.dim(), "global"));
}

std::vector<GO>
OmegahConnManager::getElementsInBlock (const std::string&) const
{
  assert(mesh.has_tag(mesh.dim(), "global"));
  auto globals_d = mesh.globals(mesh.dim());
  Omega_h::HostRead<Omega_h::GO> globalElmIds_h(globals_d);
  return std::vector<GO>(
      globalElmIds_h.data(),
      globalElmIds_h.data()+globalElmIds_h.size());
}

void OmegahConnManager::buildOffsetsAndIdCounts(
    const panzer::FieldPattern & fp,
    LO & nodeIdCnt, LO & edgeIdCnt,
    LO & faceIdCnt, LO & cellIdCnt,
    GO & nodeOffset, GO & edgeOffset,
    GO & faceOffset, GO & cellOffset) const
{
  // compute ID counts for each sub cell type
  int patternDim = fp.getDimension();
  switch(patternDim) {
    case 3:
      faceIdCnt = fp.getSubcellIndices(2,0).size();
      fprintf(stderr, "faceIdCnt %d\n", faceIdCnt);
      // Intentional fall-through.
    case 2:
      edgeIdCnt = fp.getSubcellIndices(1,0).size();
      fprintf(stderr, "edgeIdCnt %d\n", edgeIdCnt);
      // Intentional fall-through.
    case 1:
      nodeIdCnt = fp.getSubcellIndices(0,0).size();
      fprintf(stderr, "nodeIdCnt %d\n", nodeIdCnt);
      // Intentional fall-through.
    case 0:
      cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
      break;
    default:
       TEUCHOS_ASSERT(false);
  };

  // compute offsets for each sub cell type
  nodeOffset = 0;
  edgeOffset = nodeOffset+(mesh.nverts()+1)*nodeIdCnt;
  faceOffset = edgeOffset+(mesh.nedges()+1)*edgeIdCnt;
  cellOffset = faceOffset+(mesh.nfaces()+1)*faceIdCnt;

  // sanity check
  TEUCHOS_ASSERT(nodeOffset <= edgeOffset
              && edgeOffset <= faceOffset
              && faceOffset <= cellOffset);
}


void
OmegahConnManager::buildConnectivity(const panzer::FieldPattern &fp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension()>mesh.dim(), std::logic_error,
      "Error! OmegahConnManager Field pattern incompatible with stored elem_blocks.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n"
      "  - elem_blocks topo dim: " + Omega_h::topological_singular_name(mesh.family(), mesh.dim()) + "\n");

  // Build entity adjacency counts and offsets
  //    ID counts = How many IDs belong on each entity (number of mesh DOF used)
  //    Offset = What is starting index for each sub-array of adjacency information
  //             Global numbering goes like [node ids, edge ids, face ids, cell ids]
  LO nodeIdCnt=0, edgeIdCnt=0, faceIdCnt=0, cellIdCnt=0;
  GO nodeOffset=0, edgeOffset=0, faceOffset=0, cellOffset=0;
  buildOffsetsAndIdCounts(fp, nodeIdCnt,  edgeIdCnt,  faceIdCnt,  cellIdCnt,
                              nodeOffset, edgeOffset, faceOffset, cellOffset);
}

Teuchos::RCP<panzer::ConnManager>
OmegahConnManager::noConnectivityClone() const
{
  //- for stk this function copies the object without connectivity information
  //- for omegah there is little to no distinction as the mesh object contains the
  //  connectivity ... unless the host std::vectors for connectivity are 'stored'
  return Teuchos::RCP(new OmegahConnManager(mesh));
}

const std::vector<LO>&
OmegahConnManager::getAssociatedNeighbors(const LO& /* el */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Error! Albany does not use elements halos in the mesh, so the method\n"
      "       'OmegahConnManager::getAssociatedNeighbors' should not have been called.\n");

  static std::vector<LO> ret;
  return ret;
}

bool OmegahConnManager::contains (const std::string& sub_part_name) const //FIXME
{
  return false;
}

// Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
// local element $ielem belongs to sub part $sub_part_name
bool OmegahConnManager::belongs (const std::string& sub_part_name, //FIXME
         const LO ielem, const int subcell_dim, const int subcell_pos) const
{
  return false;
}

// Queries the dimension of a part
int OmegahConnManager::part_dim (const std::string&) const
{
  return mesh.dim();
}

} // namespace Albany
