//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "OmegahConnManager.hpp"
#include "Albany_config.h"

#include "Panzer_FieldPattern.hpp"

#include <Omega_h_element.hpp> //topological_singular_name
#include <Omega_h_for.hpp> //parallel_for

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
    LO entIdCnt[4], GO entOffset[4]) const
{
  // compute ID counts for each sub cell type
  int patternDim = fp.getDimension();
  switch(patternDim) {
    case 3:
      entIdCnt[2] = fp.getSubcellIndices(2,0).size();
      // Intentional fall-through.
    case 2:
      entIdCnt[1] = fp.getSubcellIndices(1,0).size();
      // Intentional fall-through.
    case 1:
      entIdCnt[0] = fp.getSubcellIndices(0,0).size();
      // Intentional fall-through.
    case 0:
      entIdCnt[3] = fp.getSubcellIndices(patternDim,0).size();
      break;
    default:
       TEUCHOS_ASSERT(false);
  };

  // compute offsets for each sub cell type
  entOffset[0] = 0;
  for(int dim=1; dim<4; dim++) {
    entOffset[dim] = entOffset[dim-1]+(mesh.nents(dim-1)+1)*entIdCnt[dim-1];
  }

  // sanity check
  TEUCHOS_ASSERT(entOffset[0] <= entOffset[1]
              && entOffset[1] <= entOffset[2]
              && entOffset[2] <= entOffset[3]);
}

void appendConnectivity(Omega_h::Write<Omega_h::LO>& elmDownAdj_d, Omega_h::Adj elmToDim[3], int dim) {
  const auto startIdx = ( dim == 0 ) ? 0 : elmToDim[dim-1].a2ab.size();
  const auto a2ab = elmToDim[dim].a2ab;
  const auto ab2b = elmToDim[dim].ab2b;
  auto append = OMEGA_H_LAMBDA(LO i) {
    for(auto ab = a2ab[i]; ab < a2ab[i]; ab++) {
      elmDownAdj_d[i+startIdx] = ab2b[ab];
    }
  };
  const std::string kernelName = "appendConnectivity_dim" + std::to_string(dim);
  Omega_h::parallel_for(a2ab.size(), append, kernelName.c_str());
}

void
OmegahConnManager::buildConnectivity(const panzer::FieldPattern &fp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension() > mesh.dim(), std::logic_error,
      "Error! OmegahConnManager Field pattern incompatible with stored elem_blocks.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n"
      "  - elem_blocks topo dim: " + Omega_h::topological_singular_name(mesh.family(), mesh.dim()) + "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension() < 1, std::logic_error,
      "Error! OmegahConnManager Field pattern must have a dimension of at least 1.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n");

  // Build entity adjacency counts and offsets
  //    ID counts = How many IDs belong on each entity (number of mesh DOF used)
  //    Offset = What is starting index for each sub-array of adjacency information
  //             Global numbering goes like [node ids, edge ids, face ids, cell ids]
  LO entIdCnt[4] = {0,0,0,0};
  GO entOffsets[4] = {0,0,0,0};
  buildOffsetsAndIdCounts(fp, entIdCnt, entOffsets);

  // loop over elements and build global connectivity
  const int numElems = mesh.nelems();
  const auto fieldDim = fp.getCellTopology().getDimension();

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  GO totSize = 0;
  for(int dim = 0; dim < fieldDim; dim++) {
    elmToDim[dim] = mesh.ask_down(mesh.dim(),dim);
    totSize += elmToDim[dim].a2ab.size();
  }

  // append the ajacency arrays to each other
  Omega_h::Write<Omega_h::LO> elmDownAdj_d(totSize);
  for(int dim = 0; dim < fieldDim; dim++)
    appendConnectivity(elmDownAdj_d, elmToDim, dim);

  // transfer to host
  Omega_h::HostRead elmDownAdj_h(Omega_h::read(elmDownAdj_d));
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
