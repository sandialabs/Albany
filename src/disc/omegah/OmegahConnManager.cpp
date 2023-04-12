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
#include <Omega_h_file.hpp> //write_array

#include <fstream>

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

void OmegahConnManager::getDofsPerEnt(const panzer::FieldPattern & fp, LO dofsPerEnt[4]) const
{
  // compute ID counts for each sub cell type
  int patternDim = fp.getDimension();
  switch(patternDim) {
    case 3:
      dofsPerEnt[2] = fp.getSubcellIndices(2,0).size();
      // Intentional fall-through.
    case 2:
      dofsPerEnt[1] = fp.getSubcellIndices(1,0).size();
      // Intentional fall-through.
    case 1:
      dofsPerEnt[0] = fp.getSubcellIndices(0,0).size();
      // Intentional fall-through.
    case 0:
      dofsPerEnt[3] = fp.getSubcellIndices(patternDim,0).size();
      break;
    default:
       TEUCHOS_ASSERT(false);
  };

  // sanity check
  TEUCHOS_ASSERT(dofsPerEnt[0] || dofsPerEnt[1] || dofsPerEnt[2] || dofsPerEnt[3]);
}

void OmegahConnManager::getConnectivityOffsets(LO fieldDim, const Omega_h::Adj elmToDim[3],
    const LO dofsPerEnt[4], GO connectivityOffsets[4], GO connectivityGlobalOffsets[4])
{
  // compute offsets for each sub cell type
  connectivityOffsets[0] = 0;
  connectivityGlobalOffsets[0] = 0;
  for(int dim=1; dim<fieldDim; dim++) {
    const auto numDownAdj = elmToDim[dim-1].ab2b.size();
    connectivityOffsets[dim] = connectivityOffsets[dim-1]+(numDownAdj*dofsPerEnt[dim-1]);
  }
  // compute totalsize
  Omega_h::GO totSize = 0;
  for(int dim=0; dim<fieldDim; dim++) {
    totSize += connectivityOffsets[dim];
  }
  // get the starting id on this rank so that the dofs on this rank will have
  // globally unique ids
  auto worldComm = mesh.library()->world();
  const auto globalOffset = worldComm->exscan(totSize, OMEGA_H_SUM);
  fprintf(stderr, "\n%d totSize %d globalOffset %d\n", worldComm->rank(), totSize, globalOffset);
  // create the global offsets for each group of local dof ids
  // associated with vtx, edge, face, and elements
  for(int dim=0; dim<fieldDim; dim++) {
    connectivityGlobalOffsets[dim] = globalOffset + connectivityOffsets[dim];
  }
  for(int rank=0; rank<worldComm->size(); rank++) {
    if(worldComm->rank() == rank) {
      fprintf(stderr, "\n");
      for(int dim=0; dim<fieldDim; dim++) {
        fprintf(stderr, "%d dim co cgo %d %d %d\n",
            rank, dim, connectivityOffsets[dim], connectivityGlobalOffsets[dim]);
      }
    }
    worldComm->barrier();
  }
}

void OmegahConnManager::appendConnectivity(const Omega_h::Adj& elmToDim, LO dofsPerEnt,
    GO startIdx, GO globalStartIdx, LO dim, Omega_h::Write<Omega_h::GO>& elmDownAdj_d) const
{
  auto worldComm = mesh.library()->world();
  const auto rank = worldComm->rank();
  const auto ab2b = elmToDim.ab2b; //values array
  TEUCHOS_ASSERT(mesh.family() == OMEGA_H_SIMPLEX);
  const auto numDownAdjEnts= Omega_h::element_degree(mesh.family(), mesh.dim(), dim);
  fprintf(stderr, "\n%d dim startIdx globalStartIdx ab2b.size() numDown %d %d %d %d %d\n",
                     rank, dim, startIdx, globalStartIdx, ab2b.size(), numDownAdjEnts);
  auto append = OMEGA_H_LAMBDA(LO elm) {
    for(int i=0; i<numDownAdjEnts; i++) {
      const auto downEntIdx = elm*numDownAdjEnts+i;
      const auto downEntId = ab2b[downEntIdx];
      for(int dof=0; dof<dofsPerEnt; dof++) {
        const auto idx = startIdx + (downEntIdx*dofsPerEnt) + dof;
        const GO id = globalStartIdx + (dofsPerEnt * downEntId) + dof;
        printf("%d 0.3 %d %d %d %d\n", rank, elm, downEntIdx, idx, id);
        elmDownAdj_d[idx] = id;
      }
    }
  };
  const std::string kernelName = "appendConnectivity_dim" + std::to_string(dim);
  Omega_h::parallel_for(mesh.nelems(), append, kernelName.c_str());
  Kokkos::fence();
}

void
OmegahConnManager::writeConnectivity()
{
  auto world = mesh.library()->world();
  auto rank = world->rank();
  std::stringstream ss;
  ss << "m_connectivity_rank" << rank << ".txt";
  std::ofstream out(ss.str(), std::ios::out);
  for(int i=0; i<m_connectivity.size(); i++)
    out << m_connectivity[i] << " ";
  out << "\n";
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
  LO dofsPerEnt[4] = {0,0,0,0};
  getDofsPerEnt(fp, dofsPerEnt);

  // loop over elements and build global connectivity
  const int numElems = mesh.nelems();
  const auto fieldDim = fp.getCellTopology().getDimension();

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  for(int dim = 0; dim < fieldDim; dim++) {
    elmToDim[dim] = mesh.ask_down(mesh.dim(),dim);
  }

  GO connectivityOffsets[4] = {0,0,0,0};
  GO connectivityGlobalOffsets[4] = {0,0,0,0};
  getConnectivityOffsets(fieldDim, elmToDim, dofsPerEnt, connectivityOffsets, connectivityGlobalOffsets);
  GO totSize = 0;
  for(int dim = 0; dim < fieldDim; dim++)
    totSize+=connectivityOffsets[dim];

  // append the ajacency arrays to each other
  Omega_h::Write<Omega_h::GO> elmDownAdj_d(totSize);
  for(int dim = 0; dim < fieldDim; dim++)
    appendConnectivity(elmToDim[dim], dofsPerEnt[dim],
                       connectivityOffsets[dim], connectivityGlobalOffsets[dim],
                       dim, elmDownAdj_d);

  // transfer to host
  m_connectivity = Omega_h::HostRead(Omega_h::read(elmDownAdj_d));
}

Teuchos::RCP<panzer::ConnManager>
OmegahConnManager::noConnectivityClone() const
{
  //FIXME there may be object ownership/persistence issues here
  return Teuchos::RCP(new OmegahConnManager(mesh)); //FIXME
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
