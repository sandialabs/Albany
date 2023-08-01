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
#include <Omega_h_file.hpp> //write_array, write_parallel
#include <Omega_h_array_ops.hpp> //get_max
#include <Omega_h_atomics.hpp> //atomic_fetch_add

#include <fstream>

namespace Albany {

struct OmegahTetFaceVtx {
  int idx[4][3];
  OmegahTetFaceVtx() {
    const int elem_dim = 3;
    const int bdry_dim = 2;
    const int vtx_dim = 0;
    for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
      for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
        idx[bdry][vert] = Omega_h::simplex_down_template(elem_dim, bdry_dim, bdry, vert);
      }
    }
  }
};

struct ShardsTetFaceVtx {
  int idx[4][3];
  ShardsTetFaceVtx() {
    shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
    const int elem_dim = 3;
    const int bdry_dim = 2;
    const int vtx_dim = 0;
    for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
      for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
        idx[bdry][vert] = tetTopo.getNodeMap(bdry_dim,bdry,vert);
      }
    }
  }
};

struct TetTriVtx {
  int perm[4][3];
  void print(std::string name) {
    std::stringstream ss;
    ss << "\ntet: " << name << "(face,vert)=idx \n";
    for (int i=0; i<4; i++) {
      ss << "(" << i << ", *)=" << " ";
      for (int j=0; j<3; j++) {
        ss << perm[i][j] << " ";
      }
      ss << "\n";
    }
    std::cerr << ss.str();
  }
};

template<typename Source, typename Dest>
TetTriVtx getPerm_tet(Source& source, Dest& dest, const int src2destTetFace[4]) {
  const int elem_dim = 3;
  const int bdry_dim = 2;
  const int vtx_dim = 0;

  //compute the shards to omegah permutation
  TetTriVtx src2destTetTriVtx;
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  auto topoData = triTopo.getCellTopologyData();
  for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
    auto src2destBdry = src2destTetFace[bdry];
    auto perm = shards::findPermutation(topoData, dest.idx[src2destBdry],source.idx[bdry]);
    if(!(perm>=0 && perm<topoData->permutation_count)) {
      fprintf(stderr, "perm %d\n", perm);
    }
    assert(perm>=0 && perm<topoData->permutation_count);
    for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
      src2destTetTriVtx.perm[src2destBdry][vert] = triTopo.getNodePermutationInverse(perm,vert);
    }
  }
  return src2destTetTriVtx;
}

struct OmegahTriVtx {
  int idx[3];
  OmegahTriVtx() {
    const int elem_dim = 2;
    const int vtx_dim = 0;
    const int ignored = -1;
    for(int vert=0; vert<Omega_h::simplex_degree(elem_dim, vtx_dim); vert++) {
      idx[vert] = Omega_h::simplex_down_template(elem_dim, vtx_dim, vert, ignored);
    }
  }
};

struct ShardsTriVtx {
  int idx[3];
  ShardsTriVtx() {
    shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
    const int elem_dim = 2;
    const int bdry_dim = 0;
    const int vtx_dim = 0;
    for(int vert=0; vert<triTopo.getSubcellCount(vtx_dim); vert++) {
      idx[vert] = triTopo.getNodeMap(elem_dim, bdry_dim, vert);
    }
  }
};

struct TriVtx {
  int perm[3];
  void print(std::string name) const {
    std::stringstream ss;
    ss << "\ntri: " << name << "vert(0,1,2)= ";
    for (int i=0; i<3; i++) {
      ss << perm[i] << " ";
    }
    ss << "\n";
    std::cerr << ss.str();
  }
};

template<typename Source, typename Dest>
TriVtx getPerm_tri(Source& source, Dest& dest) {
  const int elm_dim = 2;
  const int vtx_dim = 0;

  //compute the shards to omegah permutation
  TriVtx src2dest;
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  auto perm = shards::findPermutation(triTopo.getCellTopologyData(),dest.idx,source.idx);
  for(int vert=0; vert<Omega_h::simplex_degree(elm_dim,vtx_dim); vert++) {
    src2dest.perm[vert] = triTopo.getNodePermutationInverse(perm,vert);
  }
  return src2dest;
}


struct Omegah2ShardsPerm {
  const std::string name = "Omega_h-to-Shards";

  /////// Triangles ///////// {
  /* triVtx[i] contains the Shards vertex index for the i'th vertex of an Omegah
   * triangle
   */
  const TriVtx triVtx;
  TriVtx getOmegah2ShardsPerm_tri() { //FIXME
    const OmegahTriVtx ohTriVtx;
    const ShardsTriVtx shTriVtx;
    return getPerm_tri(ohTriVtx,shTriVtx);
  }
  ///// END Triangles }

  /////// Tetrahedrons ///////// {
  /**
   * tetFace[i] contains the Shards face index for the i'th Omegah face
   */
  const int tetFace[4] = {3,0,1,2};
  /**
   * tetTriVtx.perm[i][j] contains the Shards vertex index for the j'th vertex of the
   * i'th Omegah triangle, permuted to the correct Shards face via tetFace[i], of a tet.
   * For example, given the 1st vertex of the 2nd Omegah face, get the corresponding Shards
   * vertex index:
   * auto shardsVtxIdx = tetTriVtx.perm[tetFace[1]][0];
   */
  const TetTriVtx tetTriVtx;

  TetTriVtx getOmegah2ShardsPerm_tet() {
    const OmegahTetFaceVtx ohFaceVtx;
    const ShardsTetFaceVtx shFaceVtx;
    return getPerm_tet(ohFaceVtx, shFaceVtx, tetFace);
  }
  ///// END Tetrahedrons }

  Omegah2ShardsPerm() :
    tetTriVtx(getOmegah2ShardsPerm_tet()),
    triVtx(getOmegah2ShardsPerm_tri())
  {}
};

/**
 * this is the inverse of Omegah2ShardsPerm
 */
struct Shards2OmegahPerm {
  const std::string name = "Shards-to-Omega_h";

  /////// Triangles /////////
  const TriVtx triVtx;
  TriVtx getShards2OmegahPerm_tri() { //FIXME
    const ShardsTriVtx shTriVtx;
    const OmegahTriVtx ohTriVtx;
    return getPerm_tri(shTriVtx,ohTriVtx);
  }

  /////// Tetrahedrons /////////
  const int tetFace[4] = {1,2,3,0};
  const TetTriVtx tetTriVtx;

  TetTriVtx getShards2OmegahPerm_tet() {
    const ShardsTetFaceVtx shFaceVtx;
    const OmegahTetFaceVtx ohFaceVtx;
    return getPerm_tet(shFaceVtx, ohFaceVtx, tetFace);
  }
  ///// END Tetrahedrons }

  Shards2OmegahPerm() :
    tetTriVtx(getShards2OmegahPerm_tet()),
    triVtx(getShards2OmegahPerm_tri())
  {}
};





OmegahConnManager::
OmegahConnManager(Omega_h::Mesh& in_mesh) : mesh(in_mesh)
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

  owners = std::vector<Ownership>(1); //FIXME
}

OmegahConnManager::
OmegahConnManager(Omega_h::Mesh& in_mesh, std::string partId, const int partDim) :
  mesh(in_mesh)
{
  auto world = mesh.library()->world();
  auto rank = world->rank();

  assert(partDim <= mesh.dim());

  //TODO this needs to be tested for a tag that has no entries on some processes
  Omega_h::Read<Omega_h::I8> isInPart;
  if(mesh.has_tag(partDim, partId)) {
    isInPart = mesh.get_array<Omega_h::I8>(partDim, partId);
  }
  const auto numInPartGlobal = Omega_h::get_sum(world,isInPart);
  assert(numInPartGlobal > 0);
  std::cerr << rank << " numInPart " << numInPartGlobal << "\n";
  world->barrier();
  std::stringstream ss;
  ss << "Error! Input mesh has no mesh entities of dimension "
     << partDim << " with tag " << partId << "\n";
  auto err = ss.str();
  TEUCHOS_TEST_FOR_EXCEPTION (!numInPartGlobal, std::runtime_error, err.c_str());

  m_elem_blocks_names.push_back(partId);

  assert(mesh.has_tag(mesh.dim(), "global"));

  //create the host array of entities in the part {
  // no attempt to make this 'fast' was made...
  const auto numInPart = Omega_h::get_sum(isInPart);
  Omega_h::Write<Omega_h::LO> localPartEntIds(numInPart);
  Omega_h::Write<Omega_h::LO> index({0},"localEntIndex");
  auto getLocalPartEntIds = OMEGA_H_LAMBDA(int i) {
    if(isInPart[i]) {
      const auto idx = Omega_h::atomic_fetch_add(&index[0],1);
      localPartEntIds[idx] = i;
    }
  };
  const std::string kernelName = "getLocalPartEntIds" + std::to_string(partDim);
  Omega_h::parallel_for(mesh.nents(partDim), getLocalPartEntIds, kernelName.c_str());
  //}

  owners = std::vector<Ownership>(1); //FIXME
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

std::array<LO,4> getDofsPerEnt(const panzer::FieldPattern & fp)
{
  std::array<LO,4> dofsPerEnt = {0,0,0,0};
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
  return dofsPerEnt;
}

Omega_h::GOs createGlobalEntDofNumbering(Omega_h::Mesh& mesh, const LO entityDim, const LO dofsPerEnt, const GO startingOffset) {
  if(!dofsPerEnt)
    return Omega_h::GOs();
  const int numEnts = mesh.nents(entityDim);
  const int numDofs = numEnts*dofsPerEnt;
  auto worldComm = mesh.library()->world();
  const auto entGlobalIds = mesh.globals(entityDim);
  Omega_h::Write<Omega_h::GO> dofNum(numDofs);
  auto setNumber = OMEGA_H_LAMBDA(int i) {
    for(int j=0; j<dofsPerEnt; j++) {
      const auto dofIndex = i*dofsPerEnt+j;
      const auto dofGlobalId = entGlobalIds[i]*dofsPerEnt+j;
      dofNum[dofIndex] = startingOffset+dofGlobalId;
    }
  };
  const std::string kernelName = "setGlobalDofId_entityDim" + std::to_string(entityDim);
  Omega_h::parallel_for(numEnts, setNumber, kernelName.c_str());
  return mesh.sync_array(entityDim, Omega_h::read(dofNum), dofsPerEnt);
}

GO getMaxGlobalEntDofId(Omega_h::Mesh& mesh, Omega_h::GOs& dofGlobalIds) {
  if(!dofGlobalIds.size())
    return 0;
  auto worldComm = mesh.library()->world();
  return Omega_h::get_max(worldComm,dofGlobalIds);
}

std::array<Omega_h::GOs,4> OmegahConnManager::createGlobalDofNumbering() const {
  std::array<Omega_h::GOs,4> gdn;
  GO startingOffset = 0;
  for(int i=0; i<gdn.size(); i++) {
    gdn[i] = createGlobalEntDofNumbering(mesh, i, m_dofsPerEnt[i], startingOffset);
    const auto offset = getMaxGlobalEntDofId(mesh, gdn[i]);
    startingOffset = offset == 0 ? startingOffset : offset;
  }
  return gdn;
}

/**
 * \brief set the mask for each dof for entities of dimension adjDim that bound
 * each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementToEntDofConnectivityMask(Omega_h::Mesh& mesh, Omega_h::Read<Omega_h::I8>& maskArray,
    const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt,
    Omega_h::Write<Omega_h::GO> elm2dof) {
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), mesh.dim(), adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(mesh.dim() != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  Omegah2ShardsPerm oh2sh;
  const auto perm = oh2sh.triVtx.perm;
  auto setMask = OMEGA_H_LAMBDA(int elm) {
    const auto firstDown = elm*numDownAdjEntsPerElm;
    //loop over element-to-ent adjacencies and fill in the dofs
    for(int j=0; j<numDownAdjEntsPerElm; j++) {
      const auto adjEnt = adjEnts[firstDown+j];
      for(int k=0; k<dofsPerEnt; k++) {
        const auto shardsAdjEntIdx = perm[j]; //use the omega_h to shards permutation to convert the omegah j index to shards
        const auto connIdx = (elm*dofsPerElm)+(dofOffset+shardsAdjEntIdx+k);
        elm2dof[connIdx] = maskArray[adjEnt];
      }
    }
  };
  const auto kernelName = "setElementToEntDofConnectivityMask_dim" + std::to_string(mesh.dim());
  Omega_h::parallel_for(mesh.nelems(), setMask, kernelName.c_str());
}

/**
 * \brief set the global dof ids for entities of dimension adjDim that bound
 * each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementToEntDofConnectivity(Omega_h::Mesh& mesh, const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof) {
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), mesh.dim(), adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(mesh.dim() != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  Omegah2ShardsPerm oh2sh;
  const auto perm = oh2sh.triVtx.perm;
  auto setNumber = OMEGA_H_LAMBDA(int elm) {
    const auto firstDown = elm*numDownAdjEntsPerElm;
    //loop over element-to-ent adjacencies and fill in the dofs
    for(int j=0; j<numDownAdjEntsPerElm; j++) {
      const auto adjEnt = adjEnts[firstDown+j];
      for(int k=0; k<dofsPerEnt; k++) {
        const auto dofIndex = adjEnt*dofsPerEnt+k;
        const auto dofGlobalId = globalDofNumbering[dofIndex];
        const auto shardsAdjEntIdx = perm[j]; //use the omega_h to shards permutation to convert the omegah j index to shards
        const auto connIdx = (elm*dofsPerElm)+(dofOffset+shardsAdjEntIdx+k);
        elm2dof[connIdx] = dofGlobalId;
      }
    }
  };
  const auto kernelName = "setElementToEntDofConnectivity_dim" + std::to_string(mesh.dim());
  Omega_h::parallel_for(mesh.nelems(), setNumber, kernelName.c_str());
}

/**
 * \brief set the global dof ids for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivity(Omega_h::Mesh& mesh, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof) {
  auto setNumber = OMEGA_H_LAMBDA(int elm) {
    for(int k=0; k<dofsPerEnt; k++) {
      const auto dofIndex = elm*dofsPerEnt+k;
      const auto dofGlobalId = globalDofNumbering[dofIndex];
      const auto connIdx = (elm*dofsPerElm)+(dofOffset+k);
      elm2dof[connIdx] = dofGlobalId;
    }
  };
  const auto kernelName = "setElementDofConnectivity_dim" + std::to_string(mesh.dim());
  Omega_h::parallel_for(mesh.nelems(), setNumber, kernelName.c_str());
}

/**
 * \brief set the mask for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivityMask(Omega_h::Mesh& mesh, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::Read<Omega_h::I8> mask, Omega_h::Write<Omega_h::GO> elm2dof) {
  auto setMask = OMEGA_H_LAMBDA(int elm) {
    for(int k=0; k<dofsPerEnt; k++) {
      const auto dofIndex = elm*dofsPerEnt+k;
      const auto connIdx = (elm*dofsPerElm)+(dofOffset+k);
      elm2dof[connIdx] = mask[elm];
    }
  };
  const auto kernelName = "setElementDofConnectivityMask_dim" + std::to_string(mesh.dim());
  Omega_h::parallel_for(mesh.nelems(), setMask, kernelName.c_str());
}



LO OmegahConnManager::getConnectivitySize() const
{
  LO dofsPerElm = 0;
  for(int i=0; i<=mesh.dim(); i++) {
    const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), mesh.dim(), i);
    dofsPerElm += m_dofsPerEnt[i]*numDownAdjEntsPerElm;
  }
  dofsPerElm += m_dofsPerEnt[mesh.dim()];
  return dofsPerElm;
}

Omega_h::GOs OmegahConnManager::createElementToDofConnectivityMask(
    const std::string& tagName, const Omega_h::Adj elmToDim[3]) const
{
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  Omega_h::LO totalNumDofs = m_dofsPerElm*mesh.nelems();
  for(int i=mesh.dim()+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<mesh.dim(); adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      Omega_h::Read<Omega_h::I8> maskArray;
      if( mesh.has_tag(adjDim, tagName) ) {
        maskArray = mesh.get_array<Omega_h::I8>(adjDim, tagName);
      } else {
        maskArray = Omega_h::Read<Omega_h::I8>(mesh.nents(adjDim), 0);
      }
      setElementToEntDofConnectivityMask(mesh, maskArray, m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), mesh.dim(), adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[mesh.dim()]) {
    Omega_h::Read<Omega_h::I8> maskArray;
    if( mesh.has_tag(mesh.dim(), tagName) ) {
      maskArray = mesh.get_array<Omega_h::I8>(mesh.dim(), tagName);
    } else {
      maskArray = Omega_h::Read<Omega_h::I8>(mesh.nents(mesh.dim()), 0);
    }
    setElementDofConnectivityMask(mesh, m_dofsPerElm, dofOffset, m_dofsPerEnt[mesh.dim()], maskArray, elm2dof);
  }
  return elm2dof;
}


Omega_h::GOs OmegahConnManager::createElementToDofConnectivity(const Omega_h::Adj elmToDim[3],
    const std::array<Omega_h::GOs,4>& globalDofNumbering) const {
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  Omega_h::LO totalNumDofs = m_dofsPerElm*mesh.nelems();
  for(int i=mesh.dim()+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<mesh.dim(); adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      setElementToEntDofConnectivity(mesh, m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], globalDofNumbering[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), mesh.dim(), adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[mesh.dim()]) {
    setElementDofConnectivity(mesh, m_dofsPerElm, dofOffset, m_dofsPerEnt[mesh.dim()],
        globalDofNumbering[mesh.dim()], elm2dof);
  }
  return elm2dof;
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

  m_dofsPerEnt = getDofsPerEnt(fp);
  m_dofsPerElm = getConnectivitySize();

  m_globalDofNumbering = createGlobalDofNumbering();

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  for(int dim = 0; dim < mesh.dim(); dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh.ask_down(mesh.dim(),dim);
    }
  }

  auto elm2dof = createElementToDofConnectivity(elmToDim, m_globalDofNumbering);
  // transfer to host
  m_connectivity = Omega_h::HostRead(elm2dof);
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

bool OmegahConnManager::contains (const std::string& sub_part_name) const //WILL BE REMOVED
{
  return false;
}

// Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
// local element $ielem belongs to sub part $sub_part_name
bool OmegahConnManager::belongs (const std::string& sub_part_name, //REPLACED BY getConnectivityMask
         const LO ielem, const int subcell_dim, const int subcell_pos) const
{
  return false;
}

// Where element ielem start in the 1d connectivity array
int OmegahConnManager::getConnectivityStart (const LO ielem) const {
  return 0;
}

// Get a mask vector (1=yes, 0=no) telling if each dof entity is contained in the given mesh part
// note, the part can be associated with any dimension of mesh entity
std::vector<int> OmegahConnManager::getConnectivityMask (const std::string& sub_part_name) const {
  bool hasPartTag = false;
  for(int d=0; d<mesh.dim(); d++)
    hasPartTag |= mesh.has_tag(d, sub_part_name);
  std::stringstream ss;
  ss << "Error! Omega_h does not have a tag named \"" << sub_part_name
     << "\" associated with any mesh entity dimension\n";
  TEUCHOS_TEST_FOR_EXCEPTION (!hasPartTag, std::runtime_error, ss.str());
  ss.str(std::string());
  ss << "Error! The Omega_h dofs per element is zero.  Was buildConnectivity(...) called?\n";
  TEUCHOS_TEST_FOR_EXCEPTION (m_dofsPerElm == 0, std::runtime_error, ss.str());

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  for(int dim = 0; dim < mesh.dim(); dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh.ask_down(mesh.dim(),dim);
    }
  }

  auto elm2dofMask = createElementToDofConnectivityMask(sub_part_name, elmToDim);
  // transfer to host
  auto elm2dofMask_h = Omega_h::HostRead(elm2dofMask);
  std::vector<int> elm2dofMask_vec(elm2dofMask_h.data(), elm2dofMask_h.data()+elm2dofMask_h.size());
  assert(elm2dofMask_vec.size() == elm2dofMask_h.size());

  return elm2dofMask_vec;
}

// Queries the dimension of a part
// FIXME - what should be returned here? highest dimension
int OmegahConnManager::part_dim (const std::string&) const //FIXME
{
  return mesh.dim();
}

const Ownership* OmegahConnManager::getOwnership(LO) const //FIXME
{
  return &owners[0];
}

} // namespace Albany
