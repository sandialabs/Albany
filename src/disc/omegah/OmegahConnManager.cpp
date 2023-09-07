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
#include <Omega_h_int_scan.hpp> //offset_scan

#include "OmegahPermutation.hpp"

#include <fstream>

namespace Albany {

[[nodiscard]]
Omega_h::Read<Omega_h::I8> getIsEntInPart(Omega_h::Mesh& mesh, const OmegahPartFilter filt) {
  if(filt.name == "") {
    return Omega_h::Read<Omega_h::I8>(mesh.nents(filt.dim), 1);
  } else {
    return mesh.get_array<Omega_h::I8>(filt.dim, filt.name);
  }
}

[[nodiscard]]
Omega_h::LOs numberEntsInPart(Omega_h::Mesh& mesh, const OmegahPartFilter filt) {
  auto isInPart = getIsEntInPart(mesh, filt);
  auto partEntOffset = Omega_h::offset_scan(isInPart, "partEntIdx");
  auto partEntIdx = Omega_h::Write<Omega_h::LO>(mesh.nents(filt.dim));
  Omega_h::parallel_for(mesh.nents(filt.dim), OMEGA_H_LAMBDA(int i) {
    if(isInPart[i]) {
      partEntIdx[i] = partEntOffset[i];
    } else {
      partEntIdx[i] = 0;
    }
  });
  return Omega_h::LOs(partEntIdx);
}

[[nodiscard]]
LO getNumEntsInPart(Omega_h::Mesh& mesh, const OmegahPartFilter filt) {
  if(filt.name=="")
    return mesh.nents(filt.dim);
  else {
    const auto isInPart = getIsEntInPart(mesh,filt);
    return Omega_h::get_sum<Omega_h::I8>(isInPart);
  }
}

[[nodiscard]]
std::vector<LO> getLocalElmIds(Omega_h::Mesh& mesh, const OmegahPartFilter filt) {
  if(filt.name=="") {
    std::vector<LO> localElmIds(mesh.nelems());
    std::iota(localElmIds.begin(), localElmIds.end(), 0);
    return localElmIds;
  } else {
    auto entPartId = numberEntsInPart(mesh, filt);
    Omega_h::HostRead<Omega_h::LO> entPartId_h(entPartId);
    return std::vector<LO>(entPartId_h.data(), entPartId_h.data()+entPartId_h.size());
  }
}

OmegahConnManager::
OmegahConnManager(Omega_h::Mesh& in_mesh) : mesh(in_mesh), partFilter({mesh.dim(),""})
{
  //albany does *not* support processes without elements
  TEUCHOS_TEST_FOR_EXCEPTION (!mesh.nelems(), std::runtime_error,
      "Error! Input mesh has no elements!\n");
  TEUCHOS_TEST_FOR_EXCEPTION (mesh.dim()!=2 && mesh.dim()!=3, std::logic_error,
      "Error! The OmegahConnManager currently only supports 2d/3d meshes.\n"
      "  - input mesh dim: " + std::to_string(mesh.dim()) + "\n");

  m_elem_blocks_names.push_back("omegah_mesh_block");

  //the omegah conn manager will be recreated after each topological adaptation
  // - a change to mesh vertex coordinates (mesh motion) will not require
  //   recreating the conn manager
  localElmIds = getLocalElmIds(mesh, partFilter);
  assert(mesh.has_tag(mesh.dim(), "global"));

  owners = std::vector<Ownership>(1); //FIXME
}

OmegahConnManager::
OmegahConnManager(Omega_h::Mesh& in_mesh, std::string inPartId, const int inPartDim) :
  mesh(in_mesh), partFilter({inPartDim, inPartId})
{
  TEUCHOS_TEST_FOR_EXCEPTION (mesh.dim()!=2 && mesh.dim()!=3, std::logic_error,
      "Error! The OmegahConnManager currently only supports 2d/3d meshes.\n"
      "  - input mesh dim: " + std::to_string(mesh.dim()) + "\n");

  auto world = mesh.library()->world();
  auto rank = world->rank();

  assert(partFilter.dim <= mesh.dim());

  //TODO this needs to be tested for a tag that has no entries on some processes
  auto isEntInPart = getIsEntInPart(mesh,partFilter);
  const auto numInPartGlobal = Omega_h::get_sum(world,isEntInPart);
  assert(numInPartGlobal > 0);
  world->barrier();
  std::stringstream ss;
  ss << "Error! Input mesh has no mesh entities of dimension "
     << partFilter.dim << " with tag " << partFilter.name << "\n";
  auto err = ss.str();
  TEUCHOS_TEST_FOR_EXCEPTION (!numInPartGlobal, std::runtime_error, err.c_str());

  m_elem_blocks_names.push_back(partFilter.name);

  assert(mesh.has_tag(partFilter.dim, "global"));

  localElmIds = getLocalElmIds(mesh, partFilter);

  owners = std::vector<Ownership>(1); //FIXME
}

std::vector<GO>
OmegahConnManager::getElementsInBlock (const std::string&) const
{
  assert(mesh.has_tag(partFilter.dim, "global"));
  auto globals_d = mesh.globals(partFilter.dim);
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
void setElementToEntDofConnectivityMask(Omega_h::Mesh& mesh, const OmegahPartFilter filt, Omega_h::Read<Omega_h::I8>& maskArray,
    const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt,
    Omega_h::Write<Omega_h::GO> elm2dof) {
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), filt.dim, adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(filt.dim != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  auto isInPart = getIsEntInPart(mesh, filt);
  auto partEntIdx = numberEntsInPart(mesh, filt);
  OmegahPermutation::Omegah2ShardsPerm oh2sh;
  const auto perm = oh2sh.triVtx.perm;
  const auto totNumDofs = elm2dof.size();
  auto setMask = OMEGA_H_LAMBDA(int elm) {
    if(isInPart[elm]) {
      const auto firstDown = elm*numDownAdjEntsPerElm;
      //loop over element-to-ent adjacencies and fill in the dofs
      for(int j=0; j<numDownAdjEntsPerElm; j++) {
        const auto adjEnt = adjEnts[firstDown+j];
        for(int k=0; k<dofsPerEnt; k++) {
          const auto shardsAdjEntIdx = perm[j]; //use the omega_h to shards permutation to convert the omegah j index to shards
          const auto elmPartIdx = partEntIdx[elm];
          const auto connIdx = (elmPartIdx*dofsPerElm)+(dofOffset+shardsAdjEntIdx+k);
          assert(totNumDofs > connIdx && connIdx >= 0);
          elm2dof[connIdx] = maskArray[adjEnt];
        }
      }
    }
  };
  const auto kernelName = "setElementToEntDofConnectivityMask_dim" + std::to_string(filt.dim);
  Omega_h::parallel_for(mesh.nents(filt.dim), setMask, kernelName.c_str());
}

/**
 * \brief set the global dof ids for entities of dimension adjDim that bound
 * each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementToEntDofConnectivity(Omega_h::Mesh& mesh, const OmegahPartFilter filt, const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof) {
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), filt.dim, adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(filt.dim != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  auto isInPart = getIsEntInPart(mesh, filt);
  auto partEntIdx = numberEntsInPart(mesh, filt);
  OmegahPermutation::Omegah2ShardsPerm oh2sh;
  const auto perm = oh2sh.triVtx.perm;
  const auto totNumDofs = elm2dof.size();
  auto setNumber = OMEGA_H_LAMBDA(int elm) {
    if(isInPart[elm]) {
      const auto firstDown = elm*numDownAdjEntsPerElm;
      //loop over element-to-ent adjacencies and fill in the dofs
      for(int j=0; j<numDownAdjEntsPerElm; j++) {
        const auto adjEnt = adjEnts[firstDown+j];
        for(int k=0; k<dofsPerEnt; k++) {
          const auto shardsAdjEntIdx = perm[j]; //use the omega_h to shards permutation to convert the omegah j index to shards
          const auto elmPartIdx = partEntIdx[elm];
          const auto connIdx = (elmPartIdx*dofsPerElm)+(dofOffset+shardsAdjEntIdx+k);
          assert(totNumDofs > connIdx && connIdx >= 0);
          const auto dofIndex = adjEnt*dofsPerEnt+k;
          const auto dofGlobalId = globalDofNumbering[dofIndex];
          elm2dof[connIdx] = dofGlobalId;
        }
      }
    }
  };
  const auto kernelName = "setElementToEntDofConnectivity_dim" + std::to_string(filt.dim);
  Omega_h::parallel_for(mesh.nents(filt.dim), setNumber, kernelName.c_str());
}

/**
 * \brief set the global dof ids for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivity(Omega_h::Mesh& mesh, const OmegahPartFilter filt, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof) {
  auto isInPart = getIsEntInPart(mesh, filt);
  auto partEntIdx = numberEntsInPart(mesh, filt);
  auto setNumber = OMEGA_H_LAMBDA(int elm) {
    if(isInPart[elm]) {
      for(int k=0; k<dofsPerEnt; k++) {
        const auto dofIndex = elm*dofsPerEnt+k;
        const auto dofGlobalId = globalDofNumbering[dofIndex];
        const auto elmPartIdx = partEntIdx[elm];
        const auto connIdx = (elmPartIdx*dofsPerElm)+(dofOffset+k);
        elm2dof[connIdx] = dofGlobalId;
      }
    }
  };
  const auto kernelName = "setElementDofConnectivity_dim" + std::to_string(filt.dim);
  Omega_h::parallel_for(mesh.nents(filt.dim), setNumber, kernelName.c_str());
}

/**
 * \brief set the mask for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivityMask(Omega_h::Mesh& mesh, const LO partDim, const std::string partId, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::Read<Omega_h::I8> mask, Omega_h::Write<Omega_h::GO> elm2dof) {
  auto isInPart = getIsEntInPart(mesh, {partDim, partId});
  auto partEntIdx = numberEntsInPart(mesh, {partDim, partId});
  auto setMask = OMEGA_H_LAMBDA(int elm) {
    for(int k=0; k<dofsPerEnt; k++) {
      const auto elmPartIdx = partEntIdx[elm];
      const auto connIdx = (elmPartIdx*dofsPerElm)+(dofOffset+k);
      elm2dof[connIdx] = mask[elm];
    }
  };
  const auto kernelName = "setElementDofConnectivityMask_dim" + std::to_string(partDim);
  Omega_h::parallel_for(mesh.nents(partDim), setMask, kernelName.c_str());
}

LO OmegahConnManager::getPartConnectivitySize() const
{
  LO dofsPerElm = 0;
  for(int i=0; i<partFilter.dim; i++) {
    const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), partFilter.dim, i);
    dofsPerElm += m_dofsPerEnt[i]*numDownAdjEntsPerElm;
  }
  dofsPerElm += m_dofsPerEnt[mesh.dim()];
  return dofsPerElm;
}

Omega_h::GOs OmegahConnManager::createElementToDofConnectivityMask(
    const std::string& tagName, const Omega_h::Adj elmToDim[3]) const
{
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  const auto numEntsInPart = getNumEntsInPart(mesh, partFilter);
  Omega_h::LO totalNumDofs = m_dofsPerElm*numEntsInPart;
  for(int i=partFilter.dim+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<partFilter.dim; adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      Omega_h::Read<Omega_h::I8> maskArray;
      if( mesh.has_tag(adjDim, tagName) ) {
        maskArray = mesh.get_array<Omega_h::I8>(adjDim, tagName);
      } else {
        maskArray = Omega_h::Read<Omega_h::I8>(mesh.nents(adjDim), 0);
      }
      setElementToEntDofConnectivityMask(mesh, partFilter, maskArray, m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), partFilter.dim, adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[partFilter.dim]) {
    Omega_h::Read<Omega_h::I8> maskArray;
    if( mesh.has_tag(partFilter.dim, tagName) ) {
      maskArray = mesh.get_array<Omega_h::I8>(partFilter.dim, tagName);
    } else {
      maskArray = Omega_h::Read<Omega_h::I8>(mesh.nents(partFilter.dim), 0);
    }
    setElementDofConnectivityMask(mesh, partFilter.dim, partFilter.name, m_dofsPerElm, dofOffset, m_dofsPerEnt[partFilter.dim], maskArray, elm2dof);
  }
  return elm2dof;
}


Omega_h::GOs OmegahConnManager::createElementToDofConnectivity(const Omega_h::Adj elmToDim[3],
    const std::array<Omega_h::GOs,4>& globalDofNumbering) const {
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  const auto numEntsInPart = getNumEntsInPart(mesh, partFilter);
  Omega_h::LO totalNumDofs = m_dofsPerElm*numEntsInPart;
  for(int i=partFilter.dim+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<partFilter.dim; adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      setElementToEntDofConnectivity(mesh, partFilter, m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], globalDofNumbering[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), partFilter.dim, adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[partFilter.dim]) {
    setElementDofConnectivity(mesh, partFilter, m_dofsPerElm, dofOffset, m_dofsPerEnt[partFilter.dim],
        globalDofNumbering[partFilter.dim], elm2dof);
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
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension() > partFilter.dim, std::logic_error,
      "Error! OmegahConnManager Field pattern incompatible with stored elem_blocks.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n"
      "  - elem_blocks topo dim: " + Omega_h::topological_singular_name(mesh.family(), partFilter.dim) + "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension() < 1, std::logic_error,
      "Error! OmegahConnManager Field pattern must have a dimension of at least 1.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n");

  m_dofsPerEnt = getDofsPerEnt(fp);
  m_dofsPerElm = getPartConnectivitySize();

  m_globalDofNumbering = createGlobalDofNumbering();

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  for(int dim = 0; dim < partFilter.dim; dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh.ask_down(partFilter.dim,dim);
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

// Where element ielem start in the 1d connectivity array
int OmegahConnManager::getConnectivityStart (const LO ielem) const {
  return 0;
}

// Get a mask vector (1=yes, 0=no) telling if each dof entity is contained in the given mesh part
// note, the part can be associated with any dimension of mesh entity
std::vector<int> OmegahConnManager::getConnectivityMask (const std::string& sub_part_name) const {
  bool hasPartTag = false;
  for(int d=0; d<partFilter.dim; d++)
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
  for(int dim = 0; dim < partFilter.dim; dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh.ask_down(partFilter.dim,dim);
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
  return partFilter.dim;
}

const Ownership* OmegahConnManager::getOwnership(LO) const //FIXME
{
  return &owners[0];
}

} // namespace Albany
