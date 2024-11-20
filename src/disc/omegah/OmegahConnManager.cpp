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
Omega_h::Read<Omega_h::I8> getIsEntInPart(const OmegahGenericMesh& albanyMesh, const std::string& part_name) {
  const auto& mesh = albanyMesh.getOmegahMesh();
  const int part_dim = albanyMesh.part_dim(part_name);
  if(part_name == albanyMesh.meshSpecs[0]->ebName) {
    return Omega_h::Read<Omega_h::I8>(mesh->nents(part_dim), 1);
  } else {
    return mesh->get_array<Omega_h::I8>(part_dim, part_name);
  }
}

[[nodiscard]]
Omega_h::LOs numberEntsInPart(const OmegahGenericMesh& albanyMesh, const std::string& part_name) {
  const auto& mesh = albanyMesh.getOmegahMesh();
  auto isInPart = getIsEntInPart(albanyMesh, part_name);
  const int part_dim = albanyMesh.part_dim(part_name);
  auto partEntOffset = Omega_h::offset_scan(isInPart, "partEntIdx");
  auto partEntIdx = Omega_h::Write<Omega_h::LO>(mesh->nents(part_dim));
  Omega_h::parallel_for(mesh->nents(part_dim), OMEGA_H_LAMBDA(int i) {
    if(isInPart[i]) {
      partEntIdx[i] = partEntOffset[i];
    } else {
      partEntIdx[i] = 0;
    }
  });
  return Omega_h::LOs(partEntIdx);
}

[[nodiscard]]
LO getNumEntsInPart(const OmegahGenericMesh& albanyMesh, const std::string& part_name) {
  const auto& mesh = albanyMesh.getOmegahMesh();
  const int part_dim = albanyMesh.part_dim(part_name);
  if(part_name == albanyMesh.meshSpecs[0]->ebName) {
    return mesh->nents(part_dim);
  } else {
    const auto isInPart = getIsEntInPart(albanyMesh,part_name);
    return Omega_h::get_sum<Omega_h::I8>(isInPart);
  }
}

[[nodiscard]]
std::vector<LO> getLocalElmIds(const OmegahGenericMesh& albanyMesh,  const std::string& part_name) {
  const auto& mesh = albanyMesh.getOmegahMesh();
  if(part_name==albanyMesh.meshSpecs[0]->ebName) {
    std::vector<LO> localElmIds(mesh->nelems());
    std::iota(localElmIds.begin(), localElmIds.end(), 0);
    return localElmIds;
  } else {
    auto entPartId = numberEntsInPart(albanyMesh, part_name);
    Omega_h::HostRead<Omega_h::LO> entPartId_h(entPartId);
    return std::vector<LO>(entPartId_h.data(), entPartId_h.data()+entPartId_h.size());
  }
}

OmegahConnManager::
OmegahConnManager(const Teuchos::RCP<OmegahGenericMesh>& in_mesh)
 : OmegahConnManager(in_mesh,
                     in_mesh->meshSpecs[0]->ebName)
{
  // Nothing to do here
}

OmegahConnManager::
OmegahConnManager(const Teuchos::RCP<OmegahGenericMesh>& in_mesh,
                  const std::string& inPartId)
 : albanyMesh(in_mesh)
 , mesh(albanyMesh->getOmegahMesh())
{
  m_elem_blocks_names = {inPartId};

  TEUCHOS_TEST_FOR_EXCEPTION (mesh->dim()!=2 && mesh->dim()!=3, std::logic_error,
      "Error! The OmegahConnManager currently only supports 2d/3d meshes.\n"
      "  - input mesh dim: " + std::to_string(mesh->dim()) + "\n");

  auto world = mesh->library()->world();

  //TODO this needs to be tested for a tag that has no entries on some processes
  auto isEntInPart = getIsEntInPart(*albanyMesh,inPartId);
  const auto numInPartGlobal = Omega_h::get_sum(world,isEntInPart);
  assert(numInPartGlobal > 0);
  world->barrier();
  std::stringstream ss;
  ss << "Error! Input mesh has no mesh entities of dimension "
     << in_mesh->part_dim(inPartId) << " with tag " << inPartId << "\n";
  auto err = ss.str();
  TEUCHOS_TEST_FOR_EXCEPTION (!numInPartGlobal, std::runtime_error, err.c_str());

  assert(mesh->has_tag(in_mesh->part_dim(inPartId), "global"));

  localElmIds = getLocalElmIds(*albanyMesh, inPartId);
}

std::vector<GO>
OmegahConnManager::getElementsInBlock (const std::string&) const
{
  const int dim = albanyMesh->part_dim(elem_block_name());
  assert(mesh->has_tag(dim, "global"));
  auto globals_d = mesh->globals(dim);
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

GO getMaxGlobalEntDofId(const Omega_h::Mesh& mesh, Omega_h::GOs& dofGlobalIds) {
  if(!dofGlobalIds.size())
    return 0;
  auto worldComm = mesh.library()->world();
  return Omega_h::get_max(worldComm,dofGlobalIds);
}

std::array<Omega_h::GOs,4> OmegahConnManager::createGlobalDofNumbering() const {
  std::array<Omega_h::GOs,4> gdn;
  GO startingOffset = 0;
  for(long unsigned int i=0; i<gdn.size(); i++) {
    gdn[i] = createGlobalEntDofNumbering(*mesh, i, m_dofsPerEnt[i], startingOffset);
    const auto offset = getMaxGlobalEntDofId(*mesh, gdn[i]);
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
void setElementToEntDofConnectivityMask(const OmegahGenericMesh& albanyMesh, const std::string& part_name, Omega_h::Read<Omega_h::I8>& maskArray,
    const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt,
    Omega_h::Write<Omega_h::GO> elm2dof)
{
  const auto& mesh = *albanyMesh.getOmegahMesh(); 
  const int part_dim = albanyMesh.part_dim(part_name);
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), part_dim, adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(part_dim != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  auto isInPart = getIsEntInPart(albanyMesh, part_name);
  auto partEntIdx = numberEntsInPart(albanyMesh, part_name);
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
  const auto kernelName = "setElementToEntDofConnectivityMask_dim" + std::to_string(part_dim);
  Omega_h::parallel_for(mesh.nents(part_dim), setMask, kernelName.c_str());
}

/**
 * \brief set the global dof ids for entities of dimension adjDim that bound
 * each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementToEntDofConnectivity(OmegahGenericMesh& albanyMesh, const std::string& part_name, const LO dofsPerElm, const LO adjDim, const LO dofOffset,
    const Omega_h::Adj elmToDim, const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof) {
  auto& mesh = *albanyMesh.getOmegahMesh();
  const int part_dim = albanyMesh.part_dim(part_name);
  const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh.family(), part_dim, adjDim);
  const auto adjEnts = elmToDim.ab2b;
  TEUCHOS_TEST_FOR_EXCEPTION(part_dim != 2 && adjDim != 0, std::logic_error,
      "Error! OmegahConnManager Omega_h-to-Shards permutation only tested for vertices of triangles.\n")
  auto isInPart = getIsEntInPart(albanyMesh, part_name);
  auto partEntIdx = numberEntsInPart(albanyMesh, part_name);
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
  const auto kernelName = "setElementToEntDofConnectivity_dim" + std::to_string(part_dim);
  Omega_h::parallel_for(mesh.nents(part_dim), setNumber, kernelName.c_str());
}

/**
 * \brief set the global dof ids for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivity(const OmegahGenericMesh& albanyMesh, const std::string& part_name, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::GOs globalDofNumbering, Omega_h::Write<Omega_h::GO> elm2dof)
{
  auto isInPart = getIsEntInPart(albanyMesh, part_name);
  auto partEntIdx = numberEntsInPart(albanyMesh, part_name);
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
  const int part_dim = albanyMesh.part_dim(part_name);
  const auto kernelName = "setElementDofConnectivity_dim" + std::to_string(part_dim);

  const auto& mesh = *albanyMesh.getOmegahMesh();
  Omega_h::parallel_for(mesh.nents(part_dim), setNumber, kernelName.c_str());
}

/**
 * \brief set the mask for each element
 *
 * Note, the writes into elm2dof have a non-unit stride and performance will suffer.
 */
void setElementDofConnectivityMask(const OmegahGenericMesh& albanyMesh, const std::string partId, const LO dofsPerElm, const LO dofOffset,
    const LO dofsPerEnt, Omega_h::Read<Omega_h::I8> mask, Omega_h::Write<Omega_h::GO> elm2dof)
{
  auto isInPart = getIsEntInPart(albanyMesh, partId);
  auto partEntIdx = numberEntsInPart(albanyMesh, partId);
  auto setMask = OMEGA_H_LAMBDA(int elm) {
    for(int k=0; k<dofsPerEnt; k++) {
      const auto elmPartIdx = partEntIdx[elm];
      const auto connIdx = (elmPartIdx*dofsPerElm)+(dofOffset+k);
      elm2dof[connIdx] = mask[elm];
    }
  };
  const int partDim = albanyMesh.part_dim(partId);
  const auto kernelName = "setElementDofConnectivityMask_dim" + std::to_string(partDim);
  const auto& mesh = *albanyMesh.getOmegahMesh();
  Omega_h::parallel_for(mesh.nents(partDim), setMask, kernelName.c_str());
}

LO OmegahConnManager::getPartConnectivitySize() const
{
  LO dofsPerElm = 0;
  const int part_dim = this->part_dim();
  for(int i=0; i<part_dim; i++) {
    const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh->family(), part_dim, i);
    dofsPerElm += m_dofsPerEnt[i]*numDownAdjEntsPerElm;
  }
  dofsPerElm += m_dofsPerEnt[mesh->dim()];
  return dofsPerElm;
}

Omega_h::GOs OmegahConnManager::createElementToDofConnectivityMask(
  Omega_h::Read<Omega_h::I8> maskArray[4], const Omega_h::Adj elmToDim[3]) const
{
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  const auto numEntsInPart = getNumEntsInPart(*albanyMesh, elem_block_name());
  const int part_dim = this->part_dim();
  Omega_h::LO totalNumDofs = m_dofsPerElm*numEntsInPart;
  for(int i=part_dim+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<part_dim; adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      setElementToEntDofConnectivityMask(*albanyMesh, this->elem_block_name(), maskArray[adjDim], m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh->family(), part_dim, adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[part_dim]) {
    setElementDofConnectivityMask(*albanyMesh, elem_block_name(), m_dofsPerElm, dofOffset, m_dofsPerEnt[part_dim], maskArray[part_dim], elm2dof);
  }
  return elm2dof;
}

std::vector<Ownership> OmegahConnManager::buildConnectivityOwnership() const {
  std::stringstream ss;
  ss << "Error! The Omega_h dofs per element is zero.  Was buildConnectivity(...) called?\n";
  TEUCHOS_TEST_FOR_EXCEPTION (m_dofsPerElm == 0, std::runtime_error, ss.str());
  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  Omega_h::Read<Omega_h::I8> owned[4];
  const int part_dim = this->part_dim();
  for(int dim = 0; dim < part_dim; dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh->ask_down(part_dim,dim);
      owned[dim] = mesh->owned(dim);
    } else {
      owned[dim] = Omega_h::Read<Omega_h::I8>(mesh->nents(dim), 0);
    }
  }
  if(m_dofsPerEnt[part_dim] > 0)
    owned[part_dim] = mesh->owned(part_dim);
  else
    owned[part_dim] = Omega_h::Read<Omega_h::I8>(mesh->nents(part_dim), 0);

  auto connOwnership = createElementToDofConnectivityMask(owned, elmToDim);
  // transfer to host
  auto connOwnership_h = Omega_h::HostRead(connOwnership);
  std::vector<Ownership> connOwnership_vec(connOwnership_h.size());
  for(long unsigned int i=0; i<connOwnership_vec.size(); i++) {
    if( connOwnership_h[i] )
      connOwnership_vec[i] = Ownership::Owned;
    else
      connOwnership_vec[i] = Ownership::Ghosted;
  }
  return connOwnership_vec;
}

Omega_h::GOs OmegahConnManager::createElementToDofConnectivity(const Omega_h::Adj elmToDim[3],
    const std::array<Omega_h::GOs,4>& globalDofNumbering) const {
  //create array that is numVtxDofs+numEdgeDofs+numFaceDofs+numElmDofs long
  const auto numEntsInPart = getNumEntsInPart(*albanyMesh, elem_block_name());
  Omega_h::LO totalNumDofs = m_dofsPerElm*numEntsInPart;
  const int part_dim = this->part_dim();
  for(int i=part_dim+1; i<4; i++)
    assert(!m_dofsPerEnt[i]);//watch out for stragglers
  Omega_h::Write<Omega_h::GO> elm2dof(totalNumDofs);

  LO dofOffset = 0;
  for(int adjDim=0; adjDim<part_dim; adjDim++) {
    if(m_dofsPerEnt[adjDim]) {
      setElementToEntDofConnectivity(*albanyMesh, elem_block_name(), m_dofsPerElm, adjDim, dofOffset, elmToDim[adjDim],
          m_dofsPerEnt[adjDim], globalDofNumbering[adjDim], elm2dof);
      const auto numDownAdjEntsPerElm = Omega_h::element_degree(mesh->family(), part_dim, adjDim);
      dofOffset += m_dofsPerEnt[adjDim]*numDownAdjEntsPerElm;
    }
  }
  if(m_dofsPerEnt[part_dim]) {
    setElementDofConnectivity(*albanyMesh, elem_block_name(), m_dofsPerElm, dofOffset, m_dofsPerEnt[part_dim],
        globalDofNumbering[part_dim], elm2dof);
  }
  return elm2dof;
}

void
OmegahConnManager::writeConnectivity()
{
  auto world = mesh->library()->world();
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
  const LO dim = fp.getCellTopology().getDimension(); 
  const int part_dim = this->part_dim();
  TEUCHOS_TEST_FOR_EXCEPTION (dim > part_dim, std::logic_error,
      "Error! OmegahConnManager Field pattern incompatible with stored elem_blocks.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n"
      "  - elem_blocks topo dim: " + Omega_h::topological_singular_name(mesh->family(), part_dim) + "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension() < 1, std::logic_error,
      "Error! OmegahConnManager Field pattern must have a dimension of at least 1.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n");

  m_dofsPerEnt = getDofsPerEnt(fp);
  m_dofsPerElm = getPartConnectivitySize();

  m_globalDofNumbering = createGlobalDofNumbering();

  // get element-to-[vertex|edge|face] adjacencies
  Omega_h::Adj elmToDim[3];
  for(int dim = 0; dim < part_dim; dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh->ask_down(part_dim,dim);
    }
  }

  auto elm2dof = createElementToDofConnectivity(elmToDim, m_globalDofNumbering);
  // transfer to host
  m_connectivity = Omega_h::HostRead(elm2dof);

  // build the ownership now that the field pattern is known
  owners = buildConnectivityOwnership();
}

Teuchos::RCP<panzer::ConnManager>
OmegahConnManager::noConnectivityClone() const
{
  //FIXME there may be object ownership/persistence issues here
  return Teuchos::RCP(new OmegahConnManager(albanyMesh)); //FIXME
}

// Where element ielem start in the 1d connectivity array
int OmegahConnManager::getConnectivityStart (const LO localElmtId) const {
  return localElmtId*m_dofsPerElm;
}

// Get a mask vector (1=yes, 0=no) telling if each dof entity is contained in the given mesh part
// note, the part can be associated with any dimension of mesh entity
std::vector<int> OmegahConnManager::getConnectivityMask (const std::string& sub_part_name) const {
  bool hasPartTag = false;
  const int part_dim = this->part_dim();
  for(int d=0; d<part_dim; d++)
    hasPartTag |= mesh->has_tag(d, sub_part_name);
  std::stringstream ss;
  ss << "Error! Omega_h does not have a tag named \"" << sub_part_name
     << "\" associated with any mesh entity dimension\n";
  TEUCHOS_TEST_FOR_EXCEPTION (!hasPartTag, std::runtime_error, ss.str());
  ss.str(std::string());
  ss << "Error! The Omega_h dofs per element is zero.  Was buildConnectivity(...) called?\n";
  TEUCHOS_TEST_FOR_EXCEPTION (m_dofsPerElm == 0, std::runtime_error, ss.str());

  // get element-to-[vertex|edge|face] adjacencies and maskarray
  Omega_h::Adj elmToDim[3];
  Omega_h::Read<Omega_h::I8> mask[4];
  for(int dim = 0; dim < part_dim; dim++) {
    if(m_dofsPerEnt[dim] > 0) {
      elmToDim[dim] = mesh->ask_down(part_dim,dim);
      mask[dim] = mesh->get_array<Omega_h::I8>(dim, sub_part_name);
    } else {
      mask[dim] = Omega_h::Read<Omega_h::I8>(mesh->nents(dim), 0);
    }
  }
  if(m_dofsPerEnt[part_dim] > 0)
    mask[part_dim] = mesh->get_array<Omega_h::I8>(part_dim, sub_part_name);
  else
    mask[part_dim] = Omega_h::Read<Omega_h::I8>(mesh->nents(part_dim), 0);

  auto elm2dofMask = createElementToDofConnectivityMask(mask, elmToDim);
  // transfer to host
  auto elm2dofMask_h = Omega_h::HostRead(elm2dofMask);
  std::vector<int> elm2dofMask_vec(elm2dofMask_h.data(), elm2dofMask_h.data()+elm2dofMask_h.size());
  long unsigned int elm2dofMask_h_size = elm2dofMask_h.size();
  assert(elm2dofMask_vec.size() == elm2dofMask_h_size);

  return elm2dofMask_vec;
}

// Queries the dimension of a part
int OmegahConnManager::part_dim (const std::string& name) const
{
  return albanyMesh->part_dim(name);
}

const Ownership* OmegahConnManager::getOwnership(LO localElmtId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
      "Error! Cannot call getOwnership before connectivity is built.\n");
  return &owners.at(localElmtId*m_dofsPerElm);
}

} // namespace Albany
