//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_BlockedSTKDiscretization.hpp"
#include "STKConnManager.hpp"

#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_String_Utilities.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <limits>

namespace Albany
{

template <typename Intrepid2Type>
Teuchos::RCP<const panzer::FieldPattern> buildFieldPattern()
{
  // build a geometric pattern from a single basis
  Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = Teuchos::rcp(new Intrepid2Type);
  Teuchos::RCP<const panzer::FieldPattern> pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  return pattern;
}

BlockedSTKDiscretization::BlockedSTKDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList> &discParams_,
    Teuchos::RCP<AbstractSTKMeshStruct> &stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm> &comm_,
    const Teuchos::RCP<RigidBodyModes> &rigidBodyModes_,
    const std::map<int, std::vector<std::string>> &sideSetEquations_)
    : discParams(discParams_), out(Teuchos::VerboseObjectBase::getDefaultOStream()), comm(comm_)
{

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;
  typedef double Scalar;

  Teuchos::RCP<Teuchos::ParameterList> bDiscParams = Teuchos::sublist(discParams, "Discretization", true);

  sideName = bDiscParams->get<std::string>("Side Name", "None");
  if (sideName != "None")
  {
    hasSide = true;
    n_m_blocks = 2;
  }
  else
  {
    hasSide = false;
    n_m_blocks = 1;
  }

  int n_DiscParamsBlocks = bDiscParams->get<int>("Num Blocks");

  m_blocks.resize(n_m_blocks);

  Teuchos::RCP<AbstractSTKMeshStruct> ssSTKMeshStruct_ = Teuchos::null;

  m_blocks[0] = Teuchos::rcp(new disc_type(discParams, 1, stkMeshStruct_, comm_,
                                           rigidBodyModes_, sideSetEquations_));

  if (hasSide)
  {
    ssSTKMeshStruct_ = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(stkMeshStruct_)->sideSetMeshStructs[sideName];

    m_blocks[1] = Teuchos::rcp(new disc_type(discParams, 1, ssSTKMeshStruct_, comm_,
                                             rigidBodyModes_, sideSetEquations_));
  }

  // build the connection manager
  stkConnMngrVolume = Teuchos::rcp(new Albany::STKConnManager(stkMeshStruct_));

  if (hasSide)
    stkConnMngrSide = Teuchos::rcp(new Albany::STKConnManager(ssSTKMeshStruct_));

  Teuchos::RCP<panzer::ConnManager> connMngrVolume, connMngrSide;

  connMngrVolume = stkConnMngrVolume;
  if (hasSide)
    connMngrSide = stkConnMngrSide;

  // build the DOF manager for the problem
  if (const Teuchos::MpiComm<int> *mpiComm = dynamic_cast<const Teuchos::MpiComm<int> *>(comm.get()))
  {
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

    blockedDOFManagerVolume = Teuchos::rcp(new panzer::BlockedDOFManager(connMngrVolume, rawComm));
    if (hasSide)
      blockedDOFManagerSide = Teuchos::rcp(new panzer::BlockedDOFManager(connMngrSide, rawComm));

    // by default assume orientations are not required
    bool orientationsRequired = false;

    // set orientations required flag
    blockedDOFManagerVolume->setOrientationsRequired(orientationsRequired);
    if (hasSide)
      blockedDOFManagerSide->setOrientationsRequired(orientationsRequired);

    // blocked degree of freedom manager
    std::string fieldOrder = discParams->sublist("Solution").get<std::string>("blocks names");
    std::string fieldDiscretization = discParams->sublist("Solution").get<std::string>("blocks discretizations");

    std::vector<std::vector<std::string>> blocks;
    std::vector<std::vector<std::string>> blocksDiscretizationName;

    std::vector<std::vector<std::string>> blocksVolume;
    std::vector<std::vector<std::string>> blocksSide;

    buildNewBlocking(fieldOrder, blocks);
    buildNewBlocking(fieldDiscretization, blocksDiscretizationName);

    std::vector<shards::CellTopology> elementBlockTopologiesVolume;
    std::vector<std::string> elementBlockNamesVolume;

    std::vector<shards::CellTopology> elementBlockTopologiesSide;
    std::vector<std::string> elementBlockNamesSide;

    stkConnMngrVolume->getElementBlockTopologies(elementBlockTopologiesVolume);
    stkConnMngrVolume->getElementBlockNames(elementBlockNamesVolume);

    if (hasSide)
    {
      stkConnMngrSide->getElementBlockTopologies(elementBlockTopologiesSide);
      stkConnMngrSide->getElementBlockNames(elementBlockNamesSide);
    }

    std::vector<int> idBlocksVolume;
    std::vector<int> idBlocksSide;

    for (size_t i = 0; i < blocks.size(); ++i)
    {
      std::vector<std::string> currentBlocksVolume;
      std::vector<std::string> currentBlocksSide;

      bool previousFieldsVolume = false;
      bool previousFieldsSide = false;

      std::string currentType = "None";
      std::string currentMesh = "None";

      int n_dofs_per_field_per_element;

      for (size_t j = 0; j < blocks[i].size(); ++j)
      {
        std::string type, mesh, domain;
        shards::CellTopology eb_topology;
        for (int i_block = 0; i_block < n_DiscParamsBlocks; ++i_block)
        {
          if (blocksDiscretizationName[i][j] == bDiscParams->sublist(util::strint("Block", i_block)).get<std::string>("Name"))
          {
            type = bDiscParams->sublist(util::strint("Block", i_block)).get<std::string>("FE Type");
            mesh = bDiscParams->sublist(util::strint("Block", i_block)).get<std::string>("Mesh", "Element Block 0");
            domain = bDiscParams->sublist(util::strint("Block", i_block)).get<std::string>("Domain", "Volume");
            break;
          }
        }
        if (domain == "Volume")
        {
          TEUCHOS_TEST_FOR_EXCEPTION(previousFieldsSide, std::logic_error,
                                     "Error! Cannot have fields defined in the volume and on the side for the same block.\n");

          for (size_t i_ebn = 0; i_ebn < elementBlockNamesVolume.size(); ++i_ebn)
          {
            if (elementBlockNamesVolume[i_ebn] == mesh)
            {
              eb_topology = elementBlockTopologiesVolume[i_ebn];
              break;
            }
          }

          RCP<const panzer::FieldPattern> pattern;
          std::string topo_name = eb_topology.getName();
          if (type == "HVOL_C0")
          {
            Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis =
                Teuchos::rcp(new Intrepid2::Basis_HVOL_C0_FEM<PHX::exec_space, double, double>(eb_topology));
            pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
            n_dofs_per_field_per_element = 1;
          }
          if (type == "HGRAD_C1")
          {
            if (topo_name == "Quadrilateral_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 4;
            }
            else if (topo_name == "Triangle_3")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 3;
            }
            else if (topo_name == "Tetrahedron_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 4;
            }
            else if (topo_name == "Hexahedron_8")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 8;
            }
            else
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unsupported topology \"" << topo_name << "\".\n");
          }
          else if (type == "HGRAD_C2")
          {
            if (topo_name == "Quadrilateral_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 9;
            }
            else if (topo_name == "Triangle_3")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 6;
            }
            else if (topo_name == "Tetrahedron_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TET_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 10;
            }
            else if (topo_name == "Hexahedron_8")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_HEX_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 27;
            }
            else
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unsupported topology \"" << topo_name << "\".\n");
          }
          blockedDOFManagerVolume->addField(mesh, blocks[i][j], pattern);

          currentBlocksVolume.push_back(blocks[i][j]);

          stkConnMngrVolume->buildConnectivity(*pattern);

          previousFieldsVolume = true;
        }
        if (domain == "Side")
        {
          TEUCHOS_TEST_FOR_EXCEPTION(!hasSide, std::logic_error,
                                     "Error! Cannot have fields defined on the side as no side name has been provided.\n");

          TEUCHOS_TEST_FOR_EXCEPTION(previousFieldsVolume, std::logic_error,
                                     "Error! Cannot have fields defined in the volume and on the side for the same block.\n");

          for (size_t i_ebn = 0; i_ebn < elementBlockNamesSide.size(); ++i_ebn)
          {
            if (elementBlockNamesSide[i_ebn] == mesh)
            {
              eb_topology = elementBlockTopologiesSide[i_ebn];
              break;
            }
          }

          RCP<const panzer::FieldPattern> pattern;
          std::string topo_name = eb_topology.getName();
          if (type == "HVOL_C0")
          {
            Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis =
                Teuchos::rcp(new Intrepid2::Basis_HVOL_C0_FEM<PHX::exec_space, double, double>(eb_topology));
            pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
            n_dofs_per_field_per_element = 1;
          }
          if (type == "HGRAD_C1")
          {
            if (topo_name == "Quadrilateral_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 4;
            }
            else if (topo_name == "Triangle_3")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 3;
            }
            else
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unsupported topology \"" << topo_name << "\".\n");
          }
          else if (type == "HGRAD_C2")
          {
            if (topo_name == "Quadrilateral_4")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 9;
            }
            else if (topo_name == "Triangle_3")
            {
              pattern = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C2_FEM<PHX::exec_space, double, double>>();
              n_dofs_per_field_per_element = 6;
            }
            else
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unsupported topology \"" << topo_name << "\".\n");
          }
          blockedDOFManagerSide->addField(mesh, blocks[i][j], pattern);

          currentBlocksSide.push_back(blocks[i][j]);

          stkConnMngrSide->buildConnectivity(*pattern);

          previousFieldsSide = true;
        }

        fieldToElementBlockID[blocks[i][j]] = mesh;

        if (currentType == "None")
          currentType = type;
        else
          TEUCHOS_TEST_FOR_EXCEPTION(currentType != type, std::logic_error,
                                     "Error! Cannot have more than one type for the same block.\n");

        if (currentMesh == "None")
          currentMesh = mesh;
        else
          TEUCHOS_TEST_FOR_EXCEPTION(currentMesh != mesh, std::logic_error,
                                     "Error! Cannot have more than one mesh for the same block.\n");
      }

      if (previousFieldsVolume)
        isBlockVolume.push_back(true);
      else
        isBlockVolume.push_back(false);

      blocksVolume.push_back(currentBlocksVolume);
      if (hasSide)
        blocksSide.push_back(currentBlocksSide);

      fadLengths.push_back(blocks[i].size() * n_dofs_per_field_per_element);
    }

    blockedDOFManagerVolume->setFieldOrder(blocksVolume);
    if (hasSide)
      blockedDOFManagerSide->setFieldOrder(blocksSide);

    n_f_blocks = blockedDOFManagerVolume->getNumFieldBlocks();

    blockedDOFManagerVolume->buildGlobalUnknowns();
    blockedDOFManagerVolume->printFieldInformation(*out);

    if (hasSide)
    {
      blockedDOFManagerSide->buildGlobalUnknowns();
      blockedDOFManagerSide->printFieldInformation(*out);
    }
  }
}

void
BlockedSTKDiscretization::computeProductVectorSpaces()
{
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_vs(n_f_blocks);
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_node_vs(n_f_blocks);
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_overlap_vs(n_f_blocks);
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_overlap_node_vs(n_f_blocks);

  const std::vector<Teuchos::RCP<panzer::GlobalIndexer>> &subManagersVolume =
      blockedDOFManagerVolume->getFieldDOFManagers();

  std::vector<Teuchos::RCP<panzer::GlobalIndexer>> subManagersSide;
  if (hasSide)
    subManagersSide = blockedDOFManagerSide->getFieldDOFManagers();

  for (size_t i_block = 0; i_block < n_f_blocks; ++i_block)
  {
    std::vector<Tpetra_GO> indices, ov_indices, ghost_indices;
    std::vector<GO> t_indices, t_ov_indices;

    if (isBlockVolume[i_block])
    {
      subManagersVolume[i_block]->getOwnedIndices(indices);
      subManagersVolume[i_block]->getGhostedIndices(ghost_indices);
    }
    else if (hasSide)
    {
      subManagersSide[i_block]->getOwnedIndices(indices);
      subManagersSide[i_block]->getGhostedIndices(ghost_indices);
    }

    ov_indices.resize(indices.size() + ghost_indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
      ov_indices[i] = indices[i];
    for (size_t i = 0; i < ghost_indices.size(); ++i)
      ov_indices[indices.size() + i] = ghost_indices[i];

    // To be removed when once Tpetra_GO == GO
    // Start
    for (auto i : indices)
      t_indices.push_back(i);
    for (auto i : ov_indices)
      t_ov_indices.push_back(i);
    // To be removed when once Tpetra_GO == GO
    // End

    m_vs[i_block] = Albany::createVectorSpace(comm, t_indices);
    m_overlap_vs[i_block] = Albany::createVectorSpace(comm, t_ov_indices);
  }

  m_pvs = Thyra::productVectorSpace<ST>(m_vs);
  m_overlap_pvs = Thyra::productVectorSpace<ST>(m_overlap_vs);
}

void
BlockedSTKDiscretization::computeGraphs()
{
  m_jac_factory =
      Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_pvs,
                                                    m_pvs));
  m_overlap_jac_factory =
      Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_overlap_pvs,
                                                    m_overlap_pvs));

  for (size_t i_block = 0; i_block < n_f_blocks; ++i_block)
  {
    for (size_t j_block = 0; j_block < n_f_blocks; ++j_block)
    {
      this->computeGraphs(i_block, j_block);
    }
  }

  m_jac_factory->fillComplete();
  m_overlap_jac_factory->fillComplete();
}

int BlockedSTKDiscretization::getBlockFADLength(const size_t i_block)
{
  return fadLengths[i_block];
}

void
BlockedSTKDiscretization::computeGraphs(const size_t i_block, const size_t j_block)
{
  const std::vector<Teuchos::RCP<panzer::GlobalIndexer>> &subManagersVolume =
      blockedDOFManagerVolume->getFieldDOFManagers();

  std::vector<Teuchos::RCP<panzer::GlobalIndexer>> subManagersSide;
  if (hasSide)
    subManagersSide = blockedDOFManagerSide->getFieldDOFManagers();

  Teuchos::RCP<panzer::GlobalIndexer> manager_i_block, manager_j_block;

  // First, we check the block i and j are defined in the volume or
  // on the side and we get their corresponding manager:
  if (isBlockVolume[i_block])
    manager_i_block = subManagersVolume[i_block];
  else
    manager_i_block = subManagersSide[i_block];
  if (isBlockVolume[j_block])
    manager_j_block = subManagersVolume[j_block];
  else
    manager_j_block = subManagersSide[j_block];

  // Moreover, we check if they are both defined in the volume (or on the side):
  bool bothVolume = isBlockVolume[i_block] && isBlockVolume[j_block];
  bool bothSide = !isBlockVolume[i_block] && !isBlockVolume[j_block];

  // We get the corresponding domain and range vector spaces:
  Teuchos::RCP<const Thyra_VectorSpace> domain_vs = this->getVectorSpace(j_block);
  Teuchos::RCP<const Thyra_VectorSpace> range_vs = this->getVectorSpace(i_block);
  Teuchos::RCP<const Thyra_VectorSpace> ov_domain_vs = this->getOverlapVectorSpace(j_block);
  Teuchos::RCP<const Thyra_VectorSpace> ov_range_vs = this->getOverlapVectorSpace(i_block);

  // A new Thyra Crs Matrix Factory is created for the current block:
  Teuchos::RCP<ThyraCrsMatrixFactory> m_current_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(
      domain_vs, range_vs, ov_domain_vs, ov_range_vs));

  // And the newly created factory is set in the block factory:
  m_jac_factory->setBlockFactory(i_block, j_block, m_current_jac_factory);

  // If i and j are defined in the volume (or on the side), there is no need to
  // use the mapping between local element ID on the side with local element ID in the volume:
  if (bothVolume || bothSide)
  {
    std::vector<std::string> elementBlockID_i_block;
    std::vector<std::string> elementBlockID_j_block;

    // As we restricted ourselves to case where all the fields defined on a same block share
    // a same element block, we can access its name using the fieldToElementBlockID map and
    // the name of the first field of the block:
    std::string blockID = fieldToElementBlockID[manager_i_block->getFieldString(0)];
    std::string blockID_j = fieldToElementBlockID[manager_j_block->getFieldString(0)];

    // If the block IDs associated to i and j are not the same, there will be no entries for the current block:
    if (blockID == blockID_j)
    {
      // First, we get the local element IDs of the element block:
      std::vector<int> elementBlock;
      if (bothVolume)
        elementBlock = stkConnMngrVolume->getElementBlock(blockID);
      if (bothSide)
        elementBlock = stkConnMngrSide->getElementBlock(blockID);

      // Then, we loop over the element local IDs of the element block:
      for (int elem_local_id : elementBlock)
      {
        std::vector<Tpetra_GO> gids_i, gids_j;

        // For a given element local ID, we use the DOF manager to access
        // the global IDs of the DOFs related to the current element:
        manager_i_block->getElementGIDs(elem_local_id, gids_i);
        manager_j_block->getElementGIDs(elem_local_id, gids_j);

        // To be removed when once Tpetra_GO == GO
        // Start
        std::vector<GO> cols;
        for (auto gids_j_j : gids_j)
          cols.push_back(gids_j_j);
        // To be removed when once Tpetra_GO == GO
        // End

        // Then, regardless of if the DOFs are owned or not, we loop over the gids_i and
        // we add entries to the graph:
        for (size_t gids_i_index = 0; gids_i_index < gids_i.size(); ++gids_i_index)
          m_current_jac_factory->insertGlobalIndices(gids_i[gids_i_index], Teuchos::arrayViewFromVector(cols));
      }
    }
  }
  else
  {
    // In this branch, one block is defined in the volume and one block is defined on the side.

    // For the sake of shortness, we use i_volume as the index related to the volume and j_side for
    // the index related to the side:
    Teuchos::RCP<panzer::GlobalIndexer> manager_i_volume, manager_j_side;

    manager_i_volume = isBlockVolume[i_block] ? manager_i_block : manager_j_block;
    manager_j_side = isBlockVolume[i_block] ? manager_j_block : manager_i_block;

    std::string blockID = fieldToElementBlockID[manager_j_side->getFieldString(0)];

    std::vector<int> elementBlock = stkConnMngrSide->getElementBlock(blockID);

    // This time, we loop over the elements on the side (as all the elements of the side have one related
    // element in the volume):
    for (int elem_local_id_side : elementBlock)
    {
      // We get the element local ID of the element in the volume associated to the current element
      // of the side:
      int elem_local_id_volume = localSSElementIDtoVolElementID[elem_local_id_side];

      std::vector<Tpetra_GO> gids_i, gids_j;

      int elem_local_id_i = isBlockVolume[i_block] ? elem_local_id_volume : elem_local_id_side;
      int elem_local_id_j = isBlockVolume[i_block] ? elem_local_id_side : elem_local_id_volume;

      // For a given element local ID, we use the DOF manager to access
      // the global IDs of the DOFs related to the current element:
      manager_i_block->getElementGIDs(elem_local_id_i, gids_i);
      manager_j_block->getElementGIDs(elem_local_id_j, gids_j);

      // To be removed when once Tpetra_GO == GO
      // Start
      std::vector<GO> cols;
      for (auto gids_j_j : gids_j)
        cols.push_back(gids_j_j);
      // To be removed when once Tpetra_GO == GO
      // End

      // Then, regardless of if the DOFs are owned or not, we loop over the gids_i and
      // we add entries to the graph:
      for (size_t gids_i_index = 0; gids_i_index < gids_i.size(); ++gids_i_index)
        m_current_jac_factory->insertGlobalIndices(gids_i[gids_i_index], Teuchos::arrayViewFromVector(cols));
    }
  }
}

void
BlockedSTKDiscretization::printConnectivity() const
{
  for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    this->printConnectivity(i_block);
}
void
BlockedSTKDiscretization::printConnectivity(const size_t i_block) const
{
  m_blocks[i_block]->printConnectivity();
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getNodeVectorSpace() const
{
  return this->getNodeVectorSpace(0);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getNodeVectorSpace(const size_t i_block) const
{
  return m_blocks[i_block]->getNodeVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapNodeVectorSpace() const
{
  return this->getOverlapNodeVectorSpace(0);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapNodeVectorSpace(const size_t i_block) const
{
  return m_blocks[i_block]->getOverlapNodeVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getVectorSpace() const
{
  return this->getVectorSpace(0);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getVectorSpace(const size_t i_block) const
{
  return m_pvs->getBlock(i_block);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapVectorSpace() const
{
  return this->getOverlapVectorSpace(0);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapVectorSpace(const size_t i_block) const
{
  return m_overlap_pvs->getBlock(i_block);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getVectorSpace(const std::string &field_name) const
{
  return this->getVectorSpace(0, field_name);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getVectorSpace(const size_t i_block, const std::string &field_name) const
{
  return m_blocks[i_block]->getVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getNodeVectorSpace(const std::string &field_name) const
{
  return this->getNodeVectorSpace(0, field_name);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getNodeVectorSpace(const size_t i_block, const std::string &field_name) const
{
  return m_blocks[i_block]->getNodeVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapVectorSpace(const std::string &field_name) const
{
  return this->getOverlapVectorSpace(0, field_name);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapVectorSpace(const size_t i_block, const std::string &field_name) const
{
  return m_blocks[i_block]->getOverlapVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapNodeVectorSpace(
    const std::string &field_name) const
{
  return this->getOverlapNodeVectorSpace(0, field_name);
}
Teuchos::RCP<const Thyra_VectorSpace>
BlockedSTKDiscretization::getOverlapNodeVectorSpace(const size_t i_block,
                                                    const std::string &field_name) const
{
  return m_blocks[i_block]->getOverlapNodeVectorSpace(field_name);
}

void
BlockedSTKDiscretization::printCoords() const
{
  for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    this->printCoords(i_block);
}
void
BlockedSTKDiscretization::printCoords(const size_t i_block) const
{
  m_blocks[i_block]->printCoords();
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
BlockedSTKDiscretization::transformMesh()
{
  for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    this->transformMesh(i_block);
}
void
BlockedSTKDiscretization::transformMesh(const size_t i_block)
{
  m_blocks[i_block]->transformMesh();
}

void
BlockedSTKDiscretization::updateMesh()
{
  for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    this->updateMesh(i_block);

  if (hasSide)
  {
    // Compute the mapping from local element ID defined on the side to
    // local element ID in the volume.

    // This mapping is computed using the STK connection manager and assumes
    // that the distribution of the side and volume elements are consistent;
    // if an MPI process owns a side element, it must own the corresponding
    // volume element.

    std::vector<stk::mesh::Entity> sides;
    stkConnMngrVolume->getAllSides(sideName, sides);

    localSSElementIDtoVolElementID.resize(sides.size());

    for (auto side : sides)
    {
      //Get EntityId of sides
      stk::mesh::EntityId eId = stkConnMngrVolume->elementEntityId(side);

      const std::size_t local_volume_id = stkConnMngrVolume->get_parent_cell_id(side);

      int local_side_id = stkConnMngrSide->elementLocalId(eId);

      localSSElementIDtoVolElementID[local_side_id] = local_volume_id;
    }
  }

  computeProductVectorSpaces();

  computeGraphs();
}

void
BlockedSTKDiscretization::updateMesh(const size_t i_block)
{
  m_blocks[i_block]->updateMesh();
}

#if 0
void createExodusFile(const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
                    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory,
                    Teuchos::RCP<panzer_stk::STK_Interface> mesh,
                    const bool & exodus_out) {
for(std::size_t i=0;i<physicsBlocks.size();i++) {
  Teuchos::RCP<panzer::PhysicsBlock> pb = physicsBlocks[i]; // we are assuming only one physics block

  const std::vector<panzer::StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

  // insert all fields into a set
  std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp> fieldNames;
  fieldNames.insert(blockFields.begin(),blockFields.end());

  // build string for modifiying vectors
  std::vector<std::string> dimenStr(3);
  dimenStr[0] = "X"; dimenStr[1] = "Y"; dimenStr[2] = "Z";

  // add basis to DOF manager: block specific
  std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp>::const_iterator fieldItr;
  for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr) {
    Teuchos::RCP<const panzer::PureBasis> basis = fieldItr->second;
    if(basis->getElementSpace()==panzer::PureBasis::HGRAD)
      mesh->addSolutionField(fieldItr->first,pb->elementBlockID());
    else if(basis->getElementSpace()==panzer::PureBasis::CONST )
      mesh->addCellField(fieldItr->first,pb->elementBlockID());
    else if(basis->getElementSpace()==panzer::PureBasis::HCURL ||
        basis->getElementSpace()==panzer::PureBasis::HDIV    ) {
      for(int dim=0;dim<basis->dimension();++dim)
        mesh->addCellField(fieldItr->first+dimenStr[dim],pb->elementBlockID());
    }
  }

  std::vector<std::string> block_names;
  mesh->getElementBlockNames(block_names);

  Teuchos::ParameterList output_pl("Output");
  output_pl.sublist("Cell Average Quantities");
  Teuchos::ParameterList& cell_avg_v = output_pl.sublist("Cell Average Vectors");
  cell_avg_v.set(block_names[0],"CURRENT");
  output_pl.sublist("Cell Quantities");
  output_pl.sublist("Nodal Quantities");
  output_pl.sublist("Allocate Nodal Quantities");
  addFieldsToMesh(*mesh,output_pl);
}
mesh_factory->completeMeshConstruction(*mesh,MPI_COMM_WORLD);

if (exodus_out)
  mesh->setupExodusFile("mesh_output.exo");
}
#endif

void addFieldsToMesh(STKDiscretization &mesh,
                     const Teuchos::ParameterList &output_list)
{
  // register cell averaged scalar fields
  const Teuchos::ParameterList &cellAvgQuants = output_list.sublist("Cell Average Quantities");
  for (Teuchos::ParameterList::ConstIterator itr = cellAvgQuants.begin();
       itr != cellAvgQuants.end(); ++itr)
  {
    const std::string &blockId = itr->first;
    const std::string &fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens, fields, ",", true);

    for (std::size_t i = 0; i < tokens.size(); i++)
      mesh.addCellField(tokens[i], blockId);
  }

  // register cell averaged components of vector fields
  // just allocate space for the fields here. The actual calculation and writing
  // are done by panzer_stk::ScatterCellAvgVector.
  const Teuchos::ParameterList &cellAvgVectors = output_list.sublist("Cell Average Vectors");
  for (Teuchos::ParameterList::ConstIterator itr = cellAvgVectors.begin();
       itr != cellAvgVectors.end(); ++itr)
  {
    const std::string &blockId = itr->first;
    const std::string &fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens, fields, ",", true);

    for (std::size_t i = 0; i < tokens.size(); i++)
    {
      std::string d_mod[3] = {"X", "Y", "Z"};
      for (std::size_t d = 0; d < mesh.getDimension(); d++)
        mesh.addCellField(tokens[i] + d_mod[d], blockId);
    }
  }

  // register cell quantities
  const Teuchos::ParameterList &cellQuants = output_list.sublist("Cell Quantities");
  for (Teuchos::ParameterList::ConstIterator itr = cellQuants.begin();
       itr != cellQuants.end(); ++itr)
  {
    const std::string &blockId = itr->first;
    const std::string &fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens, fields, ",", true);

    for (std::size_t i = 0; i < tokens.size(); i++)
      mesh.addCellField(tokens[i], blockId);
  }

  // register ndoal quantities
  const Teuchos::ParameterList &nodalQuants = output_list.sublist("Nodal Quantities");
  for (Teuchos::ParameterList::ConstIterator itr = nodalQuants.begin();
       itr != nodalQuants.end(); ++itr)
  {
    const std::string &blockId = itr->first;
    const std::string &fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens, fields, ",", true);

    for (std::size_t i = 0; i < tokens.size(); i++)
      mesh.addSolutionField(tokens[i], blockId);
  }

  const Teuchos::ParameterList &allocNodalQuants = output_list.sublist("Allocate Nodal Quantities");
  for (Teuchos::ParameterList::ConstIterator itr = allocNodalQuants.begin();
       itr != allocNodalQuants.end(); ++itr)
  {
    const std::string &blockId = itr->first;
    const std::string &fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens, fields, ",", true);

    for (std::size_t i = 0; i < tokens.size(); i++)
      mesh.addSolutionField(tokens[i], blockId);
  }
}

bool BlockedSTKDiscretization::
    requiresBlocking(const std::string &fieldOrder)
{
  std::vector<std::string> tokens;

  // break it up on spaces
  panzer::StringTokenizer(tokens, fieldOrder, " ", true);

  if (tokens.size() < 2) // there has to be at least 2 tokens to block
    return false;

  // check the prefix - must signal "blocked"
  if (tokens[0] != "blocked:")
    return false;

  // loop over tokens
  bool acceptsHyphen = false;
  for (std::size_t i = 1; i < tokens.size(); i++)
  {

    // acceptsHyphen can't be false, and then a hyphen accepted
    TEUCHOS_TEST_FOR_EXCEPTION(tokens[i] == "-" && !acceptsHyphen, std::logic_error,

                               "Blocked assembly: Error \"Field Order\" hyphen error at "
                               "token "
                                   << i);

    if (acceptsHyphen && tokens[i] == "-")
      acceptsHyphen = false;
    else
    { // token must be a field
      acceptsHyphen = true;
    }
  }

  return true;
}

void BlockedSTKDiscretization::
    buildBlocking(const std::string &fieldOrder, std::vector<std::vector<std::string>> &blocks)
{
  // now we don't have to check
  TEUCHOS_ASSERT(requiresBlocking(fieldOrder));

  std::vector<std::string> tokens;

  // break it up on spaces
  panzer::StringTokenizer(tokens, fieldOrder, " ", true);

  Teuchos::RCP<std::vector<std::string>> current;
  for (std::size_t i = 1; i < tokens.size(); i++)
  {

    if (tokens[i] != "-" && tokens[i - 1] != "-")
    {
      // if there is something to add, add it to the blocks
      if (current != Teuchos::null)
        blocks.push_back(*current);

      current = Teuchos::rcp(new std::vector<std::string>);
    }

    if (tokens[i] != "-")
      current->push_back(tokens[i]);
  }

  if (current != Teuchos::null)
    blocks.push_back(*current);
}

std::string formatFieldName(const std::string &fieldName)
{
  std::string str = fieldName;

  std::string::iterator new_end =
      std::unique(str.begin(), str.end(),
                  [=](char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); });
  str.erase(new_end, str.end());

  if (str.back() == ' ')
    str.pop_back();

  return str;
}

void verifyFieldOrderFormat(const std::string &fieldOrder)
{
  std::string str = fieldOrder;
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());

  int bracket_level = 0;
  for (size_t i = 0; i < str.size(); ++i)
  {
    if (str[i] == '[')
      ++bracket_level;
    if (str[i] == ']')
      --bracket_level;
    TEUCHOS_TEST_FOR_EXCEPTION(bracket_level < 0, std::logic_error,
                               "Error! A closing bracket is located befor an opening one.\n");
    TEUCHOS_TEST_FOR_EXCEPTION(bracket_level > 2, std::logic_error,
                               "Error! Cannot have more than two level of nested blocks.\n");

    if (i > 0 && str[i - 1] == ',')
    {
      TEUCHOS_TEST_FOR_EXCEPTION(str[i] == ',', std::logic_error,
                                 "Error! Two consecutive coma.\n");
    }
    if (i > 0 && str[i - 1] == '[')
    {
      TEUCHOS_TEST_FOR_EXCEPTION(str[i] == ']', std::logic_error,
                                 "Error! Empty block.\n");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(bracket_level != 0, std::logic_error,
                             "Error! Not all blocks are closed.\n");
}

void BlockedSTKDiscretization::
    buildNewBlocking(const std::string &fieldOrder, std::vector<std::vector<std::string>> &blocks)
{
  verifyFieldOrderFormat(fieldOrder);

  Teuchos::RCP<std::vector<std::string>> current;

  std::string::size_type nextOpenBracket, lastCloseBracket, nextCloseBracket, nextComa, pos, lastPos;

  lastPos = fieldOrder.find_first_of("[");
  lastCloseBracket = fieldOrder.find_last_of("]");
  ++lastPos;

  do
  {
    current = Teuchos::rcp(new std::vector<std::string>);

    // Check if the current block has sublock
    nextOpenBracket = fieldOrder.find_first_of("[", lastPos);
    nextCloseBracket = fieldOrder.find_first_of("]", lastPos);
    nextComa = fieldOrder.find_first_of(",", lastPos);

    bool has_sublock = nextOpenBracket < nextComa ? true : false;

    if (has_sublock)
    {
      lastPos = fieldOrder.find_first_not_of("[, ", nextOpenBracket);
      bool last_subblock = false;

      while (true)
      {

        pos = fieldOrder.find_first_of(",", lastPos);

        if (pos > nextCloseBracket)
        {
          last_subblock = true;
          pos = nextCloseBracket;
        }

        current->push_back(formatFieldName(fieldOrder.substr(lastPos, pos - lastPos)));

        lastPos = fieldOrder.find_first_not_of("], ", pos);

        if (last_subblock)
          break;
      }
    }
    else
    {
      pos = fieldOrder.find_first_of("],", lastPos);

      current->push_back(formatFieldName(fieldOrder.substr(lastPos, pos - lastPos)));
      lastPos = fieldOrder.find_first_not_of(", ", pos);
    }
    blocks.push_back(*current);
    nextCloseBracket = fieldOrder.find_first_of("]", lastPos);
  } while (lastPos < lastCloseBracket);
}

} // namespace Albany
