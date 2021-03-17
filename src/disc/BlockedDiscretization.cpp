//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_String_Utilities.hpp"

#include "BlockedDiscretization.hpp"
#include "STKConnManager.hpp"

#include "Thyra_DefaultProductVectorSpace.hpp"

namespace Albany
{

  //Assume we only have one block right now
  BlockedDiscretization::BlockedDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList> &discParams_,
      Teuchos::RCP<AbstractSTKMeshStruct> &stkMeshStruct_,
      const Teuchos::RCP<const Teuchos_Comm> &comm_,
      const Teuchos::RCP<RigidBodyModes> &rigidBodyModes_,
      const std::map<int, std::vector<std::string>> &sideSetEquations_)
      : out(Teuchos::VerboseObjectBase::getDefaultOStream()), comm(comm_)
  {

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcp_dynamic_cast;
    typedef double Scalar;

    n_m_blocks = 1;

    m_blocks.resize(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      m_blocks[i_block] = Teuchos::rcp(new disc_type(discParams_, stkMeshStruct_, comm_,
                                                     rigidBodyModes_, sideSetEquations_));

    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager>
        connMngr = Teuchos::rcp(new Albany::STKConnManager(stkMeshStruct_));

    // build the DOF manager for the problem
    if (const Teuchos::MpiComm<int> *mpiComm = dynamic_cast<const Teuchos::MpiComm<int> *>(comm.get()))
    {
      MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

      Teuchos::RCP<panzer::BlockedDOFManager> dofManager = Teuchos::rcp(new panzer::BlockedDOFManager(connMngr, rawComm));
      //   dofManager->enableTieBreak(useTieBreak_);
      //   dofManager->setUseDOFManagerFEI(useDOFManagerFEI_);

      // by default assume orientations are not required
      bool orientationsRequired = false;

#if 0 // disc-fe?
   std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator physIter;
   for(physIter=physicsBlocks.begin();physIter!=physicsBlocks.end();++physIter) {
      Teuchos::RCP<const panzer::PhysicsBlock> pb = *physIter;

     const std::vector<StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

      // insert all fields into a set
      std::set<StrPureBasisPair,StrPureBasisComp> fieldNames;
      fieldNames.insert(blockFields.begin(),blockFields.end());

      // add basis to DOF manager: block specific
      std::set<StrPureBasisPair,StrPureBasisComp>::const_iterator fieldItr;
      for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr) {
         // determine if orientations are required
         orientationsRequired |= fieldItr->second->requiresOrientations();

         Teuchos::RCP< Intrepid2::Basis<PHX::Device::execution_space,double,double> > intrepidBasis
               = fieldItr->second->getIntrepid2Basis();
         Teuchos::RCP<Intrepid2FieldPattern> fp = Teuchos::rcp(new Intrepid2FieldPattern(intrepidBasis));
         dofManager->addField(pb->elementBlockID(),fieldItr->first,fp);
      }
   }
#endif

      // set orientations required flag
      dofManager->setOrientationsRequired(orientationsRequired);

      // blocked degree of freedom manager
      std::string fieldOrder = discParams_->get<std::string>("Field Order");
      std::vector<std::vector<std::string>> blocks;
      buildBlocking(fieldOrder, blocks);
      dofManager->setFieldOrder(blocks);

      dofManager->buildGlobalUnknowns();
      dofManager->printFieldInformation(*out);

#if 0
    // blocked degree of freedom manager
    std::string fieldOrder = discParams_->get<std::string>("Field Order");
    RCP<panzer::GlobalIndexer > dofManager 
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager,fieldOrder);

    // auxiliary dof manager
    std::string auxFieldOrder = discParams_->get<std::string>("Auxiliary Field Order");
    RCP<panzer::GlobalIndexer > auxDofManager 
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),
           auxPhysicsBlocks,conn_manager,auxFieldOrder);
#endif
    }
  }

  BlockedDiscretization::BlockedDiscretization(
      const Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList>> &discParams_,
      Teuchos::Array<Teuchos::RCP<AbstractSTKMeshStruct>> &stkMeshStruct_,
      const Teuchos::RCP<const Teuchos_Comm> &comm_,
      const Teuchos::RCP<RigidBodyModes> &rigidBodyModes_,
      const std::map<int, std::vector<std::string>> &sideSetEquations_)
      : out(Teuchos::VerboseObjectBase::getDefaultOStream()), comm(comm_)
  {

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcp_dynamic_cast;
    typedef double Scalar;

    n_m_blocks = discParams_.length();

    m_blocks.resize(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      m_blocks[i_block] = Teuchos::rcp(new disc_type(discParams_[i_block], stkMeshStruct_[i_block], comm_,
                                                     rigidBodyModes_, sideSetEquations_));

    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager>
        connMngr = Teuchos::rcp(new Albany::STKConnManager(stkMeshStruct_[0]));

    // build the DOF manager for the problem
    if (const Teuchos::MpiComm<int> *mpiComm = dynamic_cast<const Teuchos::MpiComm<int> *>(comm.get()))
    {
      MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

      Teuchos::RCP<panzer::BlockedDOFManager> dofManager = Teuchos::rcp(new panzer::BlockedDOFManager(connMngr, rawComm));
      //   dofManager->enableTieBreak(useTieBreak_);
      //   dofManager->setUseDOFManagerFEI(useDOFManagerFEI_);

      // by default assume orientations are not required
      bool orientationsRequired = false;

      // set orientations required flag
      dofManager->setOrientationsRequired(orientationsRequired);

      // blocked degree of freedom manager
      std::string fieldOrder = discParams_[0]->get<std::string>("Field Order");
      std::vector<std::vector<std::string>> blocks;
      buildBlocking(fieldOrder, blocks);
      dofManager->setFieldOrder(blocks);

      //dofManager->buildGlobalUnknowns();
      //dofManager->printFieldInformation(*out);
    }
  }

  void
  BlockedDiscretization::computeProductVectorSpaces()
  {
    Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_vs(n_m_blocks);
    Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_node_vs(n_m_blocks);
    Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_overlap_vs(n_m_blocks);
    Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> m_overlap_node_vs(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      m_vs[i_block] = this->getVectorSpace(i_block);
      m_node_vs[i_block] = this->getNodeVectorSpace(i_block);
      m_overlap_vs[i_block] = this->getOverlapVectorSpace(i_block);
      m_overlap_node_vs[i_block] = this->getOverlapNodeVectorSpace(i_block);
    }

    m_pvs = Thyra::productVectorSpace<ST>(m_vs);
    m_node_pvs = Thyra::productVectorSpace<ST>(m_node_vs);
    m_overlap_pvs = Thyra::productVectorSpace<ST>(m_overlap_vs);
    m_overlap_node_pvs = Thyra::productVectorSpace<ST>(m_overlap_node_vs);
  }

  void
  BlockedDiscretization::computeGraphs()
  {
    m_jac_factory =
        Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_pvs,
                                                      m_pvs));
    m_overlap_jac_factory =
        Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_overlap_pvs,
                                                      m_overlap_pvs));

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      for (size_t j_block = 0; j_block < n_m_blocks; ++j_block)
        this->computeGraphs(i_block, j_block);

    m_jac_factory->fillComplete();
    m_overlap_jac_factory->fillComplete();
  }

  void
  BlockedDiscretization::computeGraphs(const size_t i_block, const size_t j_block)
  {
    if (i_block == j_block)
    {
      //m_blocks[i_block]->computeGraphs();

      stk::mesh::Selector select_owned_in_part =
          stk::mesh::Selector(this->getSTKMetaData(i_block).universal_part()) &
          stk::mesh::Selector(this->getSTKMetaData(i_block).locally_owned_part());

      std::vector<stk::mesh::Entity> cells;
      stk::mesh::get_selected_entities(
          select_owned_in_part,
          this->getSTKBulkData(i_block).buckets(stk::topology::ELEMENT_RANK),
          cells);

      if (comm->getRank() == 0)
        *out << "BlockedDiscretization: " << cells.size() << " elements on Proc 0 " << std::endl;

      GO row, col;
      Teuchos::ArrayView<GO> colAV;

      const unsigned int neq = m_blocks[i_block]->getNumEq();

      // determining the equations that are defined on the whole domain
      std::vector<int> globalEqns;
      for (unsigned int k(0); k < neq; ++k)
      {
        if (sideSetEquations.find(k) == sideSetEquations.end())
        {
          globalEqns.push_back(k);
        }
      }

      // The global solution dof manager, to get the correct dof id (interleaved vs blocked)
      const auto dofMgr = m_blocks[i_block]->getOverlapDOFManager(solution_dof_name());
      for (const auto &e : cells)
      {
        stk::mesh::Entity const *node_rels = this->getSTKBulkData(i_block).begin_nodes(e);
        const size_t num_nodes = this->getSTKBulkData(i_block).num_nodes(e);

        // loop over local nodes
        for (std::size_t j = 0; j < num_nodes; j++)
        {
          stk::mesh::Entity rowNode = node_rels[j];

          // loop over global eqs
          for (std::size_t k = 0; k < globalEqns.size(); ++k)
          {
            row = dofMgr.getGlobalDOF(stk_gid(i_block, rowNode), globalEqns[k]);
            for (std::size_t l = 0; l < num_nodes; l++)
            {
              stk::mesh::Entity colNode = node_rels[l];
              for (std::size_t m = 0; m < globalEqns.size(); ++m)
              {
                col = dofMgr.getGlobalDOF(stk_gid(i_block, colNode), globalEqns[m]);
                colAV = Teuchos::arrayView(&col, 1);
                m_jac_factory->insertGlobalIndices(row, colAV, i_block, j_block);
              }
            }
          }
          // For sideset equations, we set a diagonal jacobian outside the side set.
          // Namely, we will set res=solution outside the side set (not res=0, otherwise
          // jac is singular).
          // Note: if this node happens to be on the side set, we will add the entry
          //       again in the next loop. But that's fine, cause ThyraCrsMatrixFactory
          //       is storing GIDs of each row in a std::set (until fill complete time).
          for (const auto &it : sideSetEquations)
          {
            int eq = it.first;
            row = dofMgr.getGlobalDOF(stk_gid(i_block, rowNode), eq);
            colAV = Teuchos::arrayView(&row, 1);
            m_jac_factory->insertGlobalIndices(row, colAV, i_block, j_block);
          }
        }
      }

      /*
      if (sideSetEquations.size() > 0)
      {
        const auto lmn = getLayeredMeshNumbering();
        const auto &nodeDofStruct = nodalDOFsStructContainer.getDOFsStruct(nodes_dof_name());
        const auto &ov_node_indexer = nodeDofStruct.overlap_node_vs_indexer;
        const int numOverlapNodes = ov_node_indexer->getNumLocalElements();

        // iterator over all sideSet-defined equations
        for (const auto &it : sideSetEquations)
        {
          // Get the eq number
          int eq = it.first;

          // In case we only have equations on side sets (no "volume" eqns),
          // there would be problem with linear solvers. To avoid this, we
          // put one diagonal entry for every side set equation.
          // NOTE: some nodes will be processed twice, but this is safe:
          //       the redundant indices will be discarded
          for (int inode = 0; inode < numOverlapNodes; ++inode)
          {
            const GO node_gid = ov_node_indexer->getGlobalElement(inode);
            row = dofMgr.getGlobalDOF(node_gid, eq);
            colAV = Teuchos::arrayView(&row, 1);
            m_jac_factory->insertGlobalIndices(row, colAV, i_block, j_block);
          }

          // Do a first loop on all sideset, to establish whether column couplling is allowed.
          // We store the sides while we're at it, to avoid redoing it later
          // Note: column coupling means that 1) the mesh is layered, and 2) the ss eqn is
          //       defined ONLY on side sets at the top or bottom.
          bool allowColumnCoupling = !lmn.is_null();
          std::map<std::string, std::vector<stk::mesh::Entity>> all_sides;
          GO baseId, iLayer;
          for (const auto &ss_name : it.second)
          {
            stk::mesh::Part &part =
                *stkMeshStruct->ssPartVec.find(ss_name)->second;

            // Get all owned sides in this side set
            stk::mesh::Selector select_owned_in_sspart =
                stk::mesh::Selector(part) &
                stk::mesh::Selector(this->getSTKMetaData(i_block).locally_owned_part());

            auto &sides = all_sides[ss_name];
            stk::mesh::get_selected_entities(
                select_owned_in_sspart,
                this->getSTKBulkData(i_block).buckets(this->getSTKMetaData(i_block).side_rank()),
                sides); // store the result in "sides"

            if (allowColumnCoupling && sides.size() > 0)
            {
              const auto &side = sides[0];
              const auto &node = this->getSTKBulkData(i_block).begin_nodes(side)[0];
              lmn->getIndices(stk_gid(i_block, node), baseId, iLayer);
              allowColumnCoupling = (iLayer == 0 || iLayer == lmn->numLayers);
            }
          }

          for (const auto &ss_name : it.second)
          {
            const auto &sides = all_sides[ss_name];

            // Loop on all the sides of this sideset
            for (const auto &sidee : sides)
            {
              stk::mesh::Entity const *node_rels = this->getSTKBulkData(i_block).begin_nodes(sidee);
              const size_t num_nodes = this->getSTKBulkData(i_block).num_nodes(sidee);

              // loop over local nodes of the side (row)
              for (std::size_t i = 0; i < num_nodes; i++)
              {
                stk::mesh::Entity rowNode = node_rels[i];
                row = dofMgr.getGlobalDOF(stk_gid(i_block, rowNode), eq);

                // loop over local nodes of the side (col)
                for (std::size_t j = 0; j < num_nodes; j++)
                {
                  stk::mesh::Entity colNode = node_rels[j];

                  // TODO: this is to accommodate the scenario where the side equation is coupled with
                  //       the volume equations over a whole column of a layered mesh. However, this
                  //       introduces pointless nonzeros if such coupling is not needed.
                  //       The only way to fix this would be to access more information from the problem.
                  //       Until then, couple with *all* equations, over the whole column.
                  if (allowColumnCoupling)
                  {
                    // It's a layered mesh. Assume the worst, and add coupling of the whole column
                    // with all the equations.
                    lmn->getIndices(stk_gid(i_block, colNode), baseId, iLayer);
                    for (int il = 0; il <= lmn->numLayers; ++il)
                    {
                      const GO node3d = lmn->getId(baseId, il);
                      for (unsigned int m = 0; m < neq; ++m)
                      {
                        col = dofMgr.getGlobalDOF(node3d, m);
                        m_jac_factory->insertGlobalIndices(
                            row, Teuchos::arrayView(&col, 1), i_block, j_block);
                        m_jac_factory->insertGlobalIndices(
                            col, Teuchos::arrayView(&row, 1), i_block, j_block);
                      }
                    }
                  }
                  else
                  {
                    // Not a layered mesh, or the eqn is not defined on top/bottom.
                    // Couple locally with volume eqn and the other ss eqn on this sideSet
                    for (auto m : globalEqns)
                    {
                      col = dofMgr.getGlobalDOF(stk_gid(i_block, colNode), m);
                      m_jac_factory->insertGlobalIndices(
                          row, Teuchos::arrayView(&col, 1), i_block, j_block);
                      m_jac_factory->insertGlobalIndices(
                          col, Teuchos::arrayView(&row, 1), i_block, j_block);
                    }
                    for (auto ssEqIt : sideSetEquations)
                    {
                      for (const auto &ssEq_ss_name : ssEqIt.second)
                      {
                        if (ssEq_ss_name == ss_name)
                        {
                          col = dofMgr.getGlobalDOF(stk_gid(i_block, colNode), ssEqIt.first);
                          m_jac_factory->insertGlobalIndices(
                              row, Teuchos::arrayView(&col, 1), i_block, j_block);
                          m_jac_factory->insertGlobalIndices(
                              col, Teuchos::arrayView(&row, 1), i_block, j_block);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      */
    }
  }

  void
  BlockedDiscretization::printConnectivity() const
  {
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      this->printConnectivity(i_block);
  }
  void
  BlockedDiscretization::printConnectivity(const size_t i_block) const
  {
    m_blocks[i_block]->printConnectivity();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getNodeVectorSpace() const
  {
    return this->getNodeVectorSpace(0);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getNodeVectorSpace(const size_t i_block) const
  {
    return m_blocks[i_block]->getNodeVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapNodeVectorSpace() const
  {
    return this->getOverlapNodeVectorSpace(0);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapNodeVectorSpace(const size_t i_block) const
  {
    return m_blocks[i_block]->getOverlapNodeVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getVectorSpace() const
  {
    return this->getVectorSpace(0);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getVectorSpace(const size_t i_block) const
  {
    return m_blocks[i_block]->getVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapVectorSpace() const
  {
    return this->getOverlapVectorSpace();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapVectorSpace(const size_t i_block) const
  {
    return m_blocks[i_block]->getOverlapVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getVectorSpace(const std::string &field_name) const
  {
    return this->getVectorSpace(0, field_name);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getVectorSpace(const size_t i_block, const std::string &field_name) const
  {
    return m_blocks[i_block]->getVectorSpace(field_name);
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getNodeVectorSpace(const std::string &field_name) const
  {
    return this->getNodeVectorSpace(0, field_name);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getNodeVectorSpace(const size_t i_block, const std::string &field_name) const
  {
    return m_blocks[i_block]->getNodeVectorSpace(field_name);
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapVectorSpace(const std::string &field_name) const
  {
    return this->getOverlapVectorSpace(0, field_name);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapVectorSpace(const size_t i_block, const std::string &field_name) const
  {
    return m_blocks[i_block]->getOverlapVectorSpace(field_name);
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapNodeVectorSpace(
      const std::string &field_name) const
  {
    return this->getOverlapNodeVectorSpace(0, field_name);
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedDiscretization::getOverlapNodeVectorSpace(const size_t i_block,
                                                   const std::string &field_name) const
  {
    return m_blocks[i_block]->getOverlapNodeVectorSpace(field_name);
  }

  void
  BlockedDiscretization::printCoords() const
  {
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      this->printCoords(i_block);
  }
  void
  BlockedDiscretization::printCoords(const size_t i_block) const
  {
    m_blocks[i_block]->printCoords();
  }

  // The function transformMesh() maps a unit cube domain by applying a
  // transformation to the mesh.
  void
  BlockedDiscretization::transformMesh()
  {
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      this->transformMesh(i_block);
  }
  void
  BlockedDiscretization::transformMesh(const size_t i_block)
  {
    m_blocks[i_block]->transformMesh();
  }

  void
  BlockedDiscretization::updateMesh()
  {
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      this->updateMesh(i_block);
  }
  void
  BlockedDiscretization::updateMesh(const size_t i_block)
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

  bool BlockedDiscretization::
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

  void BlockedDiscretization::
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

} // namespace Albany
