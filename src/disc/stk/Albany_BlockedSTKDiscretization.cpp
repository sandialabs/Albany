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

#include "Albany_BlockedSTKDiscretization.hpp"
#include "STKConnManager.hpp"

#include "Thyra_DefaultProductVectorSpace.hpp"

namespace Albany
{

  //Assume we only have one block right now
  BlockedSTKDiscretization::BlockedSTKDiscretization(
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

  BlockedSTKDiscretization::BlockedSTKDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList> &discParams_,
      Teuchos::Array<Teuchos::RCP<AbstractSTKMeshStruct>> &stkMeshStruct_,
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

    n_m_blocks = bDiscParams->get<int>("Num Blocks");

    m_blocks.resize(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      m_blocks[i_block] = Teuchos::rcp(new disc_type(discParams, stkMeshStruct_[i_block], comm_,
                                                     rigidBodyModes_, sideSetEquations_));

    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager>
        connMngr = Teuchos::rcp(new Albany::STKConnManager(stkMeshStruct_[0]));

    // build the DOF manager for the problem
    if (const Teuchos::MpiComm<int> *mpiComm = dynamic_cast<const Teuchos::MpiComm<int> *>(comm.get()))
    {
      MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

      blockedDOFManager = Teuchos::rcp(new panzer::BlockedDOFManager(connMngr, rawComm));

      // by default assume orientations are not required
      bool orientationsRequired = false;

      // set orientations required flag
      blockedDOFManager->setOrientationsRequired(orientationsRequired);

      // blocked degree of freedom manager
      std::string fieldOrder = discParams->sublist("Solution").get<std::string>("blocks names");
      std::vector<std::vector<std::string>> blocks;
      buildNewBlocking(fieldOrder, blocks);

      blockedDOFManager->setFieldOrder(blocks);

      //blockedDOFManager->buildGlobalUnknowns();
      //blockedDOFManager->printFieldInformation(*out);
    }
  }

  void
  BlockedSTKDiscretization::computeProductVectorSpaces()
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
  BlockedSTKDiscretization::computeGraphs()
  {
    m_jac_factory =
        Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_pvs,
                                                      m_pvs));
    m_overlap_jac_factory =
        Teuchos::rcp(new ThyraBlockedCrsMatrixFactory(m_overlap_pvs,
                                                      m_overlap_pvs));

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      // For the diagonal block we reuse the graph previously computed:
      m_jac_factory->setBlockFactory(i_block, i_block, m_blocks[i_block]->m_jac_factory);

      // Then, we loop over the off diagonal blocks:
      for (size_t j_block = 0; j_block < i_block; ++j_block)
        this->computeGraphs(i_block, j_block);
    }

    m_jac_factory->fillComplete();
    m_overlap_jac_factory->fillComplete();
  }

  void
  BlockedSTKDiscretization::computeGraphs(const size_t i_block, const size_t j_block)
  {
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
    return m_blocks[i_block]->getVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedSTKDiscretization::getOverlapVectorSpace() const
  {
    return this->getOverlapVectorSpace();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  BlockedSTKDiscretization::getOverlapVectorSpace(const size_t i_block) const
  {
    return m_blocks[i_block]->getOverlapVectorSpace();
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
    if (Teuchos::nonnull(blockedDOFManager))
    {
      ; //blockedDOFManager->buildGlobalUnknowns();
      ; //blockedDOFManager->printFieldInformation(*out);
    }

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      this->updateMesh(i_block);

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

  void BlockedSTKDiscretization::
      buildNewBlocking(const std::string &fieldOrder, std::vector<std::vector<std::string>> &blocks)
  {
    Teuchos::RCP<std::vector<std::string>> current;

    std::string::size_type nextOpenBracket, lastCloseBracket, nextCloseBracket, nextComa, pos, lastPos;

    lastPos = fieldOrder.find_first_not_of("[ ");
    lastCloseBracket = fieldOrder.find_last_of("]");

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
        pos = fieldOrder.find_first_of(",", lastPos);

        current->push_back(formatFieldName(fieldOrder.substr(lastPos, pos - lastPos)));
        lastPos = fieldOrder.find_first_not_of(", ", pos);
      }
      blocks.push_back(*current);
      nextCloseBracket = fieldOrder.find_first_of("]", lastPos);
    } while (nextCloseBracket < lastCloseBracket);
  }
} // namespace Albany
