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

namespace Albany {

//Assume we only have one block right now
BlockedDiscretization::BlockedDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    Teuchos::RCP<AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_){

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcp_dynamic_cast;
    typedef double Scalar;

    m_blocks.resize(1);

	m_blocks[0] = Teuchos::rcp(new disc_type(discParams_, stkMeshStruct_, comm_,
		rigidBodyModes_, sideSetEquations_));

    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager>
      connMngr = Teuchos::rcp(new Albany::STKConnManager(stkMeshStruct_));

   // build the DOF manager for the problem
   if (const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get())) {
     MPI_Comm rawComm = (*mpiComm->getRawMpiComm().get())();

   Teuchos::RCP<panzer::BlockedDOFManager> dofManager
         = Teuchos::rcp(new panzer::BlockedDOFManager(connMngr, rawComm));
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
   std::vector<std::vector<std::string> > blocks;
   buildBlocking(fieldOrder,blocks);
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

void
BlockedDiscretization::printConnectivity() const
{
	m_blocks[0]->printConnectivity();
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getNodeVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getNodeVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getOverlapVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getOverlapVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return m_blocks[0]->getOverlapNodeVectorSpace(field_name);
}

void
BlockedDiscretization::printCoords() const
{
	m_blocks[0]->printCoords();
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
BlockedDiscretization::transformMesh()
{
	m_blocks[0]->transformMesh();
}

void
BlockedDiscretization::updateMesh()
{
	m_blocks[0]->updateMesh();
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

void addFieldsToMesh(STKDiscretization& mesh,
                               const Teuchos::ParameterList& output_list)
{
  // register cell averaged scalar fields
  const Teuchos::ParameterList & cellAvgQuants = output_list.sublist("Cell Average Quantities");
  for(Teuchos::ParameterList::ConstIterator itr=cellAvgQuants.begin();
      itr!=cellAvgQuants.end();++itr) {
    const std::string & blockId = itr->first;
    const std::string & fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens,fields,",",true);

    for(std::size_t i=0;i<tokens.size();i++)
      mesh.addCellField(tokens[i],blockId);
  }

  // register cell averaged components of vector fields
  // just allocate space for the fields here. The actual calculation and writing
  // are done by panzer_stk::ScatterCellAvgVector.
  const Teuchos::ParameterList & cellAvgVectors = output_list.sublist("Cell Average Vectors");
  for(Teuchos::ParameterList::ConstIterator itr = cellAvgVectors.begin();
      itr != cellAvgVectors.end(); ++itr) {
    const std::string & blockId = itr->first;
    const std::string & fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens,fields,",",true);

    for(std::size_t i = 0; i < tokens.size(); i++) {
      std::string d_mod[3] = {"X","Y","Z"};
      for(std::size_t d = 0; d < mesh.getDimension(); d++)
        mesh.addCellField(tokens[i]+d_mod[d],blockId);
    }
  }

  // register cell quantities
  const Teuchos::ParameterList & cellQuants = output_list.sublist("Cell Quantities");
  for(Teuchos::ParameterList::ConstIterator itr=cellQuants.begin();
      itr!=cellQuants.end();++itr) {
    const std::string & blockId = itr->first;
    const std::string & fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens,fields,",",true);

    for(std::size_t i=0;i<tokens.size();i++)
      mesh.addCellField(tokens[i],blockId);
  }

  // register ndoal quantities
  const Teuchos::ParameterList & nodalQuants = output_list.sublist("Nodal Quantities");
  for(Teuchos::ParameterList::ConstIterator itr=nodalQuants.begin();
      itr!=nodalQuants.end();++itr) {
    const std::string & blockId = itr->first;
    const std::string & fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens,fields,",",true);

    for(std::size_t i=0;i<tokens.size();i++)
      mesh.addSolutionField(tokens[i],blockId);
  }

  const Teuchos::ParameterList & allocNodalQuants = output_list.sublist("Allocate Nodal Quantities");
  for(Teuchos::ParameterList::ConstIterator itr=allocNodalQuants.begin();
      itr!=allocNodalQuants.end();++itr) {
    const std::string & blockId = itr->first;
    const std::string & fields = Teuchos::any_cast<std::string>(itr->second.getAny());
    std::vector<std::string> tokens;

    // break up comma seperated fields
    panzer::StringTokenizer(tokens,fields,",",true);

    for(std::size_t i=0;i<tokens.size();i++)
      mesh.addSolutionField(tokens[i],blockId);
  }

}

bool BlockedDiscretization::
requiresBlocking(const std::string & fieldOrder)
{
   std::vector<std::string> tokens;

   // break it up on spaces
   panzer::StringTokenizer(tokens,fieldOrder," ",true);

   if(tokens.size()<2) // there has to be at least 2 tokens to block
      return false;

   // check the prefix - must signal "blocked"
   if(tokens[0]!="blocked:")
      return false;

   // loop over tokens
   bool acceptsHyphen = false;
   for(std::size_t i=1;i<tokens.size();i++) {

      // acceptsHyphen can't be false, and then a hyphen accepted
      TEUCHOS_TEST_FOR_EXCEPTION(tokens[i]=="-" && !acceptsHyphen,std::logic_error,

                                 "Blocked assembly: Error \"Field Order\" hyphen error at "
                                 "token " << i);

      if(acceptsHyphen && tokens[i]=="-")
         acceptsHyphen = false;
      else { // token must be a field
         acceptsHyphen = true;
      }
   }

   return true;
}

void BlockedDiscretization::
buildBlocking(const std::string & fieldOrder, std::vector<std::vector<std::string> > & blocks)
{
   // now we don't have to check
   TEUCHOS_ASSERT(requiresBlocking(fieldOrder));

   std::vector<std::string> tokens;

   // break it up on spaces
   panzer::StringTokenizer(tokens,fieldOrder," ",true);

   Teuchos::RCP<std::vector<std::string> > current;
   for(std::size_t i=1;i<tokens.size();i++) {

      if(tokens[i]!="-" && tokens[i-1]!="-") {
         // if there is something to add, add it to the blocks
         if(current!=Teuchos::null)
            blocks.push_back(*current);

         current = Teuchos::rcp(new std::vector<std::string>);
      }

      if(tokens[i]!="-")
         current->push_back(tokens[i]);
   }

   if(current!=Teuchos::null)
      blocks.push_back(*current);
}


} // namespace Albany
