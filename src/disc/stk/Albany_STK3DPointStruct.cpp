//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STK3DPointStruct.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

//Constructor for meshes read from ASCII file
STK3DPointStruct::
STK3DPointStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<const Teuchos_Comm>& comm,
					       const int numParams)
 : GenericSTKMeshStruct(params, 3, numParams)
{
  std::cout << "---3DPoint constructor---" << std::endl;
  partVec.push_back(&metaData->declare_part_with_topology("Block0", stk::topology::PARTICLE));
  std::cout << "finished setting cell topology to shards::Particle" << std::endl;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk::topology::PARTICLE);
  this->addElementBlockInfo(0, "Block0", partVec[0], shards_ctd);

  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData(); 
  std::cout << "finished extracting cell topology data" << std::endl;
  std::vector<std::string> nsNames;
  std::vector<std::string> ssNames;
  int worksetSize = 1;

  std::cout << "--- creating a new MeshSpecsStruct ---" << std::endl;
  this->meshSpecs[0] =
    Teuchos::rcp(new MeshSpecsStruct(ctd, numDim,
                                             nsNames, ssNames, worksetSize, "Block0",
                                             ebNameToIndex));
  std::cout << "---3DPoint constructor done---" << std::endl;

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(comm);

  // Initialize the requested sideset mesh struct in the mesh
  this->initializeSideSetMeshStructs(comm);
}

void STK3DPointStruct::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis,
              const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis)
{
  std::cout << "---3DPoint::setFieldData---" << std::endl;
  SetupFieldData(comm, sis);
  this->setSideSetFieldData(comm, side_set_sis);
}

void STK3DPointStruct::
setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  std::cout << "---3DPoint::setBulkData---" << std::endl;
  metaData->commit();
  bulkData->modification_begin(); // Begin modifying the mesh
  //TmplSTKMeshStruct<0, albany_stk_mesh_traits<0> >::buildMesh(comm);
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  singlePartVec[0] = partVec[0]; // Get the element block part to put the element in.
  // Declare element 1 is in that block
  stk::mesh::Entity pt  = bulkData->declare_element(1, singlePartVec);
  // Declare node 1 is in the node part vector
  stk::mesh::Entity node = bulkData->declare_node(1, nodePartVec);
  // Declare that the node belongs to the element "pt"
  // "node" is the zeroth node of this element
  bulkData->declare_relation(pt, node, 0);

  bulkData->modification_end();

  fieldAndBulkDataSet = true;
  this->setSideSetBulkData(comm);
}

void
STK3DPointStruct::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& /* comm */)
{
  std::cout << "---3DPoint::buildMesh---" << std::endl;
}

Teuchos::RCP<const Teuchos::ParameterList>
STK3DPointStruct::getValidDiscretizationParameters() const
{
  std::cout << "---3DPoint::getValidDiscretizationParameters---" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK3DPoint_DiscParams");

  return validPL;
}

} // namespace Albany
