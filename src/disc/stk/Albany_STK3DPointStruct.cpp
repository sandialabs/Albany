//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_STK3DPointStruct.hpp"
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

//Constructor for meshes read from ASCII file
Albany::STK3DPointStruct::STK3DPointStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                           const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct(params,Teuchos::null,3)
{
  partVec[0] = &metaData->declare_part("Block0", stk::topology::ELEMENT_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif
  std::cout << "---3DPoint constructor---" << std::endl;
  stk::mesh::set_cell_topology<shards::Particle>(*partVec[0]);
  std::cout << "finished setting cell topology to shards::Particle" << std::endl;
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
  std::cout << "finished extracting cell topology data" << std::endl;
  int cubDegree = 1;
  std::vector<std::string> nsNames;
  std::vector<std::string> ssNames;
  int worksetSize = 1;

  std::cout << "--- creating a new MeshSpecsStruct ---" << std::endl;
  this->meshSpecs[0] =
    Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cubDegree,
                                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                                             ebNameToIndex, this->interleavedOrdering));
  std::cout << "---3DPoint constructor done---" << std::endl;
}

Albany::STK3DPointStruct::~STK3DPointStruct() {};

void
Albany::STK3DPointStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{
  std::cout << "---3DPoint::setFieldAndBulkData---" << std::endl;
  SetupFieldData(commT, neq_, req, sis, worksetSize);
  metaData->commit();
  bulkData->modification_begin(); // Begin modifying the mesh
  //TmplSTKMeshStruct<0, albany_stk_mesh_traits<0> >::buildMesh(commT);
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  singlePartVec[0] = partVec[0]; // Get the element block part to put the element in.
  // Declare element 1 is in that block
  stk::mesh::Entity pt  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1, singlePartVec);
  // Declare node 1 is in the node part vector
  stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, 1, nodePartVec);
  // Declare that the node belongs to the element "pt"
  // "node" is the zeroth node of this element
  bulkData->declare_relation(pt, node, 0);

  bulkData->modification_end();
}

void
Albany::STK3DPointStruct::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  std::cout << "---3DPoint::buildMesh---" << std::endl;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::STK3DPointStruct::getValidDiscretizationParameters() const
{
  std::cout << "---3DPoint::getValidDiscretizationParameters---" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK3DPoint_DiscParams");

  return validPL;
}
