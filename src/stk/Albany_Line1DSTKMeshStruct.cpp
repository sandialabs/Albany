/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <iostream>

#include "Albany_Line1DSTKMeshStruct.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif
#include "Albany_Utils.hpp"

// Needed for rebalance 
//#include <stk_mesh/fem/DefaultFEM.hpp>
//#include <stk_rebalance/Rebalance.hpp>
//#include <stk_rebalance/Partition.hpp>
//#include <stk_rebalance/ZoltanPartition.hpp>


Albany::Line1DSTKMeshStruct::Line1DSTKMeshStruct(
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params,1),
  periodic(params->get("Periodic BC", false))
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  std::vector<std::string> nsNames;
  nsNames.push_back("NodeSet0");
  nsNames.push_back("NodeSet1");
  this->DeclareParts(nsNames);
  stk::mesh::fem::set_cell_topology< shards::Line<2> >(*partVec[0]);

  // Construct MeshSpecsStruct
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  int numEB = 1; // Hardcode a single element block for now

  // Create just enough of the mesh to figure out number of owned elements 
  // so that the problem setup can know the worksetSize
  nelem = params->get<int>("1D Elements");
  elem_map = Teuchos::rcp(new Epetra_Map(nelem, 0, *comm));

  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

  // MeshSpecs holds all info needed to set up an Albany problem
  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, worksetSize, numEB, this->interleavedOrdering));
}

void
Albany::Line1DSTKMeshStruct::setFieldAndBulkData(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize) 
{
  this->SetupFieldData(comm, neq_, sis, worksetSize);
  metaData->commit();

  // Create global mesh
  const double scale = params->get("1D Scale",     1.0);
  std::vector<double> x(nelem+1);
  double h = scale/nelem;
  for (int i=0; i<=nelem; i++) x[i] = h*i;

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

  unsigned int rightNode=0;
  // Create elements and node IDs
  for (int i=0; i<elem_map->NumMyElements(); i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int left_node  = elem_GID;
    unsigned int right_node = left_node+1;
    if (periodic) right_node %= elem_map->NumGlobalElements();
    if (rightNode < right_node) rightNode = right_node;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
    singlePartVec[0] = partVec[0];

    stk::mesh::Entity& edge  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
    stk::mesh::Entity& lnode = bulkData->declare_entity(metaData->node_rank(), 1+left_node, noPartVec);
    stk::mesh::Entity& rnode = bulkData->declare_entity(metaData->node_rank(), 1+right_node, noPartVec);
    bulkData->declare_relation(edge, lnode, 0);
    bulkData->declare_relation(edge, rnode, 1);

    double* lnode_coord = stk::mesh::field_data(*coordinates_field, lnode);
    lnode_coord[0] = x[elem_GID];
    double* rnode_coord = stk::mesh::field_data(*coordinates_field, rnode);
    rnode_coord[0] = x[elem_GID+1];

    // Set node sets
    if (elem_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(lnode, singlePartVec);
    }
    if ((elem_GID+1)==(unsigned int)elem_map->NumGlobalElements()) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }
  }

  bulkData->modification_end();
  useElementAsTopRank = true;


// Needed for rebalance 
//  stk::mesh::Selector owned_selector = metaData->locally_owned_part();
//  cout << "Before rebal " << comm->MyPID() << "  " << stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
//  // Use Zoltan to determine new partition
//  Teuchos::ParameterList emptyList;
//  stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, emptyList);
//  stk::mesh::Selector selector(metaData->universal_part());
//  stk::rebalance::rebalance(*bulkData, selector, coordinates_field, NULL, zoltan_partition);
//
//  cout << "After rebal " << comm->MyPID() << "  " << stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Line1DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK1D_DiscParams");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");

  return validPL;
}

