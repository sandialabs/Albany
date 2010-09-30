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

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/TopologyHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>

#ifdef ALBANY_IOSS
#include <stk_io/IossBridge.hpp>
#endif

enum { field_data_chunk_size = 1001 };


Albany::Line1DSTKMeshStruct::Line1DSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_) :
  periodic(params->get("Periodic BC", false))
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  numDim = 1;
  neq = neq_;

  cubatureDegree = params->get("Cubature Degree", 3);

  // Create global mesh
  const int nelem = params->get<int>("1D Elements");
  const double scale = params->get("1D Scale",     1.0);
  std::vector<double> x(nelem+1);
  double h = scale/nelem;
  for (unsigned int i=0; i<=nelem; i++) x[i] = h*i;

  // Distribute the elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem, 0, *comm));
  int numMyElements = elem_map->NumMyElements();


  //Start STK stuff, from UseCase_2 constructor
  metaData = new stk::mesh::MetaData(stk::mesh::fem_entity_rank_names() );
  bulkData = new stk::mesh::BulkData(*metaData , MPI_COMM_WORLD , field_data_chunk_size );
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );

  partVec[0] = &  metaData->declare_part( "Block_1", stk::mesh::Element );

  nsPartVec["NodeSet0"] = & metaData->declare_part( "NodeSet0", stk::mesh::Node );
  nsPartVec["NodeSet1"] = & metaData->declare_part( "NodeSet1", stk::mesh::Node );

  stk::mesh::set_cell_topology< shards::Line<2> >(*partVec[0]);
  stk::mesh::put_field( *coordinates_field , stk::mesh::Node , metaData->universal_part() , numDim );
  stk::mesh::put_field( *solution_field , stk::mesh::Node , metaData->universal_part() , neq );

#ifdef ALBANY_IOSS
  stk::io::put_io_part_attribute(*partVec[0]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet0"]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet1"]);
  stk::io::set_field_role(*coordinates_field, Ioss::Field::ATTRIBUTE);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#endif
  metaData->commit();

  // Finished with metaData, now work on bulk data

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);


  int rightNode=0;
  // Create elements and node IDs
  for (unsigned int i=0; i<numMyElements; i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int left_node  = elem_GID;
    unsigned int right_node = left_node+1;
    if (periodic) right_node %= elem_map->NumGlobalElements();
    if (rightNode < right_node) rightNode = right_node;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
    singlePartVec[0] = partVec[0];

    stk::mesh::Entity& edge  = bulkData->declare_entity(stk::mesh::Element, 1+elem_id, singlePartVec);
    stk::mesh::Entity& lnode = bulkData->declare_entity(stk::mesh::Node, 1+left_node, noPartVec);
    stk::mesh::Entity& rnode = bulkData->declare_entity(stk::mesh::Node, 1+right_node, noPartVec);
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
    if ((elem_GID+1)==elem_map->NumGlobalElements()) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }
  }

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;

  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");
}

Albany::Line1DSTKMeshStruct::~Line1DSTKMeshStruct()
{ 
  delete metaData;
  delete bulkData;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Line1DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidSTK1D_DiscParams"));;
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");

  return validPL;
}

