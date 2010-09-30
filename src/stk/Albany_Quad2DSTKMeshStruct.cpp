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

#include "Albany_Quad2DSTKMeshStruct.hpp"

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


Albany::Quad2DSTKMeshStruct::Quad2DSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_) :
  periodic(params->get("Periodic BC", false))
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  numDim = 2;
  neq = neq_;

  cubatureDegree = params->get("Cubature Degree", 3);

  // Create global mesh: 2D structured, rectangular
  int nelem_x = params->get<int>("1D Elements");
  int nelem_y = params->get<int>("2D Elements");
  double scale_x = params->get("1D Scale",     1.0);
  double scale_y = params->get("2D Scale",     1.0);

  if (comm->MyPID()==0)
    std::cout<<" Creating 2D Quadralateral mesh of size "
              <<nelem_x<<"x"<<nelem_y<<" elements and scaled to "
              <<scale_x<<"x"<<scale_y<<std::endl;

  std::vector<double> x(nelem_x+1);
  double h_x = scale_x/nelem_x;
  for (unsigned int i=0; i<=nelem_x; i++) x[i] = h_x*i;

  std::vector<double> y(nelem_y+1);
  double h_y = scale_y/nelem_y;
  for (unsigned int i=0; i<=nelem_y; i++) y[i] = h_y*i;

  // Distribute rectangle mesh of elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem_x * nelem_y, 0, *comm));
  int numMyElements = elem_map->NumMyElements();


  //Start STK stuff
  metaData = new stk::mesh::MetaData(stk::mesh::fem_entity_rank_names() );
  bulkData = new stk::mesh::BulkData(*metaData , MPI_COMM_WORLD , field_data_chunk_size );
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );

  partVec[0] = &  metaData->declare_part( "Block_1", stk::mesh::Element );

  nsPartVec["NodeSet0"] = & metaData->declare_part( "NodeSet0", stk::mesh::Node );
  nsPartVec["NodeSet1"] = & metaData->declare_part( "NodeSet1", stk::mesh::Node );
  nsPartVec["NodeSet2"] = & metaData->declare_part( "NodeSet2", stk::mesh::Node );
  nsPartVec["NodeSet3"] = & metaData->declare_part( "NodeSet3", stk::mesh::Node );

  stk::mesh::set_cell_topology< shards::Quadrilateral<4> >(*partVec[0]);

  stk::mesh::put_field( *coordinates_field , stk::mesh::Node , metaData->universal_part(), numDim );
  stk::mesh::put_field( *solution_field , stk::mesh::Node , metaData->universal_part(), neq );
  
#ifdef ALBANY_IOSS
  stk::io::put_io_part_attribute(*partVec[0]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet0"]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet1"]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet2"]);
  stk::io::put_io_part_attribute(*nsPartVec["NodeSet3"]);
  stk::io::set_field_role(*coordinates_field, Ioss::Field::ATTRIBUTE);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#endif

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

  // Create elements and node IDs
  const unsigned int nodes_x = periodic ? nelem_x : nelem_x + 1;
  const unsigned int mod_x   = periodic ? nelem_x : std::numeric_limits<unsigned int>::max();
  const unsigned int mod_y   = periodic ? nelem_y : std::numeric_limits<unsigned int>::max();
  for (unsigned int i=0; i<numMyElements; i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int x_GID = elem_GID % nelem_x; // mesh column number
    const unsigned int y_GID = elem_GID / nelem_x; // mesh row number
    const unsigned int lower_left  =  x_GID          + nodes_x* y_GID;
    const unsigned int lower_right = (x_GID+1)%mod_x + nodes_x* y_GID;
    const unsigned int upper_right = (x_GID+1)%mod_x + nodes_x*((y_GID+1)%mod_y);
    const unsigned int upper_left  =  x_GID          + nodes_x*((y_GID+1)%mod_y);

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
    singlePartVec[0] = partVec[0];

    // Add one to IDs because STK requires 1-based
    stk::mesh::Entity& face  = bulkData->declare_entity(stk::mesh::Element, 1+elem_id, singlePartVec);
    stk::mesh::Entity& llnode = bulkData->declare_entity(stk::mesh::Node, 1+lower_left, noPartVec);
    stk::mesh::Entity& lrnode = bulkData->declare_entity(stk::mesh::Node, 1+lower_right, noPartVec);
    stk::mesh::Entity& urnode = bulkData->declare_entity(stk::mesh::Node, 1+upper_right, noPartVec);
    stk::mesh::Entity& ulnode = bulkData->declare_entity(stk::mesh::Node, 1+upper_left, noPartVec);
    bulkData->declare_relation(face, llnode, 0);
    bulkData->declare_relation(face, lrnode, 1);
    bulkData->declare_relation(face, urnode, 2);
    bulkData->declare_relation(face, ulnode, 3);

    double* llnode_coord = stk::mesh::field_data(*coordinates_field, llnode);
    llnode_coord[0] = x[x_GID];   llnode_coord[1] = y[y_GID];
    double* lrnode_coord = stk::mesh::field_data(*coordinates_field, lrnode);
    lrnode_coord[0] = x[x_GID+1]; lrnode_coord[1] = y[y_GID];
    double* urnode_coord = stk::mesh::field_data(*coordinates_field, urnode);
    urnode_coord[0] = x[x_GID+1]; urnode_coord[1] = y[y_GID+1];
    double* ulnode_coord = stk::mesh::field_data(*coordinates_field, ulnode);
    ulnode_coord[0] = x[x_GID]; ulnode_coord[1] = y[y_GID+1];

    if (x_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(ulnode, singlePartVec);
    }
    if ((x_GID+1)==nelem_x) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (y_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
    }
    if ((y_GID+1)==(nelem_y)) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
  }

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;

  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");
}

Albany::Quad2DSTKMeshStruct::~Quad2DSTKMeshStruct()
{
  delete metaData;
  delete bulkData;
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::Quad2DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidSTK2D_DiscParams"));;
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");

  return validPL;
}

