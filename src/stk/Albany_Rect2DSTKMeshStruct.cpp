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

#include "Albany_Rect2DSTKMeshStruct.hpp"

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
#include "Albany_Utils.hpp"

Albany::Rect2DSTKMeshStruct::Rect2DSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_) :
  GenericSTKMeshStruct(comm),
  periodic(params->get("Periodic BC", false)),
  triangles(false)
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  int numDim_ = 2;

  string cellTopo = params->get("Cell Topology", "Quad");
  if (cellTopo == "Tri" || cellTopo == "Triangle")  triangles = true;
  else TEST_FOR_EXCEPTION (cellTopo != "Quad", std::logic_error,
     "\nUnknown Cell Topology entry in STK2D(not \'Tri\' or \'Quad\'): "
      << cellTopo);

  // Create global mesh: 2D structured, rectangular
  int nelem_x = params->get<int>("1D Elements");
  int nelem_y = params->get<int>("2D Elements");
  double scale_x = params->get("1D Scale",     1.0);
  double scale_y = params->get("2D Scale",     1.0);

  if (comm->MyPID()==0) {
    std::cout<<" Creating 2D Rectanglular mesh of size "
              <<nelem_x<<"x"<<nelem_y<<" quad elements and scaled to "
              <<scale_x<<"x"<<scale_y<<std::endl;
    if (triangles)
      std::cout<<" Quad elements cut to make twice as many triangles " <<std::endl;
  }


  std::vector<double> x(nelem_x+1);
  double h_x = scale_x/nelem_x;
  for (int i=0; i<=nelem_x; i++) x[i] = h_x*i;

  std::vector<double> y(nelem_y+1);
  double h_y = scale_y/nelem_y;
  for (int i=0; i<=nelem_y; i++) y[i] = h_y*i;

  // Distribute rectangle mesh of quad elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem_x * nelem_y, 0, *comm));
  int numMyElements = elem_map->NumMyElements();

  std::vector<std::string> nsNames;
  nsNames.push_back("NodeSet0"); 
  nsNames.push_back("NodeSet1"); 
  nsNames.push_back("NodeSet2"); 
  nsNames.push_back("NodeSet3"); 

  this->SetupMetaData(params, neq_, nstates_, numDim_);
  this->DeclareParts(nsNames);

  if (triangles)
    stk::mesh::set_cell_topology< shards::Triangle<3> >(*partVec[0]);
  else 
    stk::mesh::set_cell_topology< shards::Quadrilateral<4> >(*partVec[0]);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

  // Create elements and node IDs
  const unsigned int nodes_x = periodic ? nelem_x : nelem_x + 1;
  const unsigned int mod_x   = periodic ? nelem_x : std::numeric_limits<unsigned int>::max();
  const unsigned int mod_y   = periodic ? nelem_y : std::numeric_limits<unsigned int>::max();
  for (int i=0; i<numMyElements; i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int x_GID = elem_GID % nelem_x; // mesh column number
    const unsigned int y_GID = elem_GID / nelem_x; // mesh row number
    const unsigned int lower_left  =  x_GID          + nodes_x* y_GID;
    const unsigned int lower_right = (x_GID+1)%mod_x + nodes_x* y_GID;
    const unsigned int upper_right = (x_GID+1)%mod_x + nodes_x*((y_GID+1)%mod_y);
    const unsigned int upper_left  =  x_GID          + nodes_x*((y_GID+1)%mod_y);

    // get ID of quadrilateral -- will be doubled for trianlges below
    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
    singlePartVec[0] = partVec[0];

    // Declare NodesL= (Add one to IDs because STK requires 1-based
    stk::mesh::Entity& llnode = bulkData->declare_entity(stk::mesh::Node, 1+lower_left, noPartVec);
    stk::mesh::Entity& lrnode = bulkData->declare_entity(stk::mesh::Node, 1+lower_right, noPartVec);
    stk::mesh::Entity& urnode = bulkData->declare_entity(stk::mesh::Node, 1+upper_right, noPartVec);
    stk::mesh::Entity& ulnode = bulkData->declare_entity(stk::mesh::Node, 1+upper_left, noPartVec);

    if (triangles) { // pair of 3-node triangles
      stk::mesh::Entity& face  = bulkData->declare_entity(stk::mesh::Element, 1+2*elem_id, singlePartVec);
      bulkData->declare_relation(face, llnode, 0);
      bulkData->declare_relation(face, lrnode, 1);
      bulkData->declare_relation(face, urnode, 2);
      stk::mesh::Entity& face2 = bulkData->declare_entity(stk::mesh::Element, 1+2*elem_id+1, singlePartVec);
      bulkData->declare_relation(face2, llnode, 0);
      bulkData->declare_relation(face2, urnode, 1);
      bulkData->declare_relation(face2, ulnode, 2);
    }
    else {  //4-node quad
      stk::mesh::Entity& face  = bulkData->declare_entity(stk::mesh::Element, 1+elem_id, singlePartVec);
      bulkData->declare_relation(face, llnode, 0);
      bulkData->declare_relation(face, lrnode, 1);
      bulkData->declare_relation(face, urnode, 2);
      bulkData->declare_relation(face, ulnode, 3);
    }

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
    if ((x_GID+1)==(unsigned int)nelem_x) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (y_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
    }
    if ((y_GID+1)==(unsigned int)nelem_y) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
  }

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Rect2DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = 
    this->getValidGenericSTKParameters("ValidSTK2D_DiscParams");

  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");

  return validPL;
}
