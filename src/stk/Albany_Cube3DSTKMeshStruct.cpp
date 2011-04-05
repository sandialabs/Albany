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

#include "Albany_Cube3DSTKMeshStruct.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/FEMHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>
#include "Albany_Utils.hpp"

#ifdef ALBANY_IOSS
#include <stk_io/IossBridge.hpp>
#endif

Albany::Cube3DSTKMeshStruct::Cube3DSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const unsigned int neq_, const unsigned int nstates_) :
  GenericSTKMeshStruct(comm),
  periodic(false)
{

  params->validateParameters(*getValidDiscretizationParameters(),0);
  int numDim_ = 3;

  // Create global mesh: 3D structured, rectangular
  int nelem_x = params->get<int>("1D Elements");
  int nelem_y = params->get<int>("2D Elements");
  int nelem_z = params->get<int>("3D Elements");
  double scale_x = params->get("1D Scale",     1.0);
  double scale_y = params->get("2D Scale",     1.0);
  double scale_z = params->get("3D Scale",     1.0);

  if (comm->MyPID()==0)
    std::cout <<" Creating 3D cube mesh of size "
              <<nelem_x<<"x"<<nelem_y<<"x"<<nelem_z<<" elements and scaled to "
              <<scale_x<<"x"<<scale_y<<"x"<<scale_z<<std::endl;

  std::vector<double> x(nelem_x+1);
  double h_x = scale_x/nelem_x;
  for (int i=0; i<=nelem_x; i++) x[i] = h_x*i;

  std::vector<double> y(nelem_y+1);
  double h_y = scale_y/nelem_y;
  for (int i=0; i<=nelem_y; i++) y[i] = h_y*i;

  std::vector<double> z(nelem_z+1);
  double h_z = scale_z/nelem_z;
  for (int i=0; i<=nelem_z; i++) z[i] = h_z*i;

  // Distribute rectangle mesh of elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem_x * nelem_y * nelem_z, 0, *comm));
  int numMyElements = elem_map->NumMyElements();

  std::vector<std::string> nsNames;
  nsNames.push_back("NodeSet0");
  nsNames.push_back("NodeSet1");
  nsNames.push_back("NodeSet2");
  nsNames.push_back("NodeSet3");
  nsNames.push_back("NodeSet4");
  nsNames.push_back("NodeSet5");

  this->SetupMetaData(params, neq_, nstates_, numDim_);
  this->DeclareParts(nsNames);

  stk::mesh::fem::set_cell_topology_new< shards::Hexahedron<8> >(*partVec[0]);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

  const unsigned int nodes_x = nelem_x + 1;
  const unsigned int nodes_xy = nodes_x*(nelem_y + 1);
  // Create elements and node IDs
  for (int i=0; i<numMyElements; i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int z_GID = elem_GID / (nelem_x*nelem_y); // mesh column number
    const unsigned int xy_plane = elem_GID % (nelem_x*nelem_y); 
    const unsigned int x_GID = xy_plane % nelem_x; // mesh column number
    const unsigned int y_GID = xy_plane / nelem_x; // mesh row number
    const unsigned int lower_left  =  x_GID    + nodes_x* y_GID    + nodes_xy*z_GID;
    const unsigned int lower_right = (x_GID+1) + nodes_x* y_GID    + nodes_xy*z_GID;
    const unsigned int upper_right = (x_GID+1) + nodes_x*(y_GID+1) + nodes_xy*z_GID;
    const unsigned int upper_left  =  x_GID    + nodes_x*(y_GID+1) + nodes_xy*z_GID;
    const unsigned int lower_left_back  =  x_GID    + nodes_x* y_GID    + nodes_xy*(z_GID+1);
    const unsigned int lower_right_back = (x_GID+1) + nodes_x* y_GID    + nodes_xy*(z_GID+1);
    const unsigned int upper_right_back = (x_GID+1) + nodes_x*(y_GID+1) + nodes_xy*(z_GID+1);
    const unsigned int upper_left_back  =  x_GID    + nodes_x*(y_GID+1) + nodes_xy*(z_GID+1);

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
    singlePartVec[0] = partVec[0];

    // Add one to IDs because STK requires 1-based
    stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
    stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_left, noPartVec);
    stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_right, noPartVec);
    stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_right, noPartVec);
    stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_left, noPartVec);
    stk::mesh::Entity& llnodeb = bulkData->declare_entity(metaData->node_rank(), 1+lower_left_back, noPartVec);
    stk::mesh::Entity& lrnodeb = bulkData->declare_entity(metaData->node_rank(), 1+lower_right_back, noPartVec);
    stk::mesh::Entity& urnodeb = bulkData->declare_entity(metaData->node_rank(), 1+upper_right_back, noPartVec);
    stk::mesh::Entity& ulnodeb = bulkData->declare_entity(metaData->node_rank(), 1+upper_left_back, noPartVec);
    bulkData->declare_relation(elem, llnode, 0);
    bulkData->declare_relation(elem, lrnode, 1);
    bulkData->declare_relation(elem, urnode, 2);
    bulkData->declare_relation(elem, ulnode, 3);
    bulkData->declare_relation(elem, llnodeb, 4);
    bulkData->declare_relation(elem, lrnodeb, 5);
    bulkData->declare_relation(elem, urnodeb, 6);
    bulkData->declare_relation(elem, ulnodeb, 7);

    double* coord;
    coord = stk::mesh::field_data(*coordinates_field, llnode);
    coord[0] = x[x_GID];   coord[1] = y[y_GID];   coord[2] = z[z_GID];
    coord = stk::mesh::field_data(*coordinates_field, lrnode);
    coord[0] = x[x_GID+1]; coord[1] = y[y_GID];   coord[2] = z[z_GID];
    coord = stk::mesh::field_data(*coordinates_field, urnode);
    coord[0] = x[x_GID+1]; coord[1] = y[y_GID+1]; coord[2] = z[z_GID];
    coord = stk::mesh::field_data(*coordinates_field, ulnode);
    coord[0] = x[x_GID];   coord[1] = y[y_GID+1]; coord[2] = z[z_GID];

    coord = stk::mesh::field_data(*coordinates_field, llnodeb);
    coord[0] = x[x_GID];   coord[1] = y[y_GID];   coord[2] = z[z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, lrnodeb);
    coord[0] = x[x_GID+1]; coord[1] = y[y_GID];   coord[2] = z[z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, urnodeb);
    coord[0] = x[x_GID+1]; coord[1] = y[y_GID+1]; coord[2] = z[z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, ulnodeb);
    coord[0] = x[x_GID];   coord[1] = y[y_GID+1]; coord[2] = z[z_GID+1];

    
    if (x_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(llnodeb, singlePartVec);
       bulkData->change_entity_parts(ulnodeb, singlePartVec);
    }
    if ((x_GID+1)==(unsigned int)nelem_x) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
       bulkData->change_entity_parts(lrnodeb, singlePartVec);
       bulkData->change_entity_parts(urnodeb, singlePartVec);
    }
    if (y_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(llnodeb, singlePartVec);
       bulkData->change_entity_parts(lrnodeb, singlePartVec);
    }
    if ((y_GID+1)==(unsigned int)nelem_y) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
       bulkData->change_entity_parts(ulnodeb, singlePartVec);
       bulkData->change_entity_parts(urnodeb, singlePartVec);
    }
    if (z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet4"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if ((z_GID+1)==(unsigned int)nelem_z) {
       singlePartVec[0] = nsPartVec["NodeSet5"];
       bulkData->change_entity_parts(llnodeb, singlePartVec);
       bulkData->change_entity_parts(lrnodeb, singlePartVec);
       bulkData->change_entity_parts(ulnodeb, singlePartVec);
       bulkData->change_entity_parts(urnodeb, singlePartVec);
    }
  }

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Cube3DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK3D_DiscParams");
// AGS: 5/10: Periodic not implemented for 3D
//  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Deptt of Y discretization");
  validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");

  return validPL;
}
