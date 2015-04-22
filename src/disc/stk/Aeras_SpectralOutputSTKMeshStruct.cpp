//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>

#include "Aeras_SpectralOutputSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <Albany_STKNodeSharing.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#ifdef ALBANY_64BIT_INT
// long int == 64bit
#  define ST_LLI "%li"
#else
#  define ST_LLI "%i"
#endif


//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"


//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN


//Constructor 
Aeras::SpectralOutputSTKMeshStruct::SpectralOutputSTKMeshStruct(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Teuchos_Comm>& commT, 
                                             const int numDim_, const int worksetSize_,
                                             const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& wsElNodeID_,
                                             const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& coords_,
                                             const Teuchos::RCP<const Tpetra_Map>& node_mapT_, 
                                             const int points_per_edge_):
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false), 
  numDim(numDim_),
  points_per_edge(points_per_edge_)
{
#ifdef OUTPUT_TO_SCREEN
  *out << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

  //FIXME (Bill): set member functions wsElNodeID, coords, node_mapT.
  contigIDs = params->get("Contiguous IDs", true);
  
  params->validateParameters(*getValidDiscretizationParameters(),0);

  //just creating 1 element block.  May want to change later...
  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, stk::topology::ELEMENT_RANK );
  ebNameToIndex[ebn] = 0;
  std::vector<std::string> nsNames;
  std::vector<std::string> ssNames;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  //FIXME (Bill): the geometry is hard-coded to hex 8s - need to change to quad 3s
  stk::mesh::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);

  int cub = params->get("Cubature Degree",3);
  //FIXME: hard-coded for now that all the elements are in 1 workset
  int worksetSize = -1; 
  //int worksetSizeMax = params->get("Workset Size",50);
  //int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_mapT->getNodeNumElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

#ifdef OUTPUT_TO_SCREEN
  *out << "numDim, cub, worksetSize, points_per_edge, ctd name: " << numDim << ", " 
       << cub << ", " << worksetSize << ", " << points_per_edge << ", " << ctd.name << "\n";
#endif

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));


}

Aeras::SpectralOutputSTKMeshStruct::~SpectralOutputSTKMeshStruct()
{
}

void
Aeras::SpectralOutputSTKMeshStruct::setFieldAndBulkData(
                                               const Teuchos::RCP<const Teuchos_Comm>& commT,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const unsigned int worksetSize)

{
#ifdef OUTPUT_TO_SCREEN
  *out << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  
  //FIXME?: assuming for now 1 element block
  unsigned int ebNo = 0; 

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef Albany::AbstractSTKFieldContainer::QPScalarFieldType ElemScalarFieldType;

  Albany::AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();


  //FIXME (Bill): compute how many elements will be in the new bilinear mesh for output.  
  //int numElementsNew = # of elements in new bilinear mesh for output on each processor
  //FIXME (Bill): create a map of the new bilinear elements.  Call it elem_map_new:
  //Teuchos::RCP<Tpetra_Map> elem_map_new = Teuchos::rcp(new Tpetra_Map(...

  //NOTES FOR BILL:  The below needs to be rewritten.  It is cut/paste from Albany_ASCIISTKMeshStruct.cpp.  
  //Think of eles below as similar to wsElNodeID (connectivity array).  It assumed hex meshes.  
  //Hence, it is of size numElemenetsNew x 8.
  //xyz as an array of the coordinates of each node.  It is of size (# nodes on this processor) x 3 (for a 3D mesh).
 
/*  for (int i=0; i<numElementsNew; i++) {
     const unsigned int elem_GID = elem_map_new->getGlobalElement(i);
     //std::cout << "elem_GID: " << elem_GID << std::endl; 
     stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
     singlePartVec[0] = partVec[ebNo];
     //FIXME (Bill): populate the following from wsElNodeID for the element connectivity of your new bilinear mesh.
     //In the following line, you need to add 1 if your global IDs are 0-based; otherwise you do not. 
     stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
     stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][0], nodePartVec);
     stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][1], nodePartVec);
     stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][2], nodePartVec);
     stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][3], nodePartVec);
     stk::mesh::Entity llnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][4], nodePartVec);
     stk::mesh::Entity lrnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][5], nodePartVec);
     stk::mesh::Entity urnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][6], nodePartVec);
     stk::mesh::Entity ulnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][7], nodePartVec);
     bulkData->declare_relation(elem, llnode, 0);
     bulkData->declare_relation(elem, lrnode, 1);
     bulkData->declare_relation(elem, urnode, 2);
     bulkData->declare_relation(elem, ulnode, 3);
     bulkData->declare_relation(elem, llnodeb, 4);
     bulkData->declare_relation(elem, lrnodeb, 5);
     bulkData->declare_relation(elem, urnodeb, 6);
     bulkData->declare_relation(elem, ulnodeb, 7);

     //FIXME (Bill): populate the following using coords array and node_mapT 
     double* coord;
     int node_GID;
     unsigned int node_LID;

     node_GID = eles[i][0]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord = stk::mesh::field_data(*coordinates_field, llnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][1]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord = stk::mesh::field_data(*coordinates_field, lrnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][2]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord = stk::mesh::field_data(*coordinates_field, urnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][3]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord = stk::mesh::field_data(*coordinates_field, ulnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk::mesh::field_data(*coordinates_field, llnodeb);
     node_GID = eles[i][4]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][5]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord = stk::mesh::field_data(*coordinates_field, lrnodeb);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk::mesh::field_data(*coordinates_field, urnodeb);
     node_GID = eles[i][6]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk::mesh::field_data(*coordinates_field, ulnodeb);
     node_GID = eles[i][7]-1;
     node_LID = node_mapT->getLocalElement(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     }*/

  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}

Teuchos::RCP<const Teuchos::ParameterList>
Aeras::SpectralOutputSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  Teuchos::Array<std::string> emptyStringArray;
  validPL->set<Teuchos::Array<std::string> >("Additional Node Sets", emptyStringArray, "Declare additional node sets not present in the input file");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");
  validPL->set<double>("Restart Time", 1.0, "Exodus solution time to read for inital guess/condition.");



  return validPL;
}
