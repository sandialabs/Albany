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
  wsElNodeID(wsElNodeID_),
  coords(coords_), 
  node_mapT(node_mapT_), 
  points_per_edge(points_per_edge_)
{
//FIXME, IKT, 5/7/15: it appears node_mapT is not needed.  In this case, remove it from 
//list of member functions and argument list. 
#ifdef OUTPUT_TO_SCREEN
  *out << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

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

  stk::mesh::set_cell_topology<shards::ShellQuadrilateral<4> >(*partVec[0]);

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

  //IKT, 4/24/15: the following algorithm should work for creating the connectivity of the STK 
  //bilinear mesh from the spectral mesh.  Needs to be uncommented and double checked.  
  //
  //IKT: for an example of how the following is defined, please see Albany_AsciiSTKMeshStruct.cpp . 

#ifdef OUTPUT_TO_SCREEN
  std::cout << "Spectral Mesh # ws, # eles: " << wsElNodeID.size() << ", " << wsElNodeID[0].size() << std::endl; 
  for (int ws = 0; ws < wsElNodeID.size(); ws++){           
    for (int e = 0; e < wsElNodeID[ws].size(); e++){        
      std::cout << "Spectral Mesh Element " << e << ": Nodes = ";
      for (size_t inode = 0; inode < points_per_edge*points_per_edge; ++inode)
        std::cout << wsElNodeID[ws][e][inode] << " ";
            std::cout << std::endl;
      }
  }
#endif

  for (int ws = 0; ws < wsElNodeID.size(); ws++){             // workset
    for (int e = 0; e < wsElNodeID[ws].size(); e++){          // cell
      for (int i=0; i<points_per_edge-1; i++) {           //Each spectral element broken into (points_per_edge-1)^2 bilinear elements
        for (int j=0; j<points_per_edge-1; j++) {
          //Set connectivity for new mesh  
          //We set the element # to the global node ID of the 0th node in each new bilinear element.  
          const unsigned int elem_GID = wsElNodeID[ws][e][i+j*points_per_edge];
          stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
          singlePartVec[0] = partVec[ebNo];
          //Add 1 to elem_id in the following line b/c STK is 1-based whereas wsElNodeID is 0-based 
          stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
          stk::mesh::Entity node0 = bulkData->declare_entity(stk::topology::NODE_RANK, 
                                    1+wsElNodeID[ws][e][i+j*points_per_edge], nodePartVec);
          stk::mesh::Entity node1 = bulkData->declare_entity(stk::topology::NODE_RANK, 
                                    1+wsElNodeID[ws][e][i+1+j*points_per_edge], nodePartVec);
          stk::mesh::Entity node2 = bulkData->declare_entity(stk::topology::NODE_RANK, 
                                    1+wsElNodeID[ws][e][i+points_per_edge+1+j*points_per_edge], nodePartVec);
          stk::mesh::Entity node3 = bulkData->declare_entity(stk::topology::NODE_RANK,
                                    1+wsElNodeID[ws][e][i+points_per_edge+j*points_per_edge], nodePartVec); 
#ifdef OUTPUT_TO_SCREEN
          std::cout << "ws, e, i , j " << ws << ", " << e << ", " << i << ", " << j << std::endl; 
          std::cout << "Output Mesh elem_GID, node0, node1, node2, node3: " << elem_GID << ", " 
                    << wsElNodeID[ws][e][i+j*points_per_edge] << ", " 
                    << wsElNodeID[ws][e][i+1+j*points_per_edge] << ", " 
                    << wsElNodeID[ws][e][i+points_per_edge+1+j*points_per_edge] << ", " 
                    << wsElNodeID[ws][e][i+points_per_edge+j*points_per_edge] 
                    << std::endl; 
#endif 
          bulkData->declare_relation(elem, node0, 0);
          bulkData->declare_relation(elem, node1, 1);
          bulkData->declare_relation(elem, node2, 2);
          bulkData->declare_relation(elem, node3, 3);

          //Set coordinates of new mesh
          double* coord;
          //set node 0 in STK bilinear mesh 
          coord = stk::mesh::field_data(*coordinates_field, node0);
#ifdef OUTPUT_TO_SCREEN
         std::cout << "Output mesh node0 coords: " << coords[ws][e][i+j*points_per_edge][0] 
                   << ", " << coords[ws][e][i+j*points_per_edge][1] << ", " << coords[ws][e][i+j*points_per_edge][2] << std::endl;
#endif 
          coord[0] = coords[ws][e][i+j*points_per_edge][0];
          coord[1] = coords[ws][e][i+j*points_per_edge][1];
          coord[2] = coords[ws][e][i+j*points_per_edge][2];
          //set node 1 in STK bilinear mesh 
          coord = stk::mesh::field_data(*coordinates_field, node1);
#ifdef OUTPUT_TO_SCREEN
         std::cout << "Output mesh node1 coords: " << coords[ws][e][i+1+j*points_per_edge][0] 
                   << ", " << coords[ws][e][i+1+j*points_per_edge][1] << ", " << coords[ws][e][i+1+j*points_per_edge][2] << std::endl;
#endif 
          coord[0] = coords[ws][e][i+1+j*points_per_edge][0];
          coord[1] = coords[ws][e][i+1+j*points_per_edge][1];
          coord[2] = coords[ws][e][i+1+j*points_per_edge][2];
          //set node 2 in STK bilinear mesh 
          coord = stk::mesh::field_data(*coordinates_field, node2);
#ifdef OUTPUT_TO_SCREEN
         std::cout << "Output mesh node2 coords: " << coords[ws][e][i+points_per_edge+1+j*points_per_edge][0] 
                   << ", " << coords[ws][e][i+points_per_edge+1+j*points_per_edge][1] 
                   << ", " << coords[ws][e][i+points_per_edge+1+j*points_per_edge][2] << std::endl;
#endif 
          coord[0] = coords[ws][e][i+points_per_edge+1+j*points_per_edge][0];   
          coord[1] = coords[ws][e][i+points_per_edge+1+j*points_per_edge][1];  
          coord[2] = coords[ws][e][i+points_per_edge+1+j*points_per_edge][2];  
          //set node 3 in STK bilinear mesh 
          coord = stk::mesh::field_data(*coordinates_field, node3);
#ifdef OUTPUT_TO_SCREEN
         std::cout << "Output mesh node3 coords: " << coords[ws][e][i+points_per_edge+j*points_per_edge][0] 
                   << ", " << coords[ws][e][i+points_per_edge+j*points_per_edge][1] 
                   << ", " << coords[ws][e][i+points_per_edge+j*points_per_edge][2] << std::endl;
#endif 
          coord[0] = coords[ws][e][i+points_per_edge+j*points_per_edge][0];   
          coord[1] = coords[ws][e][i+points_per_edge+j*points_per_edge][1];  
          coord[2] = coords[ws][e][i+points_per_edge+j*points_per_edge][2];  
        }
      }
    }
  }    


  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}

void 
Aeras::SpectralOutputSTKMeshStruct::copySolutionToOutputMesh(const Tpetra_Vector& solnT) {
//FIXME: fill in this function!  Can be deferred at for now.
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
