//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include "Albany_AsciiSTKMesh2D.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_CommHelpers.hpp"

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

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

Albany::AsciiSTKMesh2D::AsciiSTKMesh2D (const Teuchos::RCP<Teuchos::ParameterList>& params,
                                        const Teuchos::RCP<const Teuchos_Comm>& commT,
					const int numParams) :
  GenericSTKMeshStruct (params, Teuchos::null, 2, numParams),
  periodic             (false)
{
  NumElemNodes = NumNodes = NumElems = NumBdEdges = 0;

  std::string fname = params->get("Ascii Input Mesh File Name", "greenland.msh");

  std::string shape, word;
  int number;
  bool globalIds = false;
  if (commT->getRank() == 0)
  {
    std::ifstream ifile;

    NumElemNodes = 0;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> word >> number;
      // If present, check the format of the mesh. If 0, old format, if 1, the mesh is also providing the global ids of node, triangles, edges)
      if(word == "Format:") {
        globalIds = (number > 0);
        ifile >> shape >> NumElemNodes;
      } else {
        shape = word;
        NumElemNodes = number;
      }
      if(shape == "Triangle")
      {
        TEUCHOS_TEST_FOR_EXCEPTION(NumElemNodes != 3, Teuchos::Exceptions::InvalidParameter,
                  std::endl << "Error in AsciiSTKMesh2D: Triangles must be linear. Number of nodes per element in file " << fname << " is: " << NumElemNodes << ". Should be 3!" << std::endl);
      }
      else if(shape == "Quadrilateral")
      {
        TEUCHOS_TEST_FOR_EXCEPTION(NumElemNodes != 4, Teuchos::Exceptions::InvalidParameter,
                  std::endl << "Error in AsciiSTKMesh2D: Quadrilaterals must be bilinear. Number of nodes per element in file " << fname << " is: "  << " is: " << NumElemNodes << ". Should be 4!" << std::endl);
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                          std::endl << "Error in AsciiSTKMesh2D: Only Triangle or Quadrilateral grids can be imported. Shape in in file " << fname << " is: " << shape << ". Should be Triangle or Quadrialteral" << std::endl);


      ifile >> NumNodes >> NumElems >> NumBdEdges;
      //read in nodes coordinates
      coords.resize(NumNodes);
      for(auto& coord:coords) coord.resize(2);
      elems.resize(NumElems);
      for(auto& elem:elems) elem.resize(NumElemNodes);
      bdEdges.resize(NumBdEdges);
      for(auto& edge:bdEdges) edge.resize(3);

      coord_flags.resize(NumNodes);
      coord_Ids.resize(NumNodes);
      ele_Ids.resize(NumElems);
      be_Ids.resize(NumBdEdges);
      if(!globalIds) {
        //initialize coords_Ids, ele_Ids and be_Ids vectors as 1,2,3,..., vector.size()
        for (int i=0; i<NumNodes; ++i) coord_Ids[i] = i+1;
        for (int i=0; i<NumElems; ++i) ele_Ids[i] = i+1;
        for (int i=0; i<NumBdEdges; ++i) be_Ids[i] = i+1;
      }

      for (int i = 0; i < NumNodes; i++) {
        if(globalIds)
          ifile >> coord_Ids[i] >> coords[i][0] >> coords[i][1] >> coord_flags[i];
        else
          ifile >> coords[i][0] >> coords[i][1] >> coord_flags[i];
      }

      int temp(0);
      if(shape == "Triangle")
      {
        for (int i = 0; i < NumElems; i++)
        {
          if(globalIds)
            ifile >> ele_Ids[i] >> elems[i][0] >> elems[i][1] >> elems[i][2] >>  temp;
          else
            ifile >> elems[i][0] >> elems[i][1] >> elems[i][2] >>  temp;
        }
      } else { // Quadrilateral
        for (int i = 0; i < NumElems; i++)
        {
          if(globalIds)
            ifile >> ele_Ids[i] >> elems[i][0] >> elems[i][1] >> elems[i][2] >> elems[i][3] >> temp;
          else
            ifile >> elems[i][0] >> elems[i][1] >> elems[i][2] >> elems[i][3] >> temp;
        }
      }


      for (int i = 0; i < NumBdEdges; i++)
      {
        if(globalIds)
          ifile >> be_Ids[i] >> bdEdges[i][0] >> bdEdges[i][1] >> bdEdges[i][2];
        else
          ifile >> bdEdges[i][0] >> bdEdges[i][1] >> bdEdges[i][2];
      }
      ifile.close();
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in AsciiSTKMesh2D: Input Mesh File " << fname << " not found!"<< std::endl);
    }
  }

  params->validateParameters(*getValidDiscretizationParameters(), 0);

  std::string ebn = "Element Block 0";
  partVec[0] = &metaData->declare_part(ebn, stk::topology::ELEMENT_RANK);
  std::map<std::string,int> ebNameToIndex;
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  //  stk::io::put_io_part_attribute(metaData->universal_part());
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  // All the nodes
  std::vector < std::string > nsNames;
  std::string nsn = "node_set";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  // All the sidesets
  std::vector < std::string > ssNames;
  std::string ssn = "boundary_side_set";
  ssNames.push_back(ssn);
  ssPartVec[ssn] = &metaData->declare_part(ssn, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif

  // Counting boundaries

  std::set<int> bdNodeTags, bdEdgeTags;
  for (int i(0); i<NumBdEdges; ++i)
    bdEdgeTags.insert(bdEdges[i][2]);
  for (int i(0); i<NumNodes; ++i)
    bdNodeTags.insert(coord_flags[i]);

  // Broadcasting the tags
  int numBdNodeTags = bdNodeTags.size();
  Teuchos::broadcast<LO,LO>(*commT, 0, &numBdNodeTags);
  std::vector<int> bdNodeTagsArray(numBdNodeTags);
  std::set<int>::iterator it=bdNodeTags.begin();
  for (int k=0; it!=bdNodeTags.end(); ++it,++k)
    bdNodeTagsArray[k] = *it;
  Teuchos::broadcast<LO,LO>(*commT, 0, numBdNodeTags, bdNodeTagsArray.data());

  // Adding boundary nodesets and sidesets separating different labels
  for (int k=0; k<numBdNodeTags; ++k)
  {
    int tag = bdNodeTagsArray[k];

    std::stringstream nsn_ss;
    nsn_ss << "boundary_node_set_" << tag;

    bdTagToNodeSetName[tag] = nsn_ss.str();

    nsNames.push_back(nsn_ss.str());

    nsPartVec[nsn_ss.str()] = &metaData->declare_part(nsn_ss.str(), stk::topology::NODE_RANK);

#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn_ss.str()]);

#endif
  }


  // Broadcasting the tags
  int numBdEdgeTags = bdEdgeTags.size();
  Teuchos::broadcast<LO,LO>(*commT, 0, &numBdEdgeTags);
  std::vector<int> bdEdgeTagsArray(numBdEdgeTags);
  it=bdEdgeTags.begin();
  for (int k=0; it!=bdEdgeTags.end(); ++it,++k)
    bdEdgeTagsArray[k] = *it;
  Teuchos::broadcast<LO,LO>(*commT, 0, numBdEdgeTags, bdEdgeTagsArray.data());

  // Adding boundary nodesets and sidesets separating different labels
  for (int k=0; k<numBdEdgeTags; ++k)
  {
    int tag = bdEdgeTagsArray[k];

    std::stringstream ssn_ss;
    ssn_ss << "boundary_side_set_"  << tag;

    bdTagToSideSetName[tag] = ssn_ss.str();

    ssNames.push_back(ssn_ss.str());

    ssPartVec[ssn_ss.str()] = &metaData->declare_part(ssn_ss.str(), metaData->side_rank());

#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn_ss.str()]);
#endif
  }

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(metaData->universal_part());
#endif

  Teuchos::broadcast<LO,LO>(*commT, 0, &NumElemNodes);
  if(NumElemNodes == 3) {
    stk::mesh::set_topology(*partVec[0], stk::topology::TRI_3_2D);
  }
  else {
    stk::mesh::set_topology(*partVec[0], stk::topology::QUAD_4_2D); 
  }

  stk::mesh::set_topology(*ssPartVec[ssn], stk::topology::LINE_2);
  numDim = 2;
  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  Teuchos::broadcast<LO,LO>(*commT, 0, &NumElems);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, NumElems);
  

  stk::topology stk_topo_data = metaData->get_topology( *partVec[0] );
  shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data); 
  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData(); 

  cullSubsetParts(ssNames, ssPartVec);
  this->meshSpecs[0] = Teuchos::rcp (
      new Albany::MeshSpecsStruct (ctd, numDim, cub, nsNames, ssNames,
                                   worksetSize, partVec[0]->name(), ebNameToIndex,
                                   this->interleavedOrdering));

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(commT);

  // Initialize the requested sideset mesh struct in the mesh
  this->initializeSideSetMeshStructs(commT);
}

Albany::AsciiSTKMesh2D::~AsciiSTKMesh2D() {}

void Albany::AsciiSTKMesh2D::setFieldAndBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& commT,
    const Teuchos::RCP<Teuchos::ParameterList>& /*params_*/,
    const unsigned int neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(commT->getRank(), commT->getSize());
  out->setOutputToRootOnly(0);

  // Only proc 0 has loaded the file
  if (commT->getRank()==0)
  {
    stk::mesh::PartVector singlePartVec(1);
    unsigned int ebNo = 0; //element block #???

    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field =
        fieldContainer->getProcRankField();
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
        fieldContainer->getCoordinatesField();

    singlePartVec[0] = nsPartVec["node_set"];

    *out << "[AsciiSTKMesh2D] Adding nodes... ";
    out->getOStream()->flush();
    for (int i = 0; i < NumNodes; i++) {
      stk::mesh::Entity node = bulkData->declare_node(coord_Ids[i], singlePartVec);

      double* coord;
      coord = stk::mesh::field_data(*coordinates_field, node);
      coord[0] = coords[i][0];
      coord[1] = coords[i][1];
      coord[2] = 0.;
    }
    *out << "done!\n";
    out->getOStream()->flush();

    *out << "[AsciiSTKMesh2D] Adding elements... ";
    out->getOStream()->flush();
    for (int i = 0; i < NumElems; i++) {

      singlePartVec[0] = partVec[ebNo];
      stk::mesh::Entity elem = bulkData->declare_element(ele_Ids[i], singlePartVec);

      for (int j = 0; j < NumElemNodes; j++) {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK,
            coord_Ids[elems[i][j]-1]);
        bulkData->declare_relation(elem, node, j);
      }

      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = commT->getRank();
    }
    *out << "done!\n";
    out->getOStream()->flush();

    *out << "[AsciiSTKMesh2D] Adding node sets... ";
    out->getOStream()->flush();
    std::map<std::pair<int, int>, int> edgeMap;
    std::string partName;
    for (int i = 0; i < NumBdEdges; i++)
    {
      auto node1_lid = bdEdges[i][0]-1, node2_lid = bdEdges[i][1]-1;
      auto node1_id = coord_Ids[node1_lid], node2_id = coord_Ids[node2_lid];
      edgeMap.insert(std::pair<std::pair<int, int>, int>(std::make_pair(node1_id, node2_id), i));

      //create parts for boundary nodes
      partName = bdTagToNodeSetName[coord_flags[node1_lid]];
      singlePartVec[0] = nsPartVec[partName];
      /* stk::mesh::Entity node1 = */ bulkData->declare_node(node1_id,singlePartVec);
      partName = bdTagToNodeSetName[coord_flags[node2_lid]];
      singlePartVec[0] = nsPartVec[partName];
      /* stk::mesh::Entity node2 = */ bulkData->declare_node(node2_id,singlePartVec);
    }

    *out << "done!\n";
    out->getOStream()->flush();

    *out << "[AsciiSTKMesh2D] Adding side sets... ";
    out->getOStream()->flush();

    stk::mesh::PartVector multiPartVec(2);
    multiPartVec[0] = ssPartVec["boundary_side_set"];
    for (int i = 0; i < NumElems; i++)
    {
      for (int j = 0; j < NumElemNodes; j++)
      {
        auto node1 = coord_Ids[elems[i][j]-1], node2 = coord_Ids[elems[i][(j + 1) % NumElemNodes]-1];
        std::map<std::pair<int, int>, int>::iterator it = edgeMap.find(
          std::make_pair(node1, node2));

        if (it == edgeMap.end()) it = edgeMap.find(std::make_pair(node2, node1));

        if (it != edgeMap.end())
        {
          partName = bdTagToSideSetName.at(bdEdges[it->second][2]);
          multiPartVec[1] = ssPartVec.at(partName);
          stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, ele_Ids[i]);
          stk::mesh::Entity side = bulkData->declare_element_side(elem, j, multiPartVec);
          stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
          for (int k = 0; k < 2; k++)
          {
            stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[j].node[k]];
            bulkData->declare_relation(side, node, k);
          }
          edgeMap.erase(it);
        }
      }
    }

    *out << "done!\n";
    out->getOStream()->flush();
  }
  bulkData->modification_end();

#ifdef ALBANY_ZOLTAN
  // Ascii2D is for sure using a serial mesh. We hard code here the rebalance options, in case the user did not set it
  params->set<bool>("Use Serial Mesh", true);
  params->set<bool>("Rebalance Mesh", true);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);
#endif

  // Loading required input fields from file
  this->loadRequiredInputFields (req,commT);

  // Finally, perform the setup of the (possible) side set meshes (including extraction if of type SideSetSTKMeshStruct)
  this->finalizeSideSetMeshStructs(commT, side_set_req, side_set_sis, worksetSize);

  fieldAndBulkDataSet = true;
}

Teuchos::RCP<const Teuchos::ParameterList> Albany::AsciiSTKMesh2D::getValidDiscretizationParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getValidGenericSTKParameters("Valid ASCII_DiscParams");
  validPL->set<std::string>("Ascii Input Mesh File Name", "greenland.msh",
      "Name of the file containing the 2D mesh, with list of coordinates, elements' connectivity and boundary edges' connectivity");

  return validPL;
}
