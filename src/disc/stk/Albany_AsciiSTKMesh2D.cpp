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
                                        const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct (params, Teuchos::null, 2),
  out                  (Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic             (false)
{
  xyz  = 0;
  eles = 0;
  be   = 0;
  NumElemNodes = NumNodes = NumEles = NumBdEdges = 0;

  std::string fname = params->get("Ascii Input Mesh File Name", "greenland.msh");

  std::string shape;
  if (commT->getRank() == 0)
  {
    std::ifstream ifile;

    NumElemNodes = 0;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> shape >> NumElemNodes;
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


      ifile >> NumNodes >> NumEles >> NumBdEdges;
      //read in nodes coordinates
      xyz = new double[NumNodes][3];
      for (int i = 0; i < NumNodes; i++) {
        ifile >> xyz[i][0] >> xyz[i][1] >> xyz[i][2];
      }

      //read in element connectivity
      eles = new int[NumEles][4];
      int temp;

      if(shape == "Triangle")
      {
        for (int i = 0; i < NumEles; i++)
        {
          ifile >> eles[i][0] >> eles[i][1] >> eles[i][2] >> temp;
        }
      }
      else
      {
        for (int i = 0; i < NumEles; i++)
        {
          ifile >> eles[i][0] >> eles[i][1] >> eles[i][2] >> eles[i][3] >> temp;
        }
      }

      //read in boundary edges connectivity
      be = new int[NumBdEdges][3];
      for (int i = 0; i < NumBdEdges; i++)
      {
        ifile >> be[i][0] >> be[i][1] >> be[i][2];
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
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  //  stk::io::put_io_part_attribute(metaData->universal_part());
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  // All the nodes
  std::vector < std::string > nsNames;
  std::string nsn = "Node";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  // All the sidesets
  std::vector < std::string > ssNames;
  std::string ssn = "LateralSide";
  ssNames.push_back(ssn);
  ssPartVec[ssn] = &metaData->declare_part(ssn, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif

  // Counting boundaries
  std::set<int> bdTags;
  for (int i(0); i<NumBdEdges; ++i)
    bdTags.insert(be[i][2]);

  // Broadcasting the tags
  int numBdTags = bdTags.size();
  Teuchos::broadcast<LO,LO>(*commT, 0, &numBdTags);
  int* bdTagsArray = new int[numBdTags];
  std::set<int>::iterator it=bdTags.begin();
  for (int k=0; it!=bdTags.end(); ++it,++k)
    bdTagsArray[k] = *it;
  Teuchos::broadcast<LO,LO>(*commT, 0, numBdTags, bdTagsArray);

  // Adding boundary nodesets and sidesets separating different labels
  for (int k=0; k<numBdTags; ++k)
  {
    int tag = bdTagsArray[k];

    std::stringstream nsn,ssn;
    nsn << "BoundaryNode" << tag;
    ssn << "LateralSide"  << tag;

    bdTagToNodeSetName[tag] = nsn.str();
    bdTagToSideSetName[tag] = ssn.str();

    nsNames.push_back(nsn.str());
    ssNames.push_back(ssn.str());

    nsPartVec[nsn.str()] = &metaData->declare_part(nsn.str(), stk::topology::NODE_RANK);
    ssPartVec[ssn.str()] = &metaData->declare_part(ssn.str(), metaData->side_rank());

#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn.str()]);
    stk::io::put_io_part_attribute(*ssPartVec[ssn.str()]);

#endif
  }

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(metaData->universal_part());
#endif

  Teuchos::broadcast<LO,LO>(*commT, 0, &NumElemNodes);
  if(NumElemNodes == 3)
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
  else
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*partVec[0]);

  stk::mesh::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
  numDim = 2;
  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  Teuchos::broadcast<LO,LO>(*commT, 0, &NumEles);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, NumEles);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
  cullSubsetParts(ssNames, ssPartVec);
  this->meshSpecs[0] = Teuchos::rcp (
      new Albany::MeshSpecsStruct (ctd, numDim, cub, nsNames, ssNames,
                                   worksetSize, partVec[0]->name(), ebNameToIndex,
                                   this->interleavedOrdering));

  this->initializeSideSetMeshStructs(commT);
}

Albany::AsciiSTKMesh2D::~AsciiSTKMesh2D() {
  delete[] xyz;
  delete[] be;
  delete[] eles;
}

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
    int sideID = 0;

    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field =
        fieldContainer->getProcRankField();
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
        fieldContainer->getCoordinatesField();

    singlePartVec[0] = nsPartVec["Node"];

    *out << "[AsciiSTKMesh2D] Adding nodes... ";
    out->getOStream()->flush();
    for (int i = 0; i < NumNodes; i++) {
      stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK,
          i + 1, singlePartVec);

      double* coord;
      coord = stk::mesh::field_data(*coordinates_field, node);
      coord[0] = xyz[i][0];
      coord[1] = xyz[i][1];
      coord[2] = 0.; //xyz[i][2];
    }
    *out << "done!\n";
    out->getOStream()->flush();

    *out << "[AsciiSTKMesh2D] Adding elements... ";
    out->getOStream()->flush();
    for (int i = 0; i < NumEles; i++) {

      singlePartVec[0] = partVec[ebNo];
      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK,
          i + 1, singlePartVec);

      for (int j = 0; j < NumElemNodes; j++) {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK,
            eles[i][j]);
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
      partName = bdTagToNodeSetName[be[i][2]];
      singlePartVec[0] = nsPartVec[partName];

      edgeMap.insert(std::pair<std::pair<int, int>, int>(std::make_pair(be[i][0], be[i][1]), i + 1));
      stk::mesh::Entity node1 = bulkData->declare_entity(stk::topology::NODE_RANK,be[i][0],singlePartVec);
      stk::mesh::Entity node2 = bulkData->declare_entity(stk::topology::NODE_RANK,be[i][1],singlePartVec);
    }

    *out << "done!\n";
    out->getOStream()->flush();

    *out << "[AsciiSTKMesh2D] Adding side sets... ";
    out->getOStream()->flush();

    stk::mesh::PartVector multiPartVec(2);
    multiPartVec[0] = ssPartVec["LateralSide"];
    for (int i = 0; i < NumEles; i++)
    {
      for (int j = 0; j < NumElemNodes; j++)
      {
        std::map<std::pair<int, int>, int>::iterator it = edgeMap.find(
          std::make_pair(eles[i][j], eles[i][(j + 1) % NumElemNodes]));

        if (it == edgeMap.end()) it = edgeMap.find(std::make_pair(eles[i][(j + 1) % NumElemNodes], eles[i][j]));

        if (it != edgeMap.end())
        {
          partName = bdTagToSideSetName.at(be[it->second-1][2]); // -1 because stk starts from 1, and use .at() for safety
          multiPartVec[1] = ssPartVec.at(partName);
          stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), it->second, multiPartVec);
          stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, i + 1);
          bulkData->declare_relation(elem, side, j /*local side id*/);
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

  // Refine the mesh before starting the simulation if indicated
  uniformRefineMesh(commT);

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
