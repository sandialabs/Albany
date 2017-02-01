//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>

#include "Albany_GmshSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"
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

Albany::GmshSTKMeshStruct::GmshSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                                              const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct (params, Teuchos::null)
{
  std::string fname = params->get("Gmsh Input Mesh File Name", "mesh.msh");

  if (commT->getRank() == 0)
  {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (!ifile.is_open())
    {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
    }

    std::string line;
    std::getline (ifile, line);

    bool legacy = false;
    bool binary = false;

    if (line=="$NOD")
    {
      legacy = true;
    }
    else if (line=="$MeshFormat")
    {
      std::getline (ifile, line);
      std::stringstream iss (line);

      float version;
      int doublesize;
      iss >> version >> binary >> doublesize;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Mesh format not recognized.\n");
    }
    ifile.close();

    if (legacy)
      loadLegacyMesh (fname);
    else if (binary)
      loadBinaryMesh (fname);
    else
      loadAsciiMesh (fname);
  }

  // GenericSTKMeshStruct's constructor could not initialize metaData, cause the dimension was not set
  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
  if(this->buildEMesh)
    entity_rank_names.push_back("FAMILY_TREE");
  metaData->initialize (this->numDim, entity_rank_names);

  double NumElemNodesD = NumElemNodes;
  Teuchos::broadcast<LO,ST>(*commT, 0, &NumElemNodesD);
  //Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromTeuchosComm(commT);
  //comm->Broadcast(&NumElemNodes, 1, 0);

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
  std::string ssn = "BoundarySide";
  ssNames.push_back(ssn);
  ssPartVec[ssn] = &metaData->declare_part(ssn, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssn]);
  stk::io::put_io_part_attribute(metaData->universal_part());
#endif

  // Counting boundaries
  std::set<int> bdTags;
  for (int i(0); i<NumSides; ++i)
    bdTags.insert(sides[NumSideNodes][i]);

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

    std::stringstream nsn_i,ssn_i;
    nsn_i << "BoundaryNode" << tag;
    ssn_i << "BoundarySide" << tag;

    bdTagToNodeSetName[tag] = nsn_i.str();
    bdTagToSideSetName[tag] = ssn_i.str();

    nsNames.push_back(nsn_i.str());
    ssNames.push_back(ssn_i.str());

    nsPartVec[nsn_i.str()] = &metaData->declare_part(nsn_i.str(), stk::topology::NODE_RANK);
    ssPartVec[ssn_i.str()] = &metaData->declare_part(ssn_i.str(), metaData->side_rank());

#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn_i.str()]);
    stk::io::put_io_part_attribute(*ssPartVec[ssn_i.str()]);
#endif
  }

  Teuchos::broadcast<LO,LO>(*commT, 0, &NumElemNodes);
  switch (NumElemNodes)
  {
    case 3:
      stk::mesh::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
      stk::mesh::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
      break;
    case 4:
      if (NumSideNodes==3)
      {
        stk::mesh::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
        stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssn]);
      }
      else
      {
        stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*partVec[0]);
        stk::mesh::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
      }
      break;
    case 8:
      stk::mesh::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
      stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssn]);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid number of element nodes (you should have got an error before though).\n");
  }

  numDim = 2;
  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, NumElems);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
  cullSubsetParts(ssNames, ssPartVec);
  this->meshSpecs[0] = Teuchos::rcp (
      new Albany::MeshSpecsStruct (ctd, numDim, cub, nsNames, ssNames,
                                   worksetSize, partVec[0]->name(), ebNameToIndex,
                                   this->interleavedOrdering));

  this->initializeSideSetMeshStructs(commT);
}

Albany::GmshSTKMeshStruct::~GmshSTKMeshStruct()
{
  delete[] pts;

  for (int i(0); i<5; ++i)
    delete[] tetra[i];
  for (int i(0); i<5; ++i)
    delete[] trias[i];
  for (int i(0); i<9; ++i)
    delete[] hexas[i];
  for (int i(0); i<5; ++i)
    delete[] quads[i];
  for (int i(0); i<3; ++i)
    delete[] lines[i];

  delete[] tetra;
  delete[] trias;
  delete[] hexas;
  delete[] quads;
  delete[] lines;
}

void Albany::GmshSTKMeshStruct::setFieldAndBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& commT,
    const Teuchos::RCP<Teuchos::ParameterList>& params,
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

  // Only proc 0 has loaded the file
  if (commT->getRank()==0)
  {
    stk::mesh::PartVector singlePartVec(1);
    unsigned int ebNo = 0; //element block #???
    int sideID = 0;

    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field =  fieldContainer->getCoordinatesField();

    singlePartVec[0] = nsPartVec["Node"];

    for (int i = 0; i < NumNodes; i++)
    {
      stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, i + 1, singlePartVec);

      double* coord;
      coord = stk::mesh::field_data(*coordinates_field, node);
      coord[0] = pts[i][0];
      coord[1] = pts[i][1];
      if (numDim==3)
        coord[2] = pts[i][2];
    }

    for (int i = 0; i < NumElems; i++)
    {
      singlePartVec[0] = partVec[ebNo];
      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, i + 1, singlePartVec);

      for (int j = 0; j < NumElemNodes; j++)
      {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, elems[j][i]);
        bulkData->declare_relation(elem, node, j);
      }

      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = commT->getRank();
    }

    std::string partName;
    stk::mesh::PartVector nsPartVec_i(1), ssPartVec_i(2);
    ssPartVec_i[0] = ssPartVec["BoundarySide"]; // The whole boundary side
    for (int i = 0; i < NumSides; i++)
    {
      std::map<int,int> elm_count;
      partName = bdTagToNodeSetName[sides[NumSideNodes][i]];
      nsPartVec_i[0] = nsPartVec[partName];

      partName = bdTagToSideSetName[sides[NumSideNodes][i]];
      ssPartVec_i[1] = ssPartVec[partName];

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), i + 1, ssPartVec_i);
      for (int j=0; j<NumSideNodes; ++j)
      {
        stk::mesh::Entity node_j = bulkData->get_entity(stk::topology::NODE_RANK,sides[j][i]);
        bulkData->change_entity_parts (node_j,nsPartVec_i); // Add node to the boundary nodeset
        bulkData->declare_relation(side, node_j, j);

        int num_e = bulkData->num_elements(node_j);
        const stk::mesh::Entity* e = bulkData->begin_elements(node_j);
        for (int k(0); k<num_e; ++k)
        {
          ++elm_count[bulkData->identifier(e[k])];
        }
      }

      // We have to find out what element has this side as a side. We check the node connectivity
      // In particular, the element that is connected to all NumSideNodes nodes is the one.
      bool found = false;

      for (auto e : elm_count)
        if (e.second==NumSideNodes)
        {
          stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEM_RANK, e.first);
          found = true;
          int num_sides = bulkData->num_sides(elem);
          bulkData->declare_relation(elem,side,num_sides);
          break;
        }

      TEUCHOS_TEST_FOR_EXCEPTION (found==false, std::logic_error, "Error! Cannot find element connected to side " << i+1 << ".\n");
    }

  }
  bulkData->modification_end();

#ifdef ALBANY_ZOLTAN
  // Gmsh is for sure using a serial mesh. We hard code it here, in case the user did not set it
  params->set<bool>("Use Serial Mesh", true);

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

Teuchos::RCP<const Teuchos::ParameterList> Albany::GmshSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid ASCII_DiscParams");
  validPL->set<std::string>("Gmsh Input Mesh File Name", "mesh.msh",
      "Name of the file containing the 2D mesh, with list of coordinates, elements' connectivity and boundary edges' connectivity");

  return validPL;
}

// -------------------------------- Read methos ---------------------------- //

void Albany::GmshSTKMeshStruct::loadLegacyMesh (const std::string& fname)
{
  std::ifstream ifile;
  ifile.open(fname.c_str());
  if (!ifile.is_open())
  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
  }

  // Start reading nodes
  std::string line;
  while (std::getline (ifile, line) && line != "$NOD")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  // Read the number of nodes
  std::getline (ifile, line);
  NumNodes = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  pts = new double [NumNodes][3];

  // Read the nodes
  int id;
  for (int i=0; i<NumNodes; ++i)
  {
    ifile >> id >> pts[i][0] >> pts[i][1] >> pts[i][2];
  }

  // Start reading elements (cells and sides)
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$ELM")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);
  int num_entities = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  // Gmsh lists elements and sides (and some points) all toghether, and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D

  int nb_tetra(0), nb_hexa(0), nb_tria(0), nb_quad(0), nb_line(0), e_type(0);
  for (int i(0); i<num_entities; ++i)
  {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type;

    switch (e_type)
    {
      case 1: ++nb_line;  break;
      case 2: ++nb_tria;  break;
      case 3: ++nb_quad;  break;
      case 4: ++nb_tetra; break;
      case 5: ++nb_hexa;  break;
      case 15: /*point*/  break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported.\n");
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexa!=0, std::logic_error, "Error! Cannot mix tetrahedra and hexahedra.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tria*nb_quad!=0, std::logic_error, "Error! Cannot mix triangles and quadrilaterals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexa+nb_tria+nb_quad==0, std::logic_error, "Error! Can only handle 2D and 3D geometries.\n");

  lines = new int*[3];
  tetra = new int*[5];
  trias = new int*[4];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i)
    tetra[i] = new int[nb_tetra];
  for (int i(0); i<5; ++i)
    trias[i] = new int[nb_tria];
  for (int i(0); i<9; ++i)
    hexas[i] = new int[nb_hexa];
  for (int i(0); i<5; ++i)
    quads[i] = new int[nb_quad];
  for (int i(0); i<3; ++i)
    lines[i] = new int[nb_line];

  if (nb_tetra>0)
  {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_tria;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  }
  else if (nb_hexa>0)
  {
    this->numDim = 3;

    NumElems = nb_hexa;
    NumSides = nb_quad;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  }
  else if (nb_tria>0)
  {
    this->numDim = 2;

    NumElems = nb_tria;
    NumSides = nb_line;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  }
  else
  {
    this->numDim = 2;

    NumElems = nb_quad;
    NumSides = nb_line;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  }

  // Reset the stream to the beginning of the element section
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$ELM")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found; however, it was found earlier. This may be a bug.\n");
  std::getline(ifile,line); // Skip line with number of elements

  // Read the elements
  int reg_phys, reg_elem, n_nodes;
  int iline(0), itria(0), iquad(0), itetra(0), ihexa(0);
  for (int i(0); i<num_entities; ++i)
  {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type >> reg_phys >> reg_elem >> n_nodes;
    switch (e_type)
    {
      case 1: // 2-pt Line
        ss >> lines[0][iline] >> lines[1][iline];
        lines[2][iline] = reg_phys;
        ++iline;
        break;
      case 2: // 3-pt Triangle
        ss >> trias[0][itria] >> trias[1][itria] >> trias[2][itria];
        trias[4][itria] = reg_phys;
        ++itria;
        break;
      case 3: // 4-pt Quad
        ss >> quads[0][iquad] >> quads[1][iquad] >> quads[2][iquad] >> quads[3][iquad];
        quads[4][iquad] = reg_phys;
        ++iquad;
        break;
      case 4: // 4-pt Tetra
        ss >> tetra[0][itetra] >> tetra[1][itetra] >> tetra[2][itetra] >> tetra[3][itetra];
        trias[4][itetra] = reg_phys;
        ++itria;
        break;
      case 5: // 8-pt Hexa
        ss >> hexas[0][ihexa] >> hexas[1][ihexa] >> hexas[2][ihexa] >> hexas[3][ihexa]
           >> hexas[4][ihexa] >> hexas[5][ihexa] >> hexas[6][ihexa] >> hexas[7][ihexa];
        hexas[8][ihexa] = reg_phys;
        ++ihexa;
        break;
      case 15: // Point
          break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported; but you should have got an error before!\n");
    }
  }

  // Close the input stream
  ifile.close();
}

void Albany::GmshSTKMeshStruct::loadAsciiMesh (const std::string& fname)
{
  std::ifstream ifile;
  ifile.open(fname.c_str());
  if (!ifile.is_open())
  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
  }

  // Start reading nodes
  std::string line;
  while (std::getline (ifile, line) && line != "$Nodes")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  // Read the number of nodes
  std::getline (ifile, line);
  NumNodes = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  pts = new double [NumNodes][3];

  // Read the nodes
  int id;
  for (int i=0; i<NumNodes; ++i)
  {
    ifile >> id >> pts[i][0] >> pts[i][1] >> pts[i][2];
  }

  // Start reading elements (cells and sides)
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$Elements")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);
  int num_entities = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  // Gmsh lists elements and sides (and some points) all toghether, and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D

  int nb_tetra(0), nb_hexa(0), nb_tria(0), nb_quad(0), nb_line(0), e_type(0);
  for (int i(0); i<num_entities; ++i)
  {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type;

    switch (e_type)
    {
      case 1: ++nb_line;  break;
      case 2: ++nb_tria;  break;
      case 3: ++nb_quad;  break;
      case 4: ++nb_tetra; break;
      case 5: ++nb_hexa;  break;
      case 15: /*point*/  break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported.\n");
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexa!=0, std::logic_error, "Error! Cannot mix tetrahedra and hexahedra.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tria*nb_quad!=0, std::logic_error, "Error! Cannot mix triangles and quadrilaterals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexa+nb_tria+nb_quad==0, std::logic_error, "Error! Can only handle 2D and 3D geometries.\n");

  lines = new int*[3];
  tetra = new int*[5];
  trias = new int*[4];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i)
    tetra[i] = new int[nb_tetra];
  for (int i(0); i<5; ++i)
    trias[i] = new int[nb_tria];
  for (int i(0); i<9; ++i)
    hexas[i] = new int[nb_hexa];
  for (int i(0); i<5; ++i)
    quads[i] = new int[nb_quad];
  for (int i(0); i<3; ++i)
    lines[i] = new int[nb_line];

  if (nb_tetra>0)
  {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_tria;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  }
  else if (nb_hexa>0)
  {
    this->numDim = 3;

    NumElems = nb_hexa;
    NumSides = nb_quad;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  }
  else if (nb_tria>0)
  {
    this->numDim = 2;

    NumElems = nb_tria;
    NumSides = nb_line;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  }
  else
  {
    this->numDim = 2;

    NumElems = nb_quad;
    NumSides = nb_line;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  }

  // Reset the stream to the beginning of the element section
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$Elements")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found; however, it was found earlier. This may be a bug.\n");
  std::getline(ifile,line); // Skip line with number of elements

  // Read the elements
  int iline(0), itria(0), iquad(0), itetra(0), ihexa(0), n_tags(0);
  std::vector<int> tags;
  for (int i(0); i<num_entities; ++i)
  {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type >> n_tags;
    TEUCHOS_TEST_FOR_EXCEPTION (n_tags<=0, Teuchos::Exceptions::InvalidParameter, "Error! Number of tags must be positive.\n");
    tags.resize(n_tags+1);
    for (int j(0); j<n_tags; ++j)
      ss >> tags[j];
    tags[n_tags] = 0;

    switch (e_type)
    {
      case 1: // 2-pt Line
        ss >> lines[0][iline] >> lines[1][iline];
        lines[2][iline] = tags[0];
        ++iline;
        break;
      case 2: // 3-pt Triangle
        ss >> trias[0][itria] >> trias[1][itria] >> trias[2][itria];
        trias[4][itria] = tags[0];
        ++itria;
        break;
      case 3: // 4-pt Quad
        ss >> quads[0][iquad] >> quads[1][iquad] >> quads[2][iquad] >> quads[3][iquad];
        quads[4][iquad] = tags[0];
        ++iquad;
        break;
      case 4: // 4-pt Tetra
        ss >> tetra[0][itetra] >> tetra[1][itetra] >> tetra[2][itetra] >> tetra[3][itetra];
        trias[4][itetra] = tags[0];
        ++itria;
        break;
      case 5: // 8-pt Hexa
        ss >> hexas[0][ihexa] >> hexas[1][ihexa] >> hexas[2][ihexa] >> hexas[3][ihexa]
           >> hexas[4][ihexa] >> hexas[5][ihexa] >> hexas[6][ihexa] >> hexas[7][ihexa];
        hexas[8][ihexa] = tags[0];
        ++ihexa;
        break;
      case 15: // Point
          break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported; but you should have got an error before!\n");
    }
  }

  // Close the input stream
  ifile.close();
}

void Albany::GmshSTKMeshStruct::loadBinaryMesh (const std::string& fname)
{
  std::ifstream ifile;
  ifile.open(fname.c_str());
  if (!ifile.is_open())
  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
  }

  std::string line;
  std::getline (ifile, line); // $MeshFormat
  std::getline (ifile, line); // 2.0 file-type data-size

  // Check file endianness
  union
  {
    int i;
    char c[sizeof (int)];
  } one;

  ifile.read (one.c, sizeof (int) );
  TEUCHOS_TEST_FOR_EXCEPTION (one.i!=1, std::runtime_error, "Error! Uncompatible binary format.\n");

  // Start reading nodes
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$Nodes")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  // Read the number of nodes
  std::getline (ifile, line);
  NumNodes = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  pts = new double[NumNodes][3];

  // Read the nodes
  int id;
  for (int i=0; i<NumNodes; ++i)
  {
    ifile.read (reinterpret_cast<char*> (&id), sizeof (int) );
    ifile.read (reinterpret_cast<char*> (pts[i]), 3 * sizeof (double) );
  }

  // Start reading elements (cells and sides)
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$Elements")
  {
    // Keep swallowing lines...
  }
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);
  int num_entities = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  // Gmsh lists elements and sides (and some points) all toghether, and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D
  std::vector<int> tmp;
  int nb_tetra(0), nb_hexa(0), nb_tria(0), nb_quad(0), nb_line(0), n_tags(0), e_type(0), entities_found(0);
  while (entities_found<num_entities)
  {
    int header[3];
    ifile.read(reinterpret_cast<char*> (header), 3*sizeof(int));

    TEUCHOS_TEST_FOR_EXCEPTION (header[1]<=0, std::logic_error, "Error! Invalid number of elements of this type.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (header[2]<=0, std::logic_error, "Error! Invalid number of tags.\n");

    e_type = header[0];
    entities_found += header[1];
    n_tags = header[2];

    int length;
    switch (e_type)
    {
      case 1: // 2-pt Line
        length = 1+n_tags+2; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_line += header[1];
        break;
      case 2: // 3-pt Triangle
        length = 1+n_tags+3; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_tria += header[1];
        break;
      case 3: // 4-pt Quad
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quad += header[1];
        break;
      case 4: // 4-pt Tetra
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quad += header[1];
        break;
      case 5: // 8-pt Hexa
        length = 1+n_tags+8; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quad += header[1];
        break;
      case 15: // Point
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported.\n");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexa!=0, std::logic_error, "Error! Cannot mix tetrahedra and hexahedra.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tria*nb_quad!=0, std::logic_error, "Error! Cannot mix triangles and quadrilaterals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexa+nb_tria+nb_quad==0, std::logic_error, "Error! Can only handle 2D and 3D geometries.\n");

  lines = new int*[3];
  tetra = new int*[5];
  trias = new int*[4];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i)
    tetra[i] = new int[nb_tetra];
  for (int i(0); i<5; ++i)
    trias[i] = new int[nb_tria];
  for (int i(0); i<9; ++i)
    hexas[i] = new int[nb_hexa];
  for (int i(0); i<5; ++i)
    quads[i] = new int[nb_quad];
  for (int i(0); i<3; ++i)
    lines[i] = new int[nb_line];

  if (nb_tetra>0)
  {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_tria;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  }
  else if (nb_hexa>0)
  {
    this->numDim = 3;

    NumElems = nb_hexa;
    NumSides = nb_quad;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  }
  else if (nb_tria>0)
  {
    this->numDim = 2;

    NumElems = nb_tria;
    NumSides = nb_line;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  }
  else
  {
    this->numDim = 2;

    NumElems = nb_quad;
    NumSides = nb_line;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  }

  // Reset the stream to the beginning of the element section
  ifile.seekg (0, std::ios::beg);
  while (std::getline (ifile, line) && line != "$Elements")
  {
    // Keep swallowing lines...
  }
  std::getline(ifile,line); // Skip line with number of elements

  entities_found = 0;
  int iline(0), itria(0), iquad(0), itetra(0), ihexa(0);
  while (entities_found<num_entities)
  {
    int header[3];
    ifile.read(reinterpret_cast<char*> (header), 3*sizeof(int));

    TEUCHOS_TEST_FOR_EXCEPTION (header[1]<=0, std::logic_error, "Error! Invalid number of elements of this type.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (header[2]<=0, std::logic_error, "Error! Invalid number of tags.\n");

    e_type = header[0];
    entities_found += header[1];
    n_tags = header[2];

    int length;
    switch (e_type)
    {
      case 1: // 2-pt Line
        length = 1+n_tags+2; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++iline)
        {
          lines[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          lines[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          lines[2][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 2: // 3-pt Triangle
        length = 1+n_tags+3; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++itria)
        {
          trias[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          trias[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          trias[2][j] = tmp[sizeof(int)*(j*length+1+n_tags+2)];
          trias[3][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 3: // 4-pt Quad
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++iquad)
        {
          quads[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          quads[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          quads[2][j] = tmp[sizeof(int)*(j*length+1+n_tags+2)];
          quads[3][j] = tmp[sizeof(int)*(j*length+1+n_tags+3)];
          quads[4][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 4: // 4-pt Tetra
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++itetra)
        {
          tetra[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          tetra[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          tetra[2][j] = tmp[sizeof(int)*(j*length+1+n_tags+2)];
          tetra[3][j] = tmp[sizeof(int)*(j*length+1+n_tags+3)];
          tetra[4][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 5: // 8-pt Hexa
        length = 1+n_tags+8; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++ihexa)
        {
          hexas[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          hexas[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          hexas[2][j] = tmp[sizeof(int)*(j*length+1+n_tags+2)];
          hexas[3][j] = tmp[sizeof(int)*(j*length+1+n_tags+3)];
          hexas[4][j] = tmp[sizeof(int)*(j*length+1+n_tags+4)];
          hexas[5][j] = tmp[sizeof(int)*(j*length+1+n_tags+5)];
          hexas[6][j] = tmp[sizeof(int)*(j*length+1+n_tags+6)];
          hexas[7][j] = tmp[sizeof(int)*(j*length+1+n_tags+7)];
          hexas[8][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 15: // Point
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported.\n");
    }
  }

  // Close the input stream
  ifile.close();
}
