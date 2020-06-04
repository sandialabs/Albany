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
                                              const Teuchos::RCP<const Teuchos_Comm>& commT,
					      const int numParams) :
  GenericSTKMeshStruct (params, Teuchos::null, -1, numParams)
{
  fname = params->get("Gmsh Input Mesh File Name", "mesh.msh");

  // Init counters to 0, pointers to null
  init_counters_to_zero();
  init_pointers_to_null();
  

  // Reading the mesh on proc 0
  if (commT->getRank() == 0) 
  {
    bool legacy = false;
    bool binary = false;
    bool ascii  = false;

    determine_file_type( legacy, binary, ascii);

    if(legacy) 
    {
      loadLegacyMesh();
    } 
    else if(binary) 
    {
      loadBinaryMesh();
    } 
    else if(ascii) 
    {
      loadAsciiMesh();
    } 
    else 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Mesh format not recognized.\n");
    }
  }
  // Broadcasting topological information about the mesh to all procs
  broadcast_topology( commT);
  // Redundant for proc 0 but needed for all others processes
  set_version_enum_from_float();

  // GenericSTKMeshStruct's constructor could not initialize metaData, cause the dimension was not set.
  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
  metaData->initialize (this->numDim, entity_rank_names);

  params->validateParameters(*getValidDiscretizationParameters(), 0);

  create_element_block();

#ifdef ALBANY_SEACAS
  //  stk::io::put_io_part_attribute(metaData->universal_part());
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  // Set boundary (sideset, nodeset) information
  std::vector < std::string > nsNames;
  std::vector < std::string > ssNames;
  set_boundaries( commT, ssNames, nsNames);

  switch (this->numDim) {
    case 2:
      if (NumElemNodes==3) {
        stk::mesh::set_topology(*partVec[0], stk::topology::TRI_3_2D); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::LINE_2); 
        }
      }
      else if( NumElemNodes == 6)
      {
        stk::mesh::set_topology(*partVec[0], stk::topology::TRI_6_2D); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::LINE_3); 
        }
      } else {
        stk::mesh::set_topology(*partVec[0], stk::topology::QUAD_4_2D); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::LINE_2); 
        }
      }
      break;
    case 3:
      if (NumElemNodes==4) {
        stk::mesh::set_topology(*partVec[0], stk::topology::TET_4); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::TRI_3); 
        }
      }
      else if( NumElemNodes == 10)
      {
        stk::mesh::set_topology(*partVec[0], stk::topology::TET_10); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::TRI_6); 
        }
      } else {
        stk::mesh::set_topology(*partVec[0], stk::topology::HEX_8); 
        for (auto ss : ssPartVec) {
          stk::mesh::set_topology(*ss.second, stk::topology::QUAD_4); 
        }
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid number of element nodes (you should have got an error before though).\n");
  }

  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, NumElems);
  stk::topology stk_topo_data = metaData->get_topology( *partVec[0] );
  shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data); 
  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData(); 
  cullSubsetParts(ssNames, ssPartVec);
  this->meshSpecs[0] = Teuchos::rcp (
      new Albany::MeshSpecsStruct (ctd, numDim, cub, nsNames, ssNames,
                                   worksetSize, partVec[0]->name(),
                                   ebNameToIndex, this->interleavedOrdering));

  this->initializeSideSetMeshStructs(commT);
}

Albany::GmshSTKMeshStruct::~GmshSTKMeshStruct()
{
  delete[] pts;

  // In theory, if one is nonnull, then they all are,
  // but for safety, check each ptr individually
  if (tetra!=nullptr) {
    for (int i(0); i<5; ++i) {
      delete[] tetra[i];
    }
  }
  if (tet10!=nullptr) {
    for (int i(0); i<11; ++i) {
      delete[] tet10[i];
    }
  }
  if (trias!=nullptr) {
    for (int i(0); i<4; ++i) {
      delete[] trias[i];
    }
  }
  if (tri6!=nullptr) {
    for (int i(0); i<7; ++i) {
      delete[] tri6[i];
    }
  }
  if (hexas!=nullptr) {
    for (int i(0); i<9; ++i) {
      delete[] hexas[i];
    }
  }
  if (quads!=nullptr) {
    for (int i(0); i<5; ++i) {
      delete[] quads[i];
    }
  }
  if (lines!=nullptr) {
    for (int i(0); i<3; ++i) {
      delete[] lines[i];
    }
  }
  if (line3!=nullptr) {
    for (int i(0); i<4; ++i) {
      delete[] line3[i];
    }
  }

  delete[] tetra;
  delete[] trias;
  delete[] hexas;
  delete[] quads;
  delete[] lines;
  delete[] line3;

  allowable_gmsh_versions.clear();
}

void Albany::GmshSTKMeshStruct::determine_file_type( bool& legacy, bool& binary, bool& ascii)
{
  std::ifstream ifile;
  open_fname( ifile);

  std::string line;
  std::getline (ifile, line);

  if (line=="$NOD") {
    legacy = true;
  } else if (line=="$MeshFormat") {

    std::getline (ifile, line);
    std::stringstream iss (line);

    int doublesize;
    iss >> version_in >> binary >> doublesize;

    ascii = !binary;

    check_version( ifile);
  }

  ifile.close();
  return;
}

void Albany::GmshSTKMeshStruct::init_counters_to_zero()
{
  NumSides = 0;
  NumNodes = 0;
  NumElems = 0;
  nb_hexas = 0;
  nb_tetra = 0;
  nb_tet10 = 0;
  nb_quads = 0;
  nb_trias = 0;
  nb_tri6  = 0;
  nb_lines = 0;
  nb_line3 = 0;

  return;
}

void Albany::GmshSTKMeshStruct::init_pointers_to_null()
{
  pts   = nullptr;
  tetra = nullptr;
  tet10 = nullptr;
  hexas = nullptr;
  trias = nullptr;
  tri6  = nullptr;
  quads = nullptr;
  lines = nullptr;
  line3 = nullptr;
  return;
}

void Albany::GmshSTKMeshStruct::broadcast_topology( const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  Teuchos::broadcast(*commT, 0, 1, &this->numDim);
  Teuchos::broadcast(*commT, 0, 1, &NumElemNodes);
  Teuchos::broadcast(*commT, 0, 1, &NumSideNodes);
  Teuchos::broadcast(*commT, 0, 1, &NumElems);
  Teuchos::broadcast(*commT, 0, 1, &version_in);

  return;
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
  if (commT->getRank()==0) {
    stk::mesh::PartVector singlePartVec(1);
    unsigned int ebNo = 0; //element block #???
    int sideID = 0;

    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field =  fieldContainer->getCoordinatesField();

    singlePartVec[0] = nsPartVec["Node"];

    for (int i = 0; i < NumNodes; i++) {
      stk::mesh::Entity node = bulkData->declare_node(i + 1, singlePartVec);

      double* coord;
      coord = stk::mesh::field_data(*coordinates_field, node);
      coord[0] = pts[i][0];
      coord[1] = pts[i][1];
      if (numDim==3)
        coord[2] = pts[i][2];
    }

    for (int i = 0; i < NumElems; i++) {
      singlePartVec[0] = partVec[ebNo];
      stk::mesh::Entity elem = bulkData->declare_element(i + 1, singlePartVec);

      for (int j = 0; j < NumElemNodes; j++) {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, elems[j][i]);
        bulkData->declare_relation(elem, node, j);
      }

      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = commT->getRank();
    }

    std::string partName;
    stk::mesh::PartVector nsPartVec_i(1), ssPartVec_i(2);
    ssPartVec_i[0] = ssPartVec["BoundarySide"]; // The whole boundary side
    for (int i = 0; i < NumSides; i++) {
      std::map<int,int> elm_count;
      partName = bdTagToNodeSetName[sides[NumSideNodes][i]];
      nsPartVec_i[0] = nsPartVec[partName];

      partName = bdTagToSideSetName[sides[NumSideNodes][i]];
      ssPartVec_i[1] = ssPartVec[partName];

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), i + 1, ssPartVec_i);
      for (int j=0; j<NumSideNodes; ++j) {
        stk::mesh::Entity node_j = bulkData->get_entity(stk::topology::NODE_RANK,sides[j][i]);
        bulkData->change_entity_parts (node_j,nsPartVec_i); // Add node to the boundary nodeset
        bulkData->declare_relation(side, node_j, j);

        int num_e = bulkData->num_elements(node_j);
        const stk::mesh::Entity* e = bulkData->begin_elements(node_j);
        for (int k(0); k<num_e; ++k) {
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

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);
#endif

  // Setting the default 3d coordinates
  this->setDefaultCoordinates3d();

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

void Albany::GmshSTKMeshStruct::loadLegacyMesh ()
{
  std::ifstream ifile;
  open_fname( ifile);

  // Start reading nodes
  std::string line;
  swallow_lines_until( ifile, line, "$NOD");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  // Read the number of nodes
  std::getline (ifile, line);
  NumNodes = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  pts = new double [NumNodes][3];

  // Read the nodes
  int id;
  for (int i=0; i<NumNodes; ++i) {
    ifile >> id >> pts[i][0] >> pts[i][1] >> pts[i][2];
  }

  // Start reading elements (cells and sides)
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$ELM");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);
  int num_entities = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  // Gmsh lists elements and sides (and some points) all toghether, and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D

  int e_type(0);
  for (int i(0); i<num_entities; ++i) {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type;

    increment_element_type( e_type);
  }

  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexas!=0, std::logic_error, "Error! Cannot mix tetrahedra and hexahedra.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_trias*nb_quads!=0, std::logic_error, "Error! Cannot mix triangles and quadrilaterals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexas+nb_trias+nb_quads==0, std::logic_error, "Error! Can only handle 2D and 3D geometries.\n");

  lines = new int*[3];
  tetra = new int*[5];
  trias = new int*[4];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i) {
    tetra[i] = new int[nb_tetra];\
  }
  for (int i(0); i<5; ++i) {
    trias[i] = new int[nb_trias];
  }
  for (int i(0); i<9; ++i) {
    hexas[i] = new int[nb_hexas];
  }
  for (int i(0); i<5; ++i) {
    quads[i] = new int[nb_quads];
  }
  for (int i(0); i<3; ++i) {
    lines[i] = new int[nb_lines];
  }

  if (nb_tetra>0) {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_trias;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  } else if (nb_hexas>0) {
    this->numDim = 3;

    NumElems = nb_hexas;
    NumSides = nb_quads;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  } else if (nb_trias>0) {
    this->numDim = 2;

    NumElems = nb_trias;
    NumSides = nb_lines;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  } else if (nb_quads>0) {
    this->numDim = 2;

    NumElems = nb_quads;
    NumSides = nb_lines;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid mesh dimension.\n");
  }

  // Reset the stream to the beginning of the element section
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$ELM");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found; however, it was found earlier. This may be a bug.\n");
  std::getline(ifile,line); // Skip line with number of elements

  // Read the elements
  int reg_phys, reg_elem, n_nodes;
  int iline(0), itria(0), iquad(0), itetra(0), ihexa(0);
  for (int i(0); i<num_entities; ++i) {
    std::getline(ifile,line);
    std::stringstream ss(line);
    ss >> id >> e_type >> reg_phys >> reg_elem >> n_nodes;
    switch (e_type) {
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

void Albany::GmshSTKMeshStruct::swallow_lines_until( std::ifstream& ifile, std::string& line, std::string line_of_interest)
{
  while (std::getline (ifile, line) && line != line_of_interest) {
    // Keep swallowing lines...
  }

  return;
}

void Albany::GmshSTKMeshStruct::set_NumNodes( std::ifstream& ifile)
{
  std::string line;
  swallow_lines_until( ifile, line, "$Nodes");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  if( version == GmshVersion::V2_2)
  {
    std::getline (ifile, line);
    NumNodes = std::atoi (line.c_str() );
  }
  else if( version == GmshVersion::V4_1)
  {
    int num_entity_blocks = 0;
    int min_node_tag      = 0;
    int max_node_tag      = 0;

    std::getline( ifile, line);
    std::stringstream ss( line);
    ss >> num_entity_blocks >> NumNodes >> min_node_tag >> max_node_tag;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  return;
}

void Albany::GmshSTKMeshStruct::load_node_data( std::ifstream& ifile)
{
  if( version == GmshVersion::V2_2)
  {
    int id = 0;
    for (int i=0; i<NumNodes; ++i) 
    {
      ifile >> id >> pts[i][0] >> pts[i][1] >> pts[i][2];
    }
  }
  else if( version == GmshVersion::V4_1)
  {
    int accounted_nodes = 0;
    while( accounted_nodes < NumNodes)
    {
      int entity_dim     = 0; 
      int entity_tag     = 0;
      int parametric     = 0;
      int num_node_block = 0;
      ifile >> entity_dim >> entity_tag >> parametric >> num_node_block;

      // First get the node id's
      int* node_IDs = new int[num_node_block];
      for( int i = 0; i < num_node_block; i++)
      {
        ifile >> node_IDs[i];
      }

      // Put node coordinates into proper ID place
      for( int i = 0; i < num_node_block; i++)
      {
        // Get this node's unique ID (minus one to index by 0, not 1)
        int this_id = node_IDs[i] -1;
        ifile >> pts[this_id][0] >> pts[this_id][1] >> pts[this_id][2];
        accounted_nodes++;
      }
      delete[] node_IDs;
    }
  }

  return;
}

void Albany::GmshSTKMeshStruct::set_num_entities( std::ifstream& ifile)
{
  std::string line;
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Elements");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);

  if( version == GmshVersion::V2_2)
  {
    num_entities = std::atoi (line.c_str() );
  }
  else if( version == GmshVersion::V4_1)
  {
    int num_entity_blocks = 0;
    int num_elements      = 0;
    int min_elem_tag      = 0;
    int max_elem_tag      = 0;

    std::stringstream iss (line);
    iss >> num_entity_blocks >> num_elements >> min_elem_tag >> max_elem_tag;

    num_entities = num_elements;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  return;
}

void Albany::GmshSTKMeshStruct::increment_element_type( int e_type)
{
  switch (e_type) 
  {
    case 1:  ++nb_lines;  break;
    case 2:  ++nb_trias;  break;
    case 3:  ++nb_quads;  break;
    case 4:  ++nb_tetra;  break;
    case 5:  ++nb_hexas;  break;
    case 8:  ++nb_line3;  break;
    case 9:  ++nb_tri6;   break;
    case 11: ++nb_tet10;  break;
    case 15: /*point*/    break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, 
                                    "Error! Element type (" << e_type << ") not supported.\n");
  }

  return;
}

void Albany::GmshSTKMeshStruct::set_specific_num_of_each_elements( std::ifstream& ifile)
{
  // Gmsh lists elements and sides (and some points) all toghether, 
  // and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to 
  // establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D

  // Reset to begining of msh file, then advance to elements section
  std::string line;
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Elements");
  // Need to start at the second line after '$Elements'
  std::getline (ifile, line);

  if( version == GmshVersion::V2_2)
  {
    for (int i(0); i<num_entities; ++i) 
    {
      int id = 0;
      int e_type = 0;
      std::getline(ifile,line);
      std::stringstream ss(line);
      ss >> id >> e_type;

      increment_element_type( e_type);
    }
  }
  else if( version == GmshVersion::V4_1)
  {
    int accounted_elems = 0;
    while (accounted_elems < num_entities)
    {
      std::getline( ifile, line);

      int entity_dim        = 0;
      int entity_tag        = 0;
      int entity_type       = 0;
      int num_elem_in_block = 0;
      
      std::stringstream iss (line);
      iss >> entity_dim >> entity_tag >> entity_type >> num_elem_in_block;
      for( int i = 0; i < num_elem_in_block; i++)
      {
        increment_element_type( entity_type);
        std::getline( ifile, line);
        accounted_elems++;
      }
    }
  }

  bool is_first_order  = (nb_lines != 0);
  bool is_second_order = (nb_line3 != 0);

  if( is_first_order)
  {
    bool mixed_order_mesh = (nb_line3!=0);
         mixed_order_mesh = (nb_tri6 !=0) || mixed_order_mesh;
         mixed_order_mesh = (nb_tet10!=0) || mixed_order_mesh;

    TEUCHOS_TEST_FOR_EXCEPTION ( mixed_order_mesh, std::logic_error, 
        "Error! Found second order elements in first order mesh.\n");

    TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexas !=0, std::logic_error, 
        "Error! Cannot mix tetrahedra and hexahedra.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (nb_trias*nb_quads!=0, std::logic_error, 
        "Error! Cannot mix triangles and quadrilaterals.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexas+nb_trias+nb_quads==0, std::logic_error, 
        "Error! Can only handle 2D and 3D geometries.\n");
  }
  else if( is_second_order)
  {
    bool mixed_order_mesh = (nb_lines!=0);
         mixed_order_mesh = (nb_trias!=0) || mixed_order_mesh;
         mixed_order_mesh = (nb_tetra!=0) || mixed_order_mesh;

    TEUCHOS_TEST_FOR_EXCEPTION ( mixed_order_mesh, std::logic_error, 
        "Error! Found first order elements in second order mesh.\n");

    bool missing_parts = (nb_line3 == 0);
         missing_parts = (nb_tri6  == 0) || missing_parts;
         missing_parts = (nb_tet10 == 0) || missing_parts;

    TEUCHOS_TEST_FOR_EXCEPTION ( missing_parts, std::logic_error, 
        "Error! This second order mesh is missing secord order parts.\n");
    
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error,
            "Error! Could not determine if mesh was first or second order.\n" <<
            "Checked for number of 2pt lines and 3pt lines, both are non-zero. \n")
  }

  return;
}

void Albany::GmshSTKMeshStruct::size_all_element_pointers()
{
  // First values are the node IDs of the element, then tag
  lines = new int*[3];
  line3 = new int*[4];
  tetra = new int*[5];
  tet10 = new int*[11];
  trias = new int*[4];
  tri6  = new int*[7];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i) 
  {
    tetra[i] = new int[nb_tetra];
  }
  for (int i(0); i<11; ++i) 
  {
    tet10[i] = new int[nb_tet10];
  }
  for (int i(0); i<4; ++i) 
  {
    trias[i] = new int[nb_trias];
  }
  for (int i(0); i<7; ++i) 
  {
    tri6[i] = new int[nb_tri6];
  }
  for (int i(0); i<9; ++i) 
  {
    hexas[i] = new int[nb_hexas];
  }
  for (int i(0); i<5; ++i) 
  {
    quads[i] = new int[nb_quads];
  }
  for (int i(0); i<3; ++i) 
  {
    lines[i] = new int[nb_lines];
  }
  for( int i(0); i<4; ++i)
  {
    line3[i] = new int[nb_line3];
  }

  return;
}

void Albany::GmshSTKMeshStruct::set_generic_mesh_info()
{
  if (nb_tetra>0) 
  {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_trias;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  } 
  else if (nb_tet10>0) 
  {
    this->numDim = 3;

    NumElems = nb_tet10;
    NumSides = nb_tri6;
    NumElemNodes = 10;
    NumSideNodes = 6;
    elems = tet10;
    sides = tri6;
  } 
  else if (nb_hexas>0) 
  {
    this->numDim = 3;

    NumElems = nb_hexas;
    NumSides = nb_quads;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  } 
  else if (nb_trias>0) 
  {
    this->numDim = 2;

    NumElems = nb_trias;
    NumSides = nb_lines;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  } 
  else if (nb_tri6>0) 
  {
    this->numDim = 2;

    NumElems = nb_tri6;
    NumSides = nb_line3;
    NumElemNodes = 6;
    NumSideNodes = 3;
    elems = tri6;
    sides = line3;
  } 
  else if (nb_quads>0) 
  {
    this->numDim = 2;

    NumElems = nb_quads;
    NumSides = nb_lines;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  } 
  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid mesh dimension.\n");
  }
  return;
}

void Albany::GmshSTKMeshStruct::store_element_info( 
      int  e_type,
      int& iline,
      int& iline3,
      int& itria,
      int& itri6,
      int& iquad,
      int& itetra,
      int& itet10,
      int& ihexa,
      std::vector<int>& tags,
      std::stringstream& ss)
{
  switch (e_type) 
  {
    case 1: // 2-pt Line
      ss >> lines[0][iline] >> lines[1][iline];
      lines[2][iline] = tags[0];
      ++iline;
      break;
    case 2: // 3-pt Triangle
      ss >> trias[0][itria] >> trias[1][itria] >> trias[2][itria];
      trias[3][itria] = tags[0];
      ++itria;
      break;
    case 3: // 4-pt Quad
      ss >> quads[0][iquad] >> quads[1][iquad] >> quads[2][iquad] >> quads[3][iquad];
      quads[4][iquad] = tags[0];
      ++iquad;
      break;
    case 4: // 4-pt Tetra
      ss >> tetra[0][itetra] >> tetra[1][itetra] >> tetra[2][itetra] >> tetra[3][itetra];
      tetra[4][itetra] = tags[0];
      ++itetra;
      break;
    case 5: // 8-pt Hexa
      ss >> hexas[0][ihexa] >> hexas[1][ihexa] >> hexas[2][ihexa] >> hexas[3][ihexa]
         >> hexas[4][ihexa] >> hexas[5][ihexa] >> hexas[6][ihexa] >> hexas[7][ihexa];
      hexas[8][ihexa] = tags[0];
      ++ihexa;
      break;
    case 8: // 3-pt Line
      ss >> line3[0][iline3] >> line3[1][iline3] >> line3[2][iline3];
      line3[3][iline3] = tags[0];
      iline3++;
      break;
    case 9: // 6-pt Triangle
      ss >> tri6[0][itri6] >> tri6[1][itri6] >> tri6[2][itri6]
         >> tri6[3][itri6] >> tri6[4][itri6] >> tri6[5][itri6];
      tri6[6][itri6] = tags[0];
      itri6++;
      break;
    case 11: // 10-pt Tetra
      // NOTE!
      // The node ordering between gmsh and STK for tet10 is the same 
      // EXCEPT for the last two. I.e., nodes 8 and 9 are switched!
      ss >> tet10[0][itet10] >> tet10[1][itet10] >> tet10[2][itet10] >> tet10[3][itet10] >> tet10[4][itet10] 
         >> tet10[5][itet10] >> tet10[6][itet10] >> tet10[7][itet10] >> tet10[9][itet10] >> tet10[8][itet10];
      tet10[10][itet10] = tags[0];
      itet10++;
      break;
    case 15: // Point
        break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported; but you should have got an error before!\n");
  }

  return;
}

void Albany::GmshSTKMeshStruct::load_element_data( std::ifstream& ifile)
{
  // Reset the stream to the beginning of the element section
  std::string line;
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Elements");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found; however, it was found earlier. This may be a bug.\n");

  // Skip line with number of elements
  std::getline(ifile,line);

  // Read the elements
  int iline(0), iline3(0), 
      itria(0), itri6(0), 
      iquad(0), 
      itetra(0), itet10(0), 
      ihexa(0), 
      n_tags(0), id(0), e_type(0);

  if( version == GmshVersion::V2_2)
  {
    std::vector<int> tags;
    for (int i(0); i<num_entities; ++i) 
    {
      std::getline(ifile,line);
      std::stringstream ss(line);
      ss >> id >> e_type >> n_tags;
      TEUCHOS_TEST_FOR_EXCEPTION (n_tags<=0, Teuchos::Exceptions::InvalidParameter, "Error! Number of tags must be positive.\n");
      tags.resize(n_tags+1);
      for (int j(0); j<n_tags; ++j) 
      {
        ss >> tags[j];
      }
      tags[n_tags] = 0;

      store_element_info( e_type, iline, iline3, itria, itri6, iquad, itetra, itet10, ihexa, tags, ss);
    }
  }
  else if( version == GmshVersion::V4_1)
  {
    int accounted_elems = 0;
    std::vector<int> tags;
    while (accounted_elems < num_entities)
    {
      std::getline( ifile, line);

      int entity_dim        = 0;
      int entity_tag        = 0;
      int entity_type       = 0;
      int num_elem_in_block = 0;
      
      std::stringstream iss (line);
      iss >> entity_dim >> entity_tag >> entity_type >> num_elem_in_block;
      tags.push_back( entity_tag);
      for( int i = 0; i < num_elem_in_block; i++)
      {
        std::getline( ifile, line);
        std::stringstream ss (line);
        int elem_id = 0;
        ss >> elem_id;

        int e_type = entity_type;
        store_element_info( e_type, iline, iline3, itria, itri6, iquad, itetra, itet10, ihexa, tags, ss);
        accounted_elems++;
      }
      tags.clear();
    }
  }

  return;
}

void Albany::GmshSTKMeshStruct::loadAsciiMesh ()
{
  std::ifstream ifile;
  open_fname( ifile);

  // Start reading nodes
  set_NumNodes( ifile);
  pts = new double [NumNodes][3];
  load_node_data( ifile);

  // Start reading elements (cells and sides)
  set_num_entities( ifile);
  set_specific_num_of_each_elements( ifile);

  // Size the element pointers to handle the element information
  size_all_element_pointers();
  set_generic_mesh_info();
  
  // Populate the element pointers with tag and node info
  load_element_data( ifile);

  // Close the input stream
  ifile.close();
}

void Albany::GmshSTKMeshStruct::create_element_block()
{
  std::string ebn = "Element Block 0";
  partVec[0] = &metaData->declare_part(ebn, stk::topology::ELEMENT_RANK);
  ebNameToIndex[ebn] = 0;

  return;
}

void Albany::GmshSTKMeshStruct::loadBinaryMesh ()
{
  std::ifstream ifile;
  open_fname( ifile);

  std::string line;
  std::getline (ifile, line); // $MeshFormat
  std::getline (ifile, line); // 2.0 file-type data-size

  // Check file endianness
  union {
    int i;
    char c[sizeof (int)];
  } one;

  ifile.read (one.c, sizeof (int) );
  TEUCHOS_TEST_FOR_EXCEPTION (one.i!=1, std::runtime_error, "Error! Uncompatible binary format.\n");

  // Start reading nodes
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Nodes");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Nodes section not found.\n");

  // Read the number of nodes
  std::getline (ifile, line);
  NumNodes = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (NumNodes<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of nodes.\n");
  pts = new double[NumNodes][3];

  // Read the nodes
  int id;
  for (int i=0; i<NumNodes; ++i)  {
    ifile.read (reinterpret_cast<char*> (&id), sizeof (int) );
    ifile.read (reinterpret_cast<char*> (pts[i]), 3 * sizeof (double) );
  }

  // Start reading elements (cells and sides)
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Elements");
  TEUCHOS_TEST_FOR_EXCEPTION (ifile.eof(), std::runtime_error, "Error! Element section not found.\n");

  // Read the number of entities
  std::getline (ifile, line);
  int num_entities = std::atoi (line.c_str() );
  TEUCHOS_TEST_FOR_EXCEPTION (num_entities<=0, Teuchos::Exceptions::InvalidParameter, "Error! Invalid number of mesh elements.\n");

  // Gmsh lists elements and sides (and some points) all toghether, and does not specify beforehand what kind of elements
  // the mesh has. Hence, we need to scan the entity list once to establish what kind of elements we have. We support
  // linear Tetrahedra/Hexahedra in 3D and linear Triangle/Quads in 2D
  std::vector<int> tmp;
  int n_tags(0), e_type(0), entities_found(0);
  while (entities_found<num_entities) {
    int header[3];
    ifile.read(reinterpret_cast<char*> (header), 3*sizeof(int));

    TEUCHOS_TEST_FOR_EXCEPTION (header[1]<=0, std::logic_error, "Error! Invalid number of elements of this type.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (header[2]<=0, std::logic_error, "Error! Invalid number of tags.\n");

    e_type = header[0];
    entities_found += header[1];
    n_tags = header[2];

    int length;
    switch (e_type) {
      case 1: // 2-pt Line
        length = 1+n_tags+2; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_lines += header[1];
        break;
      case 2: // 3-pt Triangle
        length = 1+n_tags+3; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_trias += header[1];
        break;
      case 3: // 4-pt Quad
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quads += header[1];
        break;
      case 4: // 4-pt Tetra
        length = 1+n_tags+4; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quads += header[1];
        break;
      case 5: // 8-pt Hexa
        length = 1+n_tags+8; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        nb_quads += header[1];
        break;
      case 15: // Point
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Element type not supported.\n");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra*nb_hexas!=0, std::logic_error, "Error! Cannot mix tetrahedra and hexahedra.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_trias*nb_quads!=0, std::logic_error, "Error! Cannot mix triangles and quadrilaterals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (nb_tetra+nb_hexas+nb_trias+nb_quads==0, std::logic_error, "Error! Can only handle 2D and 3D geometries.\n");

  lines = new int*[3];
  tetra = new int*[5];
  trias = new int*[4];
  hexas = new int*[9];
  quads = new int*[5];
  for (int i(0); i<5; ++i) {
    tetra[i] = new int[nb_tetra];
  }
  for (int i(0); i<5; ++i) {
    trias[i] = new int[nb_trias];
  }
  for (int i(0); i<9; ++i) {
    hexas[i] = new int[nb_hexas];
  }
  for (int i(0); i<5; ++i) {
    quads[i] = new int[nb_quads];
  }
  for (int i(0); i<3; ++i) {
    lines[i] = new int[nb_lines];
  }

  if (nb_tetra>0) {
    this->numDim = 3;

    NumElems = nb_tetra;
    NumSides = nb_trias;
    NumElemNodes = 4;
    NumSideNodes = 3;
    elems = tetra;
    sides = trias;
  } else if (nb_hexas>0) {
    this->numDim = 3;

    NumElems = nb_hexas;
    NumSides = nb_quads;
    NumElemNodes = 8;
    NumSideNodes = 4;
    elems = hexas;
    sides = quads;
  } else if (nb_trias>0) {
    this->numDim = 2;

    NumElems = nb_trias;
    NumSides = nb_lines;
    NumElemNodes = 3;
    NumSideNodes = 2;
    elems = trias;
    sides = lines;
  } else if (nb_quads>0) {
    this->numDim = 2;

    NumElems = nb_quads;
    NumSides = nb_lines;
    NumElemNodes = 4;
    NumSideNodes = 2;
    elems = quads;
    sides = lines;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid mesh dimension.\n");
  }

  // Reset the stream to the beginning of the element section
  ifile.seekg (0, std::ios::beg);
  swallow_lines_until( ifile, line, "$Elements");
  std::getline(ifile,line); // Skip line with number of elements

  entities_found = 0;
  int iline(0), itria(0), iquad(0), itetra(0), ihexa(0);
  while (entities_found<num_entities) {
    int header[3];
    ifile.read(reinterpret_cast<char*> (header), 3*sizeof(int));

    TEUCHOS_TEST_FOR_EXCEPTION (header[1]<=0, std::logic_error, "Error! Invalid number of elements of this type.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (header[2]<=0, std::logic_error, "Error! Invalid number of tags.\n");

    e_type = header[0];
    entities_found += header[1];
    n_tags = header[2];

    int length;
    switch (e_type) {
      case 1: // 2-pt Line
        length = 1+n_tags+2; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++iline) {
          lines[0][j] = tmp[sizeof(int)*(j*length+1+n_tags)];   // First pt
          lines[1][j] = tmp[sizeof(int)*(j*length+1+n_tags+1)];
          lines[2][j] = tmp[sizeof(int)*(j*length+1)];          // Use first tag
        }
        break;
      case 2: // 3-pt Triangle
        length = 1+n_tags+3; // id, tags, points
        tmp.resize(header[1]*length);
        ifile.read (reinterpret_cast<char*> (&tmp[0]), header[1]*length*sizeof(int));
        for (int j(0); j<header[1]; ++j, ++itria) {
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
        for (int j(0); j<header[1]; ++j, ++iquad) {
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
        for (int j(0); j<header[1]; ++j, ++itetra) {
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
        for (int j(0); j<header[1]; ++j, ++ihexa) {
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

void Albany::GmshSTKMeshStruct::set_all_nodes_boundary( std::vector<std::string>& nsNames)
{
  std::string nsn = "Node";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  return;
}

void Albany::GmshSTKMeshStruct::set_all_sides_boundary( std::vector<std::string>& ssNames)
{
  std::string ssn = "BoundarySide";
  ssNames.push_back(ssn);
  ssPartVec[ssn] = &metaData->declare_part(ssn, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssn]);
  stk::io::put_io_part_attribute(metaData->universal_part());
#endif

  return;
}

void Albany::GmshSTKMeshStruct::set_boundaries( const Teuchos::RCP<const Teuchos_Comm>& commT,
                                                std::vector<std::string>&  ssNames,
                                                std::vector<std::string>&  nsNames)
{
  set_all_nodes_boundary( nsNames);
  set_all_sides_boundary( ssNames);

  // Counting boundaries (only proc 0 has any stored, so far)
  std::set<int> bdTags;
  for (int i(0); i<NumSides; ++i) 
  {
    bdTags.insert(sides[NumSideNodes][i]);
  }

  // Broadcasting the tags
  int numBdTags = bdTags.size();
  Teuchos::broadcast<LO,LO>(*commT, 0, 1, &numBdTags);
  int* bdTagsArray = new int[numBdTags];
  std::set<int>::iterator it=bdTags.begin();
  for (int k=0; it!=bdTags.end(); ++it,++k) 
  {
    bdTagsArray[k] = *it;
  }
  Teuchos::broadcast<LO,LO>(*commT, 0, numBdTags, bdTagsArray);

  // Adding boundary nodesets and sidesets separating different labels
  for (int k=0; k<numBdTags; ++k) 
  {
    int tag = bdTagsArray[k];
    std::stringstream ss;
    ss << tag;
    std::string name = ss.str();

    add_nodeset( name, tag, nsNames);
    add_sideset( name, tag, ssNames);

  }
  delete[] bdTagsArray;

  // Gmsh 4.1 Allows users to give string names to surface.
  // We overwrite the number based set names with the string ones
  // if any exist
  if( version == GmshVersion::V4_1)
  {
    // Map has format: "name",  physical_tag
    std::map<std::string, int> physical_names; 
    get_physical_names( physical_names, commT);

    std::map< std::string, int>::iterator it;
    for( it = physical_names.begin(); it != physical_names.end(); it++)
    {
      std::string name = it->first;
      int         tag  = it->second;

      add_nodeset( name, tag, nsNames);
      add_sideset( name, tag, ssNames);

    }
  }

  return;
}

void Albany::GmshSTKMeshStruct::add_sideset( std::string sideset_name, int tag, std::vector<std::string>& ssNames)
{
  std::stringstream ssn_i;
  ssn_i << "BoundarySideSet" << sideset_name;

  bdTagToSideSetName[tag] = ssn_i.str();
  ssNames.push_back(ssn_i.str());

  ssPartVec[ssn_i.str()] = &metaData->declare_part(ssn_i.str(), metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssn_i.str()]);
#endif

  return;
}


void Albany::GmshSTKMeshStruct::add_nodeset( std::string nodeset_name, int tag, std::vector<std::string>& nsNames)
{
  std::stringstream nsn_i;
  nsn_i << "BoundaryNodeSet" << nodeset_name;

  bdTagToNodeSetName[tag] = nsn_i.str();
  nsNames.push_back(nsn_i.str());

  nsPartVec[nsn_i.str()] = &metaData->declare_part(nsn_i.str(), stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn_i.str()]);
#endif

  return;
}

void Albany::GmshSTKMeshStruct::set_allowable_gmsh_versions()
{
  allowable_gmsh_versions.insert( 2.2);
  allowable_gmsh_versions.insert( 4.1);

  return;
}

bool Albany::GmshSTKMeshStruct::set_version_enum_from_float()
{
  bool can_read = true;
  if( version_in == (float)2.2 )
  {
    version = GmshVersion::V2_2;
  }
  else if( version_in == (float)4.1 )
  {
    version = GmshVersion::V4_1;
  }
  else
  {
    can_read = false;
  }

  return can_read;
}

void Albany::GmshSTKMeshStruct::check_version( std::ifstream& ifile)
{
  // Tell user what gmsh version we're reading
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  *out << "The gmsh version is: "
       << version_in
       << std::endl;

  // Check if we know how to read this gmsh version.
  // Return an error if we do not.
  if( !set_version_enum_from_float() )
  {
    *out << "Allowable gmsh *.msh file versions are: " << std::endl;
    set_allowable_gmsh_versions();
    for( std::set<float>::iterator it = allowable_gmsh_versions.begin(); it != allowable_gmsh_versions.end(); it++)
    {
      *out << *it << std::endl;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Cannot read this version of gmsh *.msh file!");
  }

  return;
}

void Albany::GmshSTKMeshStruct::open_fname( std::ifstream& ifile)
{
  ifile.open(fname.c_str());
  if (!ifile.is_open()) 
  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
  }
  
  return;
}

void Albany::GmshSTKMeshStruct::get_name_for_physical_names( std::string& name, std::ifstream& ifile)
{
  std::string line;
  int         id;
  int         dim;
  
  std::getline( ifile, line);
  std::stringstream ss (line);
  ss >> dim >> id >> name;

  // If this entity has a name, then assign it.
  // Use the id otherwise.
  if( name.empty() )
  {
    std::stringstream ss;
    ss << id;
    name = ss.str();
  }
  else
  {
    // Need to remove quote marks from name 
    // and prepend with underscore
    name.erase( std::remove(name.begin(), name.end(), '"'), name.end());
    name = "_" + name;
  }

  return;
}

void Albany::GmshSTKMeshStruct::get_physical_tag_to_surface_tag_map( 
      std::ifstream&      ifile, 
      std::map<int, int>& physical_surface_tags,
      int                 num_surfaces)
{
  int    surface_tag         = 0;
  double min_x               = 0.0;
  double min_y               = 0.0;
  double min_z               = 0.0;
  double max_x               = 0.0;
  double max_y               = 0.0;
  double max_z               = 0.0;
  int    num_physical_tags   = 0;
  int    physical_tag        = 0;
  int    num_bounding_curves = 0;
  int    curve_tag           = 0;

  std::string line;
  for( int i = 0; i < num_surfaces; i++)
  {
    std::getline( ifile, line);
    std::stringstream ss (line);
    ss >> surface_tag
       >> min_x
       >> min_y
       >> min_z
       >> max_x
       >> max_y
       >> max_z
       >> num_physical_tags  
       >> physical_tag       
       >> num_bounding_curves
       >> curve_tag;

    TEUCHOS_TEST_FOR_EXCEPTION ( num_physical_tags > 1, std::runtime_error, 
                                "Cannot support more than one physical tag per surface.\n");

    TEUCHOS_TEST_FOR_EXCEPTION ( num_physical_tags < 0, std::runtime_error, 
                                "Cannot have a negative number of physical tags per surface.\n");

    if( num_physical_tags == 1)
    {
      physical_surface_tags.insert( std::make_pair( physical_tag, surface_tag));
    }
  }

  return;
}
                                                             

void Albany::GmshSTKMeshStruct::read_physical_names_from_file( std::map<std::string, int>& physical_names)
{
  std::ifstream ifile;
  open_fname( ifile);

  // Advance to the PhysicalNames section
  std::string line;
  swallow_lines_until( ifile, line, "$PhysicalNames");
  if( ifile.peek() != EOF)
  {
    // Get number of Physical Names
    int num_physical_names = 0;
    std::getline( ifile, line);
    std::stringstream ss (line);
    ss >> num_physical_names;

    // Get the list of physical names
    std::vector< std::string> names;
    for( size_t i = 0; i < num_physical_names; i++)
    {
      std::string name;
      get_name_for_physical_names( name, ifile);
      names.push_back( name);
    }

    // Advance to Surface Entities section
    ifile.seekg (0, std::ios::beg);
    swallow_lines_until( ifile, line, "$Entities");

    // Get number of each entity type
    int num_points   = 0;
    int num_curves   = 0;
    int num_surfaces = 0;
    int num_volumes  = 0;
    std::getline( ifile, line);
    std::stringstream iss (line);
    iss >> num_points >> num_curves >> num_surfaces >> num_volumes;

    // Skip to the surfaces
    int num_lines_to_skip = num_points + num_curves;
    for( int i = 0; i < num_lines_to_skip; i++)
    { 
      std::getline( ifile, line);
    }
    std::map< int, int> physical_surface_tags;
    get_physical_tag_to_surface_tag_map( ifile, physical_surface_tags, num_surfaces);

    std::stringstream error_msg;
    error_msg << "Cannot support more than one physical tag per surface \n"
              << "(but you should have gotten an error before this!)    \n"
              << "physical_surface_tags.size() = " << physical_surface_tags.size() << ". \n"
              << "names.size() = " << names.size() << ". \n";
    TEUCHOS_TEST_FOR_EXCEPTION ( physical_surface_tags.size() != names.size(), std::runtime_error, error_msg.str());

    // Add each physical name pair to the map
    for( int i = 0; i < names.size(); i++)
    {
      std::string name = names[i];
      // Index by i+1 since gmsh starts counting at 1 and not 0
      int surface_tag  = physical_surface_tags[i+1];

      physical_names.insert( std::make_pair( name, surface_tag));
    }

  }
  ifile.close();

  return;
}

void Albany::GmshSTKMeshStruct::broadcast_name_tag_pair( std::vector< std::string>               names,
                                                         int*                                    tags_array,
                                                         int                                     pair_number,
                                                         const Teuchos::RCP<const Teuchos_Comm>& commT,
                                                         std::map< std::string, int>&            physical_names)
{
  std::string name;
  if( commT->getRank() == 0) 
  {
    name = names[pair_number];

    int strsize = name.size();
    Teuchos::broadcast<int, int>( *commT, 0, &strsize);

    char* ptr = (strsize) ? (&name[0]) : 0;
    Teuchos::broadcast<int, char>( *commT, 0, strsize, ptr);
  }
  else 
  {
    int strsize;
    Teuchos::broadcast<int, int>( *commT, 0, &strsize);

    name.resize( strsize);
    char* ptr = (strsize) ? (&name[0]) : 0;
    Teuchos::broadcast<int, char>( *commT, 0, strsize, ptr);
  }

  int tag = tags_array[pair_number];
  physical_names.insert( std::make_pair( name, tag));

  return;
}
                                                         

void Albany::GmshSTKMeshStruct::broadcast_physical_names( std::map<std::string, int>&             physical_names,
                                                          const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  // Broadcast the number of name-tag pairs
  int num_pairs = physical_names.size();
  Teuchos::broadcast(*commT, 0, 1, &num_pairs);

  // First unpack the names and tags from the map.
  // Only proc 0 will be doing anything here. 
  // Maps on other procs will be empty.
  std::vector< std::string> names;
  int* tags_array = new int[num_pairs];

  std::map< std::string, int>::iterator it;
  int counter = 0;
  for( it = physical_names.begin(); it != physical_names.end(); it++)
  {
    names.push_back( it->first);
    tags_array[counter] = it->second;
    counter++;
  }

  // Clear out the map to rebuild it together.
  physical_names.clear();

  // Broadcast names and tags
  Teuchos::broadcast<LO,LO>(*commT, 0, num_pairs, tags_array);
  for( int i = 0; i < num_pairs; i++)
  {
    broadcast_name_tag_pair( names, tags_array, i, commT, physical_names);
  }

  delete[] tags_array;
  return;
}

void Albany::GmshSTKMeshStruct::get_physical_names( std::map<std::string, int>&             physical_names,
                                                    const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  if( commT->getRank() == 0 )
  {
    read_physical_names_from_file( physical_names);
  }
  broadcast_physical_names( physical_names, commT);

  return;
}
