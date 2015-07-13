//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>
#include "Teuchos_VerboseObject.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include <Shards_BasicTopologies.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <Albany_STKNodeSharing.hpp>

#include <stk_mesh/base/FEMHelpers.hpp>
#include "Albany_Utils.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#ifdef ALBANY_64BIT_INT
// long int == 64bit
#  define ST_LLU "%ld"
#else
#  define ST_LLU "%d"
#endif

// Test 64 bit Tpetra address space
const GO StartIndex = 0;
//const GO StartIndex = 2147483647L; // 2^31 - 1

template<unsigned Dim, class traits>
Albany::TmplSTKMeshStruct<Dim, traits>::TmplSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
                  const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct(params, adaptParams_, traits_type::size),
  periodic_x(params->get("Periodic_x BC", false)),
  periodic_y(params->get("Periodic_y BC", false)),
  periodic_z(params->get("Periodic_z BC", false)),
  triangles(false)
{

/*
  There are two use cases of interest here.

  1. Do not specify element block information ("Element Blocks"). In this case, numEB = 1 and
      the scale (domain dimensions) and discretization is set globally across the mesh.

  2. "Element Blocks" is specified in the input file. In this case, the logical size of the element blocks
      must be specified, as well as the number of elements in each block and their size (block length).

*/

  GO total_elems;

  int input_nEB = params->get<int>("Element Blocks", -1); // Read the number of element blocks. -1 if not listed

  if(input_nEB <= 0 || Dim == 0) // If "Element Blocks" are not present in input file

    numEB = 1;

  else

    numEB = input_nEB;

  params->validateParameters(*this->getValidDiscretizationParameters(), 0,
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  nelem[0] = 1; // One element in the case of a single point 0D mesh (the below isn't executed)
  scale[0] = 1.0; // one unit length in the x dimension

  // The number of element blocks. Always at least one.

  EBSpecs.resize(numEB);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

  for(unsigned i = 0; i < Dim; i++){ // Get the number of elements in each dimension from params
                                // Note that nelem will default to 0 and scale to 1 if element
                                // blocks are specified

#pragma clang diagnostic pop

    // Read the values for "1D Elements", "2D Elements", "3D Elements"
     std::stringstream buf;
     buf << i + 1 << "D Elements";
     nelem[i] = params->get<int>(buf.str(), 0);

    // Read the values for "1D Scale", "2D Scale", "3D Scale"
     std::stringstream buf2;
     buf2 << i + 1 << "D Scale";

     scale[i] = params->get<double>(buf2.str(),     1.0);
  }

  if(input_nEB <= 0 || Dim == 0){ // If "Element Blocks" are not present in input file

    EBSpecs[0].Initialize(nelem, scale);

    // Format and print out information about the mesh that is being generated

    if (commT->getRank()==0 && Dim > 0){ // Not reached for 0D problems

     std::cout <<"TmplSTKMeshStruct:: Creating " << Dim << "D mesh of size ";

     std::stringstream nelem_txt, scale_txt;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

     for(unsigned idx=0; idx < Dim - 1; idx++){

#pragma clang diagnostic pop

       nelem_txt << nelem[idx] << "x";
       scale_txt << scale[idx] << "x";

     }

     nelem_txt << nelem[Dim - 1];
     scale_txt << scale[Dim - 1];

     std::cout << nelem_txt.str() << " elements and scaled to " <<
                 scale_txt.str() << std::endl;

     if (triangles)
       std::cout<<" Quad elements cut to make twice as many triangles " <<std::endl;

   }

   // Calculate total number of elements
   total_elems = nelem[0];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

   for(unsigned i = 1; i < Dim; i++)
      total_elems *= nelem[i];

#pragma clang diagnostic pop

  }
  else { // Element blocks are present in input

    std::vector<GO> min(Dim), max(Dim);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

    for(unsigned i = 0; i < Dim; i++){

#pragma clang diagnostic pop

      min[i] = std::numeric_limits<GO>::max();
      max[i] = std::numeric_limits<GO>::min();

      nelem[i] = 0;

    }

    // Read the EB extents from the parameter list and initialize the EB structs
    for(unsigned int eb = 0; eb < numEB; eb++){

      EBSpecs[eb].Initialize(eb, params);

//      for(int i = 0; i < Dim; i++)

//        nelem[i] += EBSpecs[eb].numElems(i);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

      for(unsigned i = 0; i < Dim; i++){

#pragma clang diagnostic pop


        min[i] = (min[i] < EBSpecs[eb].min[i]) ? min[i] : EBSpecs[eb].min[i];
        max[i] = (max[i] > EBSpecs[eb].max[i]) ? max[i] : EBSpecs[eb].max[i];

      }

    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

    for(unsigned i = 0; i < Dim; i++)

#pragma clang diagnostic pop

      nelem[i] = max[i] - min[i];

    // Calculate total number of elements
    total_elems = nelem[0];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

    for(unsigned i = 1; i < Dim; i++)

#pragma clang diagnostic pop

      total_elems *= nelem[i];

  }

  std::vector<std::string> nsNames;

  // Construct the nodeset names

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"

  for(unsigned idx=0; idx < Dim*2; idx++){ // 2 nodesets per dimension (one at beginning, one at end)
    std::stringstream buf;
    buf << "NodeSet" << idx;
    nsNames.push_back(buf.str());
  }
  // For 2D and 3D, and extra node set of a single node for setting Pressure in confined incompressible flows
  if (Dim==2 || Dim==3) nsNames.push_back("NodeSet99");

  std::vector<std::string> ssNames;

  // Construct the sideset names

  if(Dim > 1 ) // Sidesets present only for 2 and 3D problems

    for(unsigned idx=0; idx < Dim*2; idx++){ // 2 sidesets per dimension (one at beginning, one at end)
      std::stringstream buf;
      buf << "SideSet" << idx;
      ssNames.push_back(buf.str());
    }

  DeclareParts(EBSpecs, ssNames, nsNames);

  // Different element topologies. Note that there is only a choice for triangles/quads in 2D
  // (all other dimensions select the default element type for now)

  std::string cellTopo = params->get("Cell Topology", "Quad");
  if (cellTopo == "Tri" || cellTopo == "Triangle")
    triangles = true;
  // else TEST_FOR_EXCEPTION (cellTopo != "Quad", std::logic_error,
  //    "\nUnknown Cell Topology entry in STK2D(not \'Tri\' or \'Quad\'): "
  //     << cellTopo);

  // Set the element types in the EBs

  //get the type of transformation of STK mesh (for FELIX/Aeras problems)
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh (for FELIX problems)
  felixAlpha = params->get("FELIX alpha", 0.0);
  felixL = params->get("FELIX L", 1.0);

  points_per_edge = params->get("Points Per Edge", 2); //get # of nodes per edge for Aeras::SpectralDiscretization (Aeras problems)

  //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);

  //Does user want to write coordinates to matrix market file (e.g., for ML analysis)?
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false);

  for(unsigned int i = 0; i < numEB; i++){

    if (triangles)
      stk::mesh::set_cell_topology<optional_element_type>(*partVec[i]);
    else
      stk::mesh::set_cell_topology<default_element_type>(*partVec[i]);
  }

  for(std::map<std::string, stk::mesh::Part*>::const_iterator it = ssPartVec.begin();
    it != ssPartVec.end(); ++it)

    stk::mesh::set_cell_topology<default_element_side_type>(*it->second);

  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);

  // Create just enough of the mesh to figure out number of owned elements
  // so that the problem setup can know the worksetSize

  // Distribute the elems equally. Build total_elems elements, with nodeIDs starting at StartIndex
  elem_map = Teuchos::rcp(new Tpetra_Map(total_elems, StartIndex, commT, Tpetra::GloballyDistributed));

  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->getNodeNumElements() * (triangles ? 2 : 1));

  // Build a map to get the EB name given the index

  for (unsigned int eb=0; eb<numEB; eb++)

    ebNameToIndex[partVec[eb]->name()] = eb;

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {

    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                               nsNames, ssNames, worksetSize, partVec[0]->name(),
                               ebNameToIndex, this->interleavedOrdering));
  }
  else {

    meshSpecs.resize(numEB);

    this->allElementBlocksHaveSamePhysics=false;

    for (unsigned int eb=0; eb<numEB; eb++) {

      // MeshSpecs holds all info needed to set up an Albany problem

      const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();

      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                                nsNames, ssNames, worksetSize, partVec[eb]->name(),
                                ebNameToIndex, this->interleavedOrdering, true));
    }
 }
}

template<unsigned Dim, class traits>
void
Albany::TmplSTKMeshStruct<Dim, traits>::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& /*side_set_sis*/)
{

  // Create global mesh: Dim-D structured, rectangular

//  std::vector<std::vector<double> > h_dim;
  std::vector<double> h_dim[traits_type::size];
//  h_dim.resize(traits_type::size);
//  x.resize(traits_type::size);
//  x.resize(Dim);

  for(unsigned idx=0; idx < Dim; idx++){

    // Allocate the storage

    x[idx].resize(nelem[idx] + 1);
    h_dim[idx].resize(nelem[idx] + 1);

  }

  for(unsigned int eb = 0; eb < numEB; eb++)

    EBSpecs[eb].calcElemSizes(h_dim);

  for(unsigned idx=0; idx < Dim; idx++){

    x[idx][0] = 0;

    for(unsigned int i=1; i <= nelem[idx]; i++)

      x[idx][i] = x[idx][i - 1] + h_dim[idx][i - 1]; // place the coordinates of the element nodes


  }

  SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh

  buildMesh(commT);

  // STK
  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();

#ifdef ALBANY_FELIX
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);

  std::vector<stk::mesh::Entity> elems;
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK), elems);

  GO numOwnedNodes(0);
  GO numOwnedElems(0);
  numOwnedNodes = stk::mesh::count_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::NODE_RANK));
  numOwnedElems = stk::mesh::count_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK));

  GO numGlobalVertices = 0;
  GO numGlobalElements = 0;
  Teuchos::reduceAll<int, GO>(*commT, Teuchos::REDUCE_SUM, 1, &numOwnedNodes, &numGlobalVertices);
  Teuchos::reduceAll<int, GO>(*commT, Teuchos::REDUCE_SUM, 1, &numOwnedElems, &numGlobalElements);

  if (commT->getRank() == 0)
  {
    *out << "Checking if requirements are already stored in the mesh. If not, we import them from ascii files.\n";
  }

  Teuchos::Array<GO> nodeIndices(nodes.size()), elemIndices(elems.size());
  for (int i = 0; i < nodes.size(); ++i)
    nodeIndices[i] = bulkData->identifier(nodes[i]) - 1;
  for (int i = 0; i < elems.size(); ++i)
    elemIndices[i] = bulkData->identifier(elems[i]) - 1;


  // Creating the serial and parallel node maps
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

  Teuchos::RCP<const Tpetra_Map> nodes_map = Tpetra::createNonContigMapWithNode<LO, GO> (nodeIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());
  Teuchos::RCP<const Tpetra_Map> elems_map = Tpetra::createNonContigMapWithNode<LO, GO> (elemIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());

  int numMyNodes = (commT->getRank() == 0) ? numGlobalVertices : 0;
  int numMyElements = (commT->getRank() == 0) ? numGlobalElements : 0;
  Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyNodes, 0, commT));
  Teuchos::RCP<const Tpetra_Map> serial_elems_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, commT));

  // Creating the Tpetra_Import object (to transfer from serial to parallel vectors)
  Tpetra_Import importOperatorNode (serial_nodes_map, nodes_map);
  Tpetra_Import importOperatorElem (serial_elems_map, elems_map);

  Teuchos::ParameterList* req_fields_info;
  if (params->isSublist("Required Fields Info"))
  {
    req_fields_info = &params->sublist("Required Fields Info");

    for (AbstractFieldContainer::FieldContainerRequirements::const_iterator it=req.begin(); it!=req.end(); ++it)
    {
      // Get the file name
      std::string temp_str = *it + " File Name";
      std::string fname = req_fields_info->get<std::string>(temp_str,"");

      // Ge the file type (if not specified, assume Scalar)
      temp_str = *it + " Field Type";
      std::string ftype = req_fields_info->get<std::string>(temp_str,"");
      if (ftype=="")
      {
        *out << "Warning! No field type specified for field " << *it << ". We skip it and hope is already present in the mesh...\n";
        continue;
      }

      stk::mesh::Entity node, elem;
      stk::mesh::EntityId nodeId, elemId;
      int lid;
      double* values;

      typedef AbstractSTKFieldContainer::QPScalarFieldType  QPScalarFieldType;
      typedef AbstractSTKFieldContainer::QPVectorFieldType  QPVectorFieldType;
      typedef AbstractSTKFieldContainer::ScalarFieldType    ScalarFieldType;
      typedef AbstractSTKFieldContainer::VectorFieldType    VectorFieldType;

      // Depending on the field type, we need to use different pointers
      if (ftype == "Node Scalar")
      {
        // Creating the serial and (possibly) parallel Tpetra service vectors
        Tpetra_Vector serial_req_vec(serial_nodes_map);
        Tpetra_Vector req_vec(nodes_map);

        temp_str = *it + " Value";
        if (req_fields_info->isParameter(temp_str))
        {
          *out << "Discarding other info about Node Scalar field " << *it << " and filling it with given constant value " << req_fields_info->get<double>(temp_str) << "\n";
          // For debug, we allow to fill the field with a given uniform value
          fillTpetraVec (serial_req_vec,req_fields_info->get<double>(temp_str));
        }
        else if (fname!="")
        {
          *out << "Reading Node Scalar field " << *it << " from file " << fname << "\n";
          // Read the input file and stuff it in the Tpetra vector
          readScalarFileSerial (fname,serial_req_vec,commT);
        }
        else
        {
          *out << "No file name nor constant value specified for Node Scalar field " << *it << "; initializing it to 0.\n";
          fillTpetraVec (serial_req_vec,0.);
        }

        // Fill the (possibly) parallel vector
        req_vec.doImport(serial_req_vec,importOperatorNode,Tpetra::INSERT);

        // Extracting the mesh field and the tpetra vector view
        ScalarFieldType* field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, *it);

        TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is 'Elem Scalar'?).\n");

        Teuchos::ArrayRCP<const ST> req_vec_view = req_vec.get1dView();

        //Now we have to stuff the vector in the mesh data
        for (int i(0); i<nodes.size(); ++i)
        {
          nodeId = bulkData->identifier(nodes[i]) - 1;
          lid    = nodes_map->getLocalElement((GO)(nodeId));

          values = stk::mesh::field_data(*field, nodes[i]);
          values[0] = req_vec_view[lid];
        }
      }
      else if (ftype == "Elem Scalar")
      {
        // Creating the serial and (possibly) parallel Tpetra service vectors
        Tpetra_Vector serial_req_vec(serial_elems_map);
        Tpetra_Vector req_vec(elems_map);

        temp_str = *it + " Value";
        if (req_fields_info->isParameter(temp_str))
        {
          *out << "Discarding other info about Elem Scalar field " << *it << " and filling it with constant value " << req_fields_info->get<double>(temp_str) << "\n";
          // For debug, we allow to fill the field with a given uniform value
          fillTpetraVec (serial_req_vec,req_fields_info->get<double>(temp_str));
        }
        else if (fname!="")
        {
          *out << "Reading Elem Scalar field " << *it << " from file " << fname << "\n";
          // Read the input file and stuff it in the Tpetra vector
          readScalarFileSerial (fname,serial_req_vec,commT);
        }
        else
        {
          *out << "No file name nor constant value specified for Elem Scalar field " << *it << "; initializing it to 0.\n";
          fillTpetraVec (serial_req_vec,0.);
        }

        // Fill the (possibly) parallel vector
        req_vec.doImport(serial_req_vec,importOperatorElem,Tpetra::INSERT);

        // Extracting the mesh field and the tpetra vector view
        QPScalarFieldType* field = metaData->get_field<QPScalarFieldType>(stk::topology::ELEM_RANK, *it);
        TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is 'Node Scalar'?).\n");

        Teuchos::ArrayRCP<const ST> req_vec_view = req_vec.get1dView();

        //Now we have to stuff the vector in the mesh data
        for (int i(0); i<elems.size(); ++i)
        {
          elemId = bulkData->identifier(elems[i]) - 1;
          lid    = elems_map->getLocalElement((GO)(elemId));

          values = stk::mesh::field_data(*field, elems[i]);
          values[0] = req_vec_view[lid];
        }
      }

      else if (ftype == "Node Vector")
      {
        // Loading the dimension of the Vector Field (by default equal to the mesh dimension)
        temp_str = *it + " Field Dimension";
        int fieldDim = req_fields_info->get<int>(temp_str,this->meshSpecs[0]->numDim);

        // Creating the serial and (possibly) parallel Tpetra service multivectors
        Tpetra_MultiVector serial_req_mvec(serial_nodes_map,fieldDim);
        Tpetra_MultiVector req_mvec(nodes_map,fieldDim);

        temp_str = *it + " Value";
        if (req_fields_info->isParameter(temp_str))
        {
          *out << "Discarding other info about Node Vector field " << *it << " and filling it with constant value "
               << req_fields_info->get<Teuchos::Array<double> >(temp_str) << "\n";
          // For debug, we allow to fill the field with a given uniform value
          fillTpetraMVec (serial_req_mvec,req_fields_info->get<Teuchos::Array<double> >(temp_str));
        }
        else if (fname!="")
        {
          *out << "Reading Node Vector field " << *it << " from file " << fname << "\n";
          // Read the input file and stuff it in the Tpetra multivector
          readVectorFileSerial (fname,serial_req_mvec,commT);
        }
        else
        {
          *out << "No file name nor constant value specified for Node Vector field " << *it << "; initializing it to 0.\n";
          Teuchos::Array<double> vals(fieldDim,0.);
          fillTpetraMVec (serial_req_mvec,vals);
        }

        // Fill the (possibly) parallel vector
        req_mvec.doImport(serial_req_mvec,importOperatorNode,Tpetra::INSERT);

        // Extracting the mesh field and the tpetra vector views
        VectorFieldType* field = metaData->get_field<VectorFieldType>(stk::topology::NODE_RANK, *it);
        TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is 'Elem Vector'?).\n");

        std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;
        for (int i(0); i<fieldDim; ++i)
          req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

        //Now we have to stuff the vector in the mesh data
        for (int i(0); i<nodes.size(); ++i)
        {
          nodeId = bulkData->identifier(nodes[i]) - 1;
          lid    = nodes_map->getLocalElement((GO)(nodeId));

          values = stk::mesh::field_data(*field, nodes[i]);

          for (int iDim(0); iDim<fieldDim; ++iDim)
            values[iDim] = req_mvec_view[iDim][lid];
        }
      }
      else if (ftype == "Elem Vector")
      {
        // Loading the dimension of the Vector Field (by default equal to the mesh dimension)
        temp_str = *it + " Field Dimension";
        int fieldDim = req_fields_info->get<int>(temp_str,this->meshSpecs[0]->numDim);

        // Creating the serial and (possibly) parallel Tpetra service multivectors
        Tpetra_MultiVector serial_req_mvec(serial_elems_map,fieldDim);
        Tpetra_MultiVector req_mvec(elems_map,fieldDim);

        temp_str = *it + " Value";
        if (req_fields_info->isParameter(temp_str))
        {
          *out << "Discarding other info about Elem Vector field " << *it << " and filling it with constant value "
               << req_fields_info->get<Teuchos::Array<double> >(temp_str) << "\n";
          // For debug, we allow to fill the field with a given uniform value
          fillTpetraMVec (serial_req_mvec,req_fields_info->get<Teuchos::Array<double> >(temp_str));
        }
        else if (fname!="")
        {
          *out << "Reading Elem Vector field " << *it << " from file " << fname << "\n";
          // Read the input file and stuff it in the Tpetra multivector
          readVectorFileSerial (fname,serial_req_mvec,commT);
        }
        else
        {
          *out << "No file name nor constant value specified for Elem Vector field " << *it << "; initializing it to 0.\n";
          Teuchos::Array<double> vals(fieldDim,0.);
          fillTpetraMVec (serial_req_mvec,vals);
        }

        // Fill the (possibly) parallel vector
        req_mvec.doImport(serial_req_mvec,importOperatorNode,Tpetra::INSERT);

        // Extracting the mesh field and the tpetra vector views
        VectorFieldType* field = metaData->get_field<VectorFieldType>(stk::topology::ELEM_RANK, *it);
        TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is 'Node Vector'?).\n");
        std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;
        for (int i(0); i<fieldDim; ++i)
          req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

        //Now we have to stuff the vector in the mesh data
        for (int i(0); i<elems.size(); ++i)
        {
          elemId = bulkData->identifier(elems[i]) - 1;
          lid    = elems_map->getLocalElement((GO)(elemId));

          values = stk::mesh::field_data(*field, elems[i]);

          for (int iDim(0); iDim<fieldDim; ++iDim)
            values[iDim] = req_mvec_view[iDim][lid];
        }
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                    "Sorry, I haven't yet implemented the case of field that are not Scalar nor Vector or that is not at nodal nor elemental.\n");
      }
    }
  }

  if (params->get<bool>("Write points coordinates to ascii file", false))
  {
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
    std::ofstream ofile;
    ofile.open("coordinates.ascii");
    if (!ofile.is_open())
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Cannot open coordinates file.\n");
    }

    ofile << nodes.size() << " " << 3 << "\n";

    stk::mesh::Entity node;

    for (int i(0); i<nodes.size(); ++i)
    {
      node   = bulkData->get_entity(stk::topology::NODE_RANK, i + 1);
      double* coord = stk::mesh::field_data(*coordinates_field, node);

      ofile << bulkData->identifier (nodes[i]) << " " << coord[0] << " " << coord[1]  << "\n";
    }

    ofile.close();
  }

#endif

  // Refine the mesh before starting the simulation if indicated
  uniformRefineMesh(commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);

  // Build additional mesh connectivity needed for mesh fracture (if indicated)
  computeAddlConnectivity();


}

template <unsigned Dim, class traits>
void
Albany::TmplSTKMeshStruct<Dim, traits>::DeclareParts(
              std::vector<EBSpecsStruct<Dim, traits> > ebStructArray,
              std::vector<std::string> ssNames,
              std::vector<std::string> nsNames)
{
  // Element blocks
  for (std::size_t i=0; i<ebStructArray.size(); i++) {
    // declare each element block present in the mesh
    partVec[i] = & metaData->declare_part(ebStructArray[i].name, stk::topology::ELEMENT_RANK );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*partVec[i]);
#endif
  }

  // SideSets
  for (std::size_t i=0; i<ssNames.size(); i++) {
    std::string ssn = ssNames[i];
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif
  }

  // NodeSets
  for (std::size_t i=0; i<nsNames.size(); i++) {
    std::string nsn = nsNames[i];
    nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  }

}

template <unsigned Dim, class traits>
void
Albany::EBSpecsStruct<Dim, traits>::Initialize(GO nnelems[], double blLen[]){

    name = "Block0";

    for(unsigned i = 0; i < Dim; i++){

      min[i] = 0;
      max[i] = nnelems[i];
      blLength[i] = blLen[i];

    }
}

// Template specialization functions for the different dimensions


// Specializations to read the element block information for each dimension

template<>
GO
Albany::EBSpecsStruct<0>::numElems(int i){
    return 1L;
}

template<>
void
Albany::EBSpecsStruct<0>::calcElemSizes(std::vector<double> h[]){
     h[0][0] = 1.0;
}

template<>
void
Albany::EBSpecsStruct<0>::Initialize(GO nelems[], double blLen[]){
    // Never more than one element block in a 0D problem
    name = "Block0";
    blLength[0] = blLen[0];
}

template<>
void
Albany::EBSpecsStruct<0>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params){
    // Never more than one element block in a 0D problem
    name = "Block0";
    blLength[0] = 1.0;
}

template<>
void
Albany::EBSpecsStruct<1>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params){

  // Read element block specs from input file. Note that this is called only for the multiple element
  // block case, once per element block.

    // Construct the name of the element block desired
    std::stringstream ss;
    ss << "Block " << i;

    // Get the parameter string for this block. Note the default (below) applies if there is no block
    // information in the xml file.

    std::string blkinfo = params->get<std::string>(ss.str(), "begins at 0 ends at 100 length 1.0 named Block0");

    std::string junk;

    // Parse object to break up the line
    std::stringstream parsess(blkinfo);

    //       "begins"  "at"      0      "ends"   "at"     100   "length"     1.0        "named" "Bck0"
    parsess >> junk >> junk >> min[0] >> junk >> junk >> max[0] >> junk >> blLength[0] >> junk >> name;

}

template<>
void
Albany::EBSpecsStruct<2>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{

  // Read element block specs from input file, or set defaults
    char buf[256];

    // Construct the name of the element block desired
    std::stringstream ss;
    ss << "Block " << i;

    // Get the parameter string for this block. Note the default (below) applies if there is no block
    // information in the xml file.

    std::string blkinfo = params->get<std::string>(ss.str(),
      "begins at (0, 0) ends at (100, 100) length (1.0, 1.0) named Block0");

    // Parse it
    sscanf(&blkinfo[0], "begins at (" ST_LLU"," ST_LLU") ends at (" ST_LLU"," ST_LLU") length (%lf,%lf) named %s",
      &min[0], &min[1], &max[0], &max[1], &blLength[0], &blLength[1], buf);

    name = buf;

}

template<>
void
Albany::EBSpecsStruct<3>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
{

  // Read element block specs from input file, or set defaults
    char buf[256];

    // Construct the name of the element block desired
    std::stringstream ss;
    ss << "Block " << i;

    // Get the parameter string for this block. Note the default (below) applies if there is no block
    // information in the xml file.

    std::string blkinfo = params->get<std::string>(ss.str(),
      "begins at (0, 0, 0) ends at (100, 100, 100) length (1.0, 1.0, 1.0) named Block0");

    // Parse it
    sscanf(&blkinfo[0], "begins at (" ST_LLU"," ST_LLU"," ST_LLU") ends at (" ST_LLU"," ST_LLU"," ST_LLU") length (%lf,%lf,%lf) named %s",
      &min[0], &min[1], &min[2], &max[0], &max[1], &max[2], &blLength[0], &blLength[1], &blLength[2], buf);

    name = buf;

}

// Specializations to build the mesh for each dimension
template<>
void
Albany::TmplSTKMeshStruct<0>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT)
{

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  singlePartVec[0] = partVec[0]; // Get the element block part to put the element in.
                                    // Only one block in 0D mesh

    // Declare element 1 is in that block
    stk::mesh::Entity pt  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1, singlePartVec);
    // Declare node 1 is in the node part vector
    stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, 1, nodePartVec);
    // Declare that the node belongs to the element "pt"
    // "node" is the zeroth node of this element
    bulkData->declare_relation(pt, node, 0);

    // No node sets or side sets in 0D

}

// Specialized for 0 D

template<>
void
Albany::TmplSTKMeshStruct<0, Albany::albany_stk_mesh_traits<0> >::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& /*side_set_sis*/)
{

  SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh

  TmplSTKMeshStruct<0, albany_stk_mesh_traits<0> >::buildMesh(commT);

  // STK
  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();

}

template<>
void
Albany::TmplSTKMeshStruct<1>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT)
{

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  std::vector<GO> elemNumber(1);
  unsigned int ebNo;

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
      this->PBCStruct.periodic[0] = true;
      this->PBCStruct.scale[0] = scale[0];
  }

  // Create elements and node IDs
  for (int i=0; i<elem_map->getNodeNumElements(); i++) {
    const GO elem_GID = elem_map->getGlobalElement(i);
    const GO left_node  = elem_GID;
    GO right_node = left_node+1;
    if (periodic_x) right_node %= elem_map->getGlobalNumElements();
//    if (rightNode < right_node) rightNode = right_node;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;

    // The left node number is the same as the element number
    elemNumber[0] = elem_GID;

    // find out which EB the element is in

    if(numEB == 1) // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for(ebNo = 0; ebNo < numEB; ebNo++)
       // Does the elemNumber lie in the EB?
       if(EBSpecs[ebNo].inEB(elemNumber))
         break;

      if(ebNo == numEB){ // error, we didn't find an element block that this element
                          // should fit in

          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error: Could not place element " << elem_GID <<
            " in its corresponding element block." << std::endl);

      }
    }

    singlePartVec[0] = partVec[ebNo];

    // Build element "1+elem_id" and put it in the element block
    stk::mesh::Entity edge  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
    // Build the left and right nodes of this element and put them in the node part Vec
    stk::mesh::Entity lnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+left_node, nodePartVec);
    stk::mesh::Entity rnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+right_node, nodePartVec);
    // node number 0 of this element
    bulkData->declare_relation(edge, lnode, 0);
    // node number 1 of this element
    bulkData->declare_relation(edge, rnode, 1);

    // set the coordinate values for these nodes
    double* lnode_coord = stk::mesh::field_data(*coordinates_field, lnode);
    lnode_coord[0] = x[0][left_node];
    double* rnode_coord = stk::mesh::field_data(*coordinates_field, rnode);
    rnode_coord[0] = x[0][right_node];

/*
    if(proc_rank_field){
      int* p_rank = stk::mesh::field_data(*proc_rank_field, edge);
      p_rank[0] = comm->MyPID();
    }
*/

    // Set node sets. There are no side sets currently with 1D problems (only 2D and 3D)
    if (left_node==StartIndex) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(lnode, singlePartVec);

    }
    if (right_node==elem_map->getGlobalNumElements() + StartIndex) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }
    // Periodic case -- right node is wrapped to left side
    if (right_node==StartIndex) {
      singlePartVec[0] = nsPartVec["NodeSet0"];
      bulkData->change_entity_parts(rnode, singlePartVec);
    }

// Note: Side_ranks are not currently registered for 1D elements, see
// $TRILINOS_DIR/packages/stk/stk_mesh/stk_mesh/fem/MetaData.cpp line 104.
// Skip sideset construction in 1D for this reason (GAH)

  }
}

template<>
void
Albany::TmplSTKMeshStruct<2>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT)
{

  // STK
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<GO> elemNumber(2);
  unsigned int ebNo;

  // Create elements and node IDs

  const GO nodes_x = periodic_x ? nelem[0] : nelem[0] + 1;
  const GO mod_x   = periodic_x ? nelem[0] : std::numeric_limits<GO>::max();
  const GO mod_y   = periodic_y ? nelem[1] : std::numeric_limits<GO>::max();

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
      this->PBCStruct.periodic[0] = true;
      this->PBCStruct.scale[0] = scale[0];
  }
  if (periodic_y) {
      this->PBCStruct.periodic[1] = true;
      this->PBCStruct.scale[1] = scale[1];
  }

  for (int i=0; i<elem_map->getNodeNumElements(); i++) {

    const GO elem_GID = elem_map->getGlobalElement(i);
    const GO x_GID = elem_GID % nelem[0]; // mesh column number
    const GO y_GID = elem_GID / nelem[0]; // mesh row number

    const GO x_GIDplus1 = (x_GID+1)%mod_x; // = x_GID+1 unless last element of periodic system
    const GO y_GIDplus1 = (y_GID+1)%mod_y; // = y_GID+1 unless last element of periodic system

    const GO lower_left  =  x_GID      + nodes_x * y_GID;
    const GO lower_right =  x_GIDplus1 + nodes_x * y_GID;
    const GO upper_right =  x_GIDplus1 + nodes_x * y_GIDplus1;
    const GO upper_left  =  x_GID      + nodes_x * y_GIDplus1;

    // get ID of quadrilateral -- will be doubled for trianlges below
    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;

    // Calculate elemNumber of element
    elemNumber[0] = x_GID;
    elemNumber[1] = y_GID;

    // find out which EB the element is in

    if(numEB == 1) // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for(ebNo = 0; ebNo < numEB; ebNo++){
       // Does the elemNumber lie in the EB?
//std::cout << elem_GID << " " << i << " " << ebNo << std::endl;
       if(EBSpecs[ebNo].inEB(elemNumber))
         break;
      }

      if(ebNo == numEB){ // error, we didn't find an element block that this element
                          // should fit in

          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error: Could not place element " << elem_GID <<
            " in its corresponding element block." << std::endl);

      }
    }

    singlePartVec[0] = partVec[ebNo];

//if (x_GID==0 && y_GID==0) cout << " FOUND global node " << lower_left << endl;

    // Declare Nodes = (Add one to IDs because STK requires 1-based
    stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_left, nodePartVec);
    stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_right, nodePartVec);
    stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_right, nodePartVec);
    stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_left, nodePartVec);

    if (triangles) { // pair of 3-node triangles

      stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+2*elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      stk::mesh::Entity elem2 = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+2*elem_id+1, singlePartVec);
      bulkData->declare_relation(elem2, llnode, 0);
      bulkData->declare_relation(elem2, urnode, 1);
      bulkData->declare_relation(elem2, ulnode, 2);

      // Triangle sideset construction
      if (x_GID==0) { // left edge of mesh, elem2 has side 2 on left boundary

         singlePartVec[0] = ssPartVec["SideSet0"];
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem2, side,  2 /*local side id*/);

         bulkData->declare_relation(side, ulnode, 0);
         bulkData->declare_relation(side, llnode, 1);

      }
      if (x_GIDplus1==nelem[0]) { // right edge of mesh, elem has side 1 on right boundary

         singlePartVec[0] = ssPartVec["SideSet1"];
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  1 /*local side id*/);

         bulkData->declare_relation(side, lrnode, 0);
         bulkData->declare_relation(side, urnode, 1);

      }
      if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary

         singlePartVec[0] = ssPartVec["SideSet2"];
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  0 /*local side id*/);

         bulkData->declare_relation(side, llnode, 0);
         bulkData->declare_relation(side, lrnode, 1);

      }
      if (y_GIDplus1==nelem[1]) { // top edge of mesh, elem2 has side 1 on upper boundary

         singlePartVec[0] = ssPartVec["SideSet3"];
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem2, side,  1 /*local side id*/);

         bulkData->declare_relation(side, urnode, 0);
         bulkData->declare_relation(side, ulnode, 1);

      }
    } // end pair of triangles

    else {  //4-node quad

      stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      bulkData->declare_relation(elem, ulnode, 3);

      // Quad sideset construction
      if (x_GID==0) { // left edge of mesh, elem has side 3 on left boundary

         singlePartVec[0] = ssPartVec["SideSet0"];

         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  3 /*local side id*/);

         bulkData->declare_relation(side, ulnode, 0);
         bulkData->declare_relation(side, llnode, 1);

      }
      if (x_GIDplus1==nelem[0]) { // right edge of mesh, elem has side 1 on right boundary

         singlePartVec[0] = ssPartVec["SideSet1"];

         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  1 /*local side id*/);

         bulkData->declare_relation(side, lrnode, 0);
         bulkData->declare_relation(side, urnode, 1);

      }
      if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary

         singlePartVec[0] = ssPartVec["SideSet2"];

         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  0 /*local side id*/);

         bulkData->declare_relation(side, llnode, 0);
         bulkData->declare_relation(side, lrnode, 1);

      }
      if (y_GIDplus1==nelem[1]) { // tope edge of mesh, elem has side 2 on upper boundary

         singlePartVec[0] = ssPartVec["SideSet3"];

         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);

         stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  2 /*local side id*/);

         bulkData->declare_relation(side, urnode, 0);
         bulkData->declare_relation(side, ulnode, 1);

      }
    } // end 4 node quad

    double* llnode_coord = stk::mesh::field_data(*coordinates_field, llnode);
    llnode_coord[0] = x[0][x_GID];   llnode_coord[1] = x[1][y_GID];
    double* lrnode_coord = stk::mesh::field_data(*coordinates_field, lrnode);
    lrnode_coord[0] = x[0][x_GIDplus1]; lrnode_coord[1] = x[1][y_GID];
    double* urnode_coord = stk::mesh::field_data(*coordinates_field, urnode);
    urnode_coord[0] = x[0][x_GIDplus1]; urnode_coord[1] = x[1][y_GIDplus1];
    double* ulnode_coord = stk::mesh::field_data(*coordinates_field, ulnode);
    ulnode_coord[0] = x[0][x_GID]; ulnode_coord[1] = x[1][y_GIDplus1];

    // Nodesets

    if (x_GID==0) { // left of elem is on left bdy
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(ulnode, singlePartVec);
    }
    if ((x_GIDplus1)==nelem[0]) { // right of elem is on right bdy
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (y_GID==0) { // bottom of elem is on bottom bdy
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
    }
    if ((y_GIDplus1)==nelem[1]) { // top of elem is on top bdy
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }

    // Periodic cases -- last node wraps around to first
    if ((x_GIDplus1)==0) { // right of elem is on right bdy
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if ((y_GIDplus1)==0) { // top of elem is on top bdy
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }

    // Single node at origin
    if (x_GID==0 && y_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(llnode, singlePartVec);
    }
    // Periodic cases
    if (x_GIDplus1==0 && y_GIDplus1==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (x_GID==0 && y_GIDplus1==0)  {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
    }
    if (x_GIDplus1==0 && y_GID==0) { // single node at bottom corner
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
    }
  }
}

template<>
void
Albany::TmplSTKMeshStruct<3>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT)
{

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<GO> elemNumber(3);
  unsigned int ebNo;

  const GO nodes_x  = periodic_x ? nelem[0] : nelem[0] + 1;
  const GO nodes_xy = (periodic_y && periodic_y) ? nodes_x * nelem[1] : nodes_x*(nelem[1] + 1);
  const GO mod_x    = periodic_x ? nelem[0] : std::numeric_limits<GO>::max();
  const GO mod_y    = periodic_y ? nelem[1] : std::numeric_limits<GO>::max();
  const GO mod_z    = periodic_z ? nelem[2] : std::numeric_limits<GO>::max();

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  if (periodic_x) {
      this->PBCStruct.periodic[0] = true;
      this->PBCStruct.scale[0] = scale[0];
  }
  if (periodic_y) {
      this->PBCStruct.periodic[1] = true;
      this->PBCStruct.scale[1] = scale[1];
  }
  if (periodic_z) {
      this->PBCStruct.periodic[2] = true;
      this->PBCStruct.scale[2] = scale[2];
  }

  // Create elements and node IDs
  for (int i=0; i<elem_map->getNodeNumElements(); i++) {
    const GO elem_GID = elem_map->getGlobalElement(i);
    const GO z_GID    = elem_GID / (nelem[0]*nelem[1]); // mesh column number
    const GO xy_plane = elem_GID % (nelem[0]*nelem[1]);
    const GO x_GID    = xy_plane % nelem[0]; // mesh column number
    const GO y_GID    = xy_plane / nelem[0]; // mesh row number

    const GO x_GIDplus1 = (x_GID+1)%mod_x; // = x_GID+1 unless last element of periodic system
    const GO y_GIDplus1 = (y_GID+1)%mod_y; // = y_GID+1 unless last element of periodic system
    const GO z_GIDplus1 = (z_GID+1)%mod_z; // = z_GID+1 unless last element of periodic system

    const GO lower_left  =  x_GID      + nodes_x * y_GID      + nodes_xy*z_GID;
    const GO lower_right =  x_GIDplus1 + nodes_x * y_GID      + nodes_xy*z_GID;
    const GO upper_right =  x_GIDplus1 + nodes_x * y_GIDplus1 + nodes_xy*z_GID;
    const GO upper_left  =  x_GID      + nodes_x * y_GIDplus1 + nodes_xy*z_GID;

    const GO lower_left_back  =  x_GID      + nodes_x * y_GID      + nodes_xy * z_GIDplus1;
    const GO lower_right_back =  x_GIDplus1 + nodes_x * y_GID      + nodes_xy * z_GIDplus1;
    const GO upper_right_back =  x_GIDplus1 + nodes_x * y_GIDplus1 + nodes_xy * z_GIDplus1;
    const GO upper_left_back  =  x_GID      + nodes_x * y_GIDplus1 + nodes_xy * z_GIDplus1;

    stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;

    // Calculate elemNumber of element
    elemNumber[0] = x_GID;
    elemNumber[1] = y_GID;
    elemNumber[2] = z_GID;

    // find out which EB the element is in

    if(numEB == 1) // assume all elements are in element block if there is only one
      ebNo = 0;
    else {
      for(ebNo = 0; ebNo < numEB; ebNo++)
       // Does the elemNumber lie in the EB?
       if(EBSpecs[ebNo].inEB(elemNumber))
         break;

      if(ebNo == numEB){ // error, we didn't find an element block that this element
                          // should fit in

          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error: Could not place element " << elem_GID <<
            " in its corresponding element block." << std::endl);

      }
    }

    singlePartVec[0] = partVec[ebNo];

    // Add one to IDs because STK requires 1-based
    stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
    stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_left, nodePartVec);
    stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_right, nodePartVec);
    stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_right, nodePartVec);
    stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_left, nodePartVec);
    stk::mesh::Entity llnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_left_back, nodePartVec);
    stk::mesh::Entity lrnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1+lower_right_back, nodePartVec);
    stk::mesh::Entity urnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_right_back, nodePartVec);
    stk::mesh::Entity ulnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, 1+upper_left_back, nodePartVec);
    bulkData->declare_relation(elem, llnode, 0);
    bulkData->declare_relation(elem, lrnode, 1);
    bulkData->declare_relation(elem, urnode, 2);
    bulkData->declare_relation(elem, ulnode, 3);
    bulkData->declare_relation(elem, llnodeb, 4);
    bulkData->declare_relation(elem, lrnodeb, 5);
    bulkData->declare_relation(elem, urnodeb, 6);
    bulkData->declare_relation(elem, ulnodeb, 7);

/*
    if(proc_rank_field){
      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = comm->MyPID();
    }
*/

    double* coord;
    coord = stk::mesh::field_data(*coordinates_field, llnode);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, lrnode);
    coord[0] = x[0][x_GIDplus1]; coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, urnode);
    coord[0] = x[0][x_GIDplus1]; coord[1] = x[1][y_GIDplus1]; coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, ulnode);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GIDplus1]; coord[2] = x[2][z_GID];

    coord = stk::mesh::field_data(*coordinates_field, llnodeb);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID];   coord[2] = x[2][z_GIDplus1];
    coord = stk::mesh::field_data(*coordinates_field, lrnodeb);
    coord[0] = x[0][x_GIDplus1]; coord[1] = x[1][y_GID];   coord[2] = x[2][z_GIDplus1];
    coord = stk::mesh::field_data(*coordinates_field, urnodeb);
    coord[0] = x[0][x_GIDplus1]; coord[1] = x[1][y_GIDplus1]; coord[2] = x[2][z_GIDplus1];
    coord = stk::mesh::field_data(*coordinates_field, ulnodeb);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GIDplus1]; coord[2] = x[2][z_GIDplus1];

/*
 * Do sidesets
 */

    // Hex sideset construction

    if (x_GID==0) { // left edge of mesh, elem has side 3 on left boundary

      // Sideset construction
      // elem has side 3 (0473) on left boundary

     singlePartVec[0] = ssPartVec["SideSet0"];

     stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) +
      (nelem[0] + nelem[1]) * 2 * z_GID);

     stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
     bulkData->declare_relation(elem, side,  3 /*local side id*/);

     bulkData->declare_relation(side, llnode, 0);
     bulkData->declare_relation(side, llnodeb, 2);
     bulkData->declare_relation(side, ulnodeb, 3);
     bulkData->declare_relation(side, ulnode, 1);

    }
    if ((x_GIDplus1)==nelem[0]) { // right edge of mesh, elem has side 1 on right boundary


      // elem has side 1 (1265) on right boundary

       singlePartVec[0] = ssPartVec["SideSet1"];

       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1 +
        (nelem[0] + nelem[1]) * 2 * z_GID);

       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  1 /*local side id*/);

       bulkData->declare_relation(side, lrnode, 0);
       bulkData->declare_relation(side, urnode, 1);
       bulkData->declare_relation(side, urnodeb, 3);
       bulkData->declare_relation(side, lrnodeb, 2);

    }
    if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary

    // elem has side 0 (0154) on bottom boundary

       singlePartVec[0] = ssPartVec["SideSet2"];

       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID + (nelem[0] + nelem[1]) * 2 * z_GID);

       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  0 /*local side id*/);

       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, lrnode, 1);
       bulkData->declare_relation(side, lrnodeb, 3);
       bulkData->declare_relation(side, llnodeb, 2);

    }
    if ((y_GIDplus1)==nelem[1]) { // tope edge of mesh, elem has side 2 on upper boundary

     // elem has side 2 (2376) on top boundary

     singlePartVec[0] = ssPartVec["SideSet3"];

     stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID +
      (nelem[0] + nelem[1]) * 2 * z_GID);

     stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
     bulkData->declare_relation(elem, side,  2 /*local side id*/);

     bulkData->declare_relation(side, urnode, 0);
     bulkData->declare_relation(side, ulnode, 1);
     bulkData->declare_relation(side, ulnodeb, 3);
     bulkData->declare_relation(side, urnodeb, 2);

  }
  if (z_GID==0) {

      // elem has side 4 (0321) on front boundary

     singlePartVec[0] = ssPartVec["SideSet4"];

     stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] +
      x_GID + (2 * nelem[0]) * y_GID);

     stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
     bulkData->declare_relation(elem, side,  4 /*local side id*/);

     bulkData->declare_relation(side, llnode, 0);
     bulkData->declare_relation(side, ulnode, 3);
     bulkData->declare_relation(side, urnode, 2);
     bulkData->declare_relation(side, lrnode, 1);

  }
  if ((z_GIDplus1)==nelem[2]) {

      // elem has side 5 (4567) on back boundary

     singlePartVec[0] = ssPartVec["SideSet5"];
     stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] +
      x_GID + (2 * nelem[0]) * y_GID + nelem[0]);

     stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
     bulkData->declare_relation(elem, side,  5 /*local side id*/);

     bulkData->declare_relation(side, llnodeb, 0);
     bulkData->declare_relation(side, lrnodeb, 1);
     bulkData->declare_relation(side, urnodeb, 2);
     bulkData->declare_relation(side, ulnodeb, 3);

    }

/*
 * Do Nodesets
 */

    if (x_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec); // node 0
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
    }
    if ((x_GIDplus1)==nelem[0]) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (y_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
    }
    if ((y_GIDplus1)==nelem[1]) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet4"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
    }
    if ((z_GIDplus1)==nelem[2]) {
       singlePartVec[0] = nsPartVec["NodeSet5"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }


    // Periodic cases -- last node wraps around to first
    if ((x_GIDplus1)==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if ((y_GIDplus1)==0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if ((z_GIDplus1)==0) {
       singlePartVec[0] = nsPartVec["NodeSet4"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }

    // Single node at origin
    if (x_GID==0 && y_GID==0 && z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(llnode, singlePartVec); // node 0
    }
    // Seven Periodic cases
    if (x_GIDplus1==0 && y_GID==0 && z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(lrnode, singlePartVec); // node 0
    }
    if (x_GID==0 && y_GIDplus1==0 && z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(ulnode, singlePartVec); // node 0
    }
    if (x_GIDplus1==0 && y_GIDplus1==0 && z_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(urnode, singlePartVec); // node 0
    }
    if (x_GID==0 && y_GID==0 && z_GIDplus1==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // node 0
    }
    if (x_GIDplus1==0 && y_GID==0 && z_GIDplus1==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // node 0
    }
    if (x_GID==0 && y_GIDplus1==0 && z_GIDplus1==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // node 0
    }
    if (x_GIDplus1==0 && y_GIDplus1==0 && z_GIDplus1==0) {
       singlePartVec[0] = nsPartVec["NodeSet99"];
       bulkData->change_entity_parts(urnodeb, singlePartVec); // node 0
    }

  } // end element loop
}

template<>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::TmplSTKMeshStruct<0>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK0D_DiscParams");

  return validPL;
}

template<>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::TmplSTKMeshStruct<1>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK1D_DiscParams");
  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X domain");

  for(unsigned int i = 0; i < numEB; i++){

    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(ss.str(), "begins at 0 ends at 100 length 1.0 named Bck0",
         "Beginning and ending parametric coordinates of block, block name");

  }

  return validPL;
}

template<>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::TmplSTKMeshStruct<2>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK2D_DiscParams");

  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<bool>("Periodic_y BC", false, "Flag to indicate periodic mesh in y-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
  validPL->set<std::string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");
  validPL->set<bool>("Write points coordinates to ascii file", false, "If true, writes the mesh points coordinates on file");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y domain");

  for(unsigned int i = 0; i < numEB; i++){

    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(ss.str(), "begins at (0, 0) ends at (100, 100) length (1.0, 1.0) named Bck0",
         "Beginning and ending parametric coordinates of block, block name");

  }

  return validPL;
}

template<>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::TmplSTKMeshStruct<3>::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK3D_DiscParams");

  validPL->set<bool>("Periodic_x BC", false, "Flag to indicate periodic mesh in x-dimesnsion");
  validPL->set<bool>("Periodic_y BC", false, "Flag to indicate periodic mesh in y-dimesnsion");
  validPL->set<bool>("Periodic_z BC", false, "Flag to indicate periodic mesh in z-dimesnsion");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Depth of Y discretization");
  validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");
  validPL->sublist("Required Fields Info", false, "Info for the loading of the required fields");

  // Multiple element blocks parameters
  validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y-Z domain");

  for(unsigned int i = 0; i < numEB; i++){

    std::stringstream ss;
    ss << "Block " << i;

    validPL->set<std::string>(ss.str(),
          "begins at (0, 0, 0) ends at (100, 100, 100) length (1.0, 1.0, 1.0) named Bck0",
          "Beginning and ending parametric coordinates of block, block name");

  }

  return validPL;
}

#ifdef ALBANY_FELIX

template<unsigned Dim, class traits>
void Albany::TmplSTKMeshStruct<Dim, traits>::readScalarFileSerial (std::string& fname,
                                                                   Tpetra_MultiVector& content,
                                                                   const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  GO numNodes;
  Teuchos::ArrayRCP<ST> content_nonConstView = content.get1dViewNonConst();
  if (comm->getRank() == 0)
  {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> numNodes;
      TEUCHOS_TEST_FOR_EXCEPTION (numNodes != content.getLocalLength(), Teuchos::Exceptions::InvalidParameterValue,
                                  std::endl << "Error in TmplSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << content.getLocalLength() << ")" << std::endl);

      for (GO i = 0; i < numNodes; i++)
      {
        ifile >> content_nonConstView[i];
      }
      ifile.close();
    }
    else
    {
      std::cout << "Warning in TmplSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
}

template<unsigned Dim, class traits>
void Albany::TmplSTKMeshStruct<Dim, traits>::readVectorFileSerial (std::string& fname,
                                                                   Tpetra_MultiVector& contentVec,
                                                                   const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  GO numNodes;
  int numComponents;
  if (comm->getRank() == 0)
  {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> numNodes >> numComponents;
      TEUCHOS_TEST_FOR_EXCEPTION (numNodes != contentVec.getLocalLength(), Teuchos::Exceptions::InvalidParameterValue,
                                  std::endl << "Error in TmplSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << contentVec.getLocalLength() << ")" << std::endl);
      TEUCHOS_TEST_FOR_EXCEPTION(numComponents != contentVec.getNumVectors(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in TmplSTKMeshStruct: Number of components in file " << fname << " (" << numComponents << ") is different from the number expected (" << contentVec.getNumVectors() << ")" << std::endl);

      for (int icomp(0); icomp<numComponents; ++icomp)
      {
        Teuchos::ArrayRCP<ST> contentVec_nonConstView = contentVec.getVectorNonConst(icomp)->get1dViewNonConst();
        for (GO i = 0; i < numNodes; i++)
          ifile >> contentVec_nonConstView[i];
      }
      ifile.close();
    }
    else
    {
      std::cout << "Warning in IossSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
}

template<unsigned Dim, class traits>
void Albany::TmplSTKMeshStruct<Dim, traits>::fillTpetraVec (Tpetra_Vector& vec, double value)
{
  int numElements = vec.getMap()->getNodeNumElements();
  Teuchos::ArrayRCP<Tpetra_Vector::scalar_type> vec_vals = vec.get1dViewNonConst();
  for (int i(0); i<numElements; ++i)
  {
    vec_vals[i] = value;
  }
}

template<unsigned Dim, class traits>
void Albany::TmplSTKMeshStruct<Dim, traits>::fillTpetraMVec (Tpetra_MultiVector& mvec, const Teuchos::Array<double>& values)
{
  int numElements = mvec.getMap()->getNodeNumElements();
  int numVectors = mvec.getNumVectors();

  TEUCHOS_TEST_FOR_EXCEPTION (numVectors!=values.size(),Teuchos::Exceptions::InvalidParameterValue,
                              "Error! Length of given array of values does not match the field dimension.\n");

  for (int iv(0); iv<numVectors; ++iv)
  {
    Teuchos::ArrayRCP<Tpetra_MultiVector::scalar_type> vec_vals = mvec.getVectorNonConst(iv)->get1dViewNonConst();
    for (int i(0); i<numElements; ++i)
    {
      vec_vals[i] = values[iv];
    }
  }
}

#endif // ALBANY_FELIX

