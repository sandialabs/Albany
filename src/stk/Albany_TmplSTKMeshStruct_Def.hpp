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
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#include <iostream>
#include "Albany_TmplSTKMeshStruct.hpp"
#include <Shards_BasicTopologies.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>
#include "Albany_Utils.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

// Rebalance 
#ifdef ALBANY_ZOLTAN
#include <stk_rebalance/Rebalance.hpp>
#include <stk_rebalance/Partition.hpp>
#include <stk_rebalance/ZoltanPartition.hpp>
#include <stk_rebalance_utils/RebalanceUtils.hpp>
#endif

// Refinement
#if 0  // Work in progress GAH
#include <stk_percept/PerceptMesh.hpp>
#include <stk_adapt/UniformRefinerPattern.hpp>
#include <stk_adapt/UniformRefiner.hpp>
#endif


template<int Dim, class traits>
Albany::TmplSTKMeshStruct<Dim, traits>::TmplSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params, traits_type::size),
  periodic(params->get("Periodic BC", false)),
  triangles(false)
{

/*
  There are two use cases of interest here.

  1. Do not specify element block information ("Element Blocks"). In this case, numEB = 1 and
      the scale (domain dimensions) and discretization is set globally across the mesh.

  2. "Element Blocks" is specified in the input file. In this case, the logical size of the element blocks
      must be specified, as well as the number of elements in each block and their size (block length).

*/

  int total_elems;

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

  for(int i = 0; i < Dim; i++){ // Get the number of elements in each dimension from params
                                // Note that nelem will default to 0 and scale to 1 if element
                                // blocks are specified

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

    if (comm->MyPID()==0 && Dim > 0){ // Not reached for 0D problems

     std::cout <<"TmplSTKMeshStruct:: Creating " << Dim << "D mesh of size ";

     std::stringstream nelem_txt, scale_txt;

     for(int idx=0; idx < Dim - 1; idx++){

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
   for(int i = 1; i < Dim; i++)
      total_elems *= nelem[i];

  }
  else { // Element blocks are present in input

    std::vector<int> min(Dim), max(Dim);

    for(int i = 0; i < Dim; i++){

      min[i] = INT_MAX;
      max[i] = INT_MIN;

      nelem[i] = 0;

    }

    // Read the EB extents from the parameter list and initialize the EB structs
    for(unsigned int eb = 0; eb < numEB; eb++){

      EBSpecs[eb].Initialize(eb, params);

//      for(int i = 0; i < Dim; i++)

//        nelem[i] += EBSpecs[eb].numElems(i);
      for(int i = 0; i < Dim; i++){

        min[i] = (min[i] < EBSpecs[eb].min[i]) ? min[i] : EBSpecs[eb].min[i];
        max[i] = (max[i] > EBSpecs[eb].max[i]) ? max[i] : EBSpecs[eb].max[i];

      }

    }

    for(int i = 0; i < Dim; i++)

      nelem[i] = max[i] - min[i];

    // Calculate total number of elements
    total_elems = nelem[0];

    for(int i = 1; i < Dim; i++)

      total_elems *= nelem[i];

  }

  std::vector<std::string> nsNames;

  // Construct the nodeset names

  for(int idx=0; idx < Dim*2; idx++){ // 2 nodesets per dimension (one at beginning, one at end)
    std::stringstream buf;
    buf << "NodeSet" << idx;
    nsNames.push_back(buf.str());
  }

  std::vector<std::string> ssNames;

  // Construct the sideset names

  if(Dim > 1 && !periodic) // Sidesets present only for 2 and 3D problems without periodic boundaries

    for(int idx=0; idx < Dim*2; idx++){ // 2 sidesets per dimension (one at beginning, one at end)
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

  for(unsigned int i = 0; i < numEB; i++){

    if (triangles)
      stk::mesh::fem::set_cell_topology<optional_element_type>(*partVec[i]);
    else 
      stk::mesh::fem::set_cell_topology<default_element_type>(*partVec[i]);
  }

  if(!periodic){ // Loop over defined sidesets and set side topo

    for(std::map<std::string, stk::mesh::Part*>::const_iterator it = ssPartVec.begin(); 
      it != ssPartVec.end(); ++it)

      stk::mesh::fem::set_cell_topology<default_element_side_type>(*it->second);

  }

  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);

  // Create just enough of the mesh to figure out number of owned elements 
  // so that the problem setup can know the worksetSize

std::cout << "Mesh size is " << total_elems << " elems." << std::endl;
  elem_map = Teuchos::rcp(new Epetra_Map(total_elems, 0, *comm)); // Distribute the elems equally

  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

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
                                ebNameToIndex, this->interleavedOrdering));
    }
 }
}

template<int Dim, class traits>
void
Albany::TmplSTKMeshStruct<Dim, traits>::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{

  // Create global mesh: Dim-D structured, rectangular

//  std::vector<std::vector<double> > h_dim;
  std::vector<double> h_dim[traits_type::size];
//  h_dim.resize(traits_type::size);
//  x.resize(traits_type::size);
//  x.resize(Dim);

  for(int idx=0; idx < Dim; idx++){ 

    // Allocate the storage

    x[idx].resize(nelem[idx] + 1);
    h_dim[idx].resize(nelem[idx] + 1);

  }

  for(unsigned int eb = 0; eb < numEB; eb++)

    EBSpecs[eb].calcElemSizes(h_dim);

  for(int idx=0; idx < Dim; idx++){

    x[idx][0] = 0;

    for(unsigned int i=1; i <= nelem[idx]; i++)

      x[idx][i] = x[idx][i - 1] + h_dim[idx][i - 1]; // place the coordinates of the element nodes


  }

  SetupFieldData(comm, neq_, sis, worksetSize);

// Setup to refine if requested
// Refine if requested
#if 0  // Work in progress GAH

    stk::percept::PerceptMesh eMesh(metaData, bulkData);

//    eMesh.printInfo("Mesh input to refiner", 0);

    // Reopen the metaData so we can refine the existing mesh
//    eMesh.reopen();
    stk::adapt::Quad4_Quad4_4 subdivideQuads(eMesh);
//    eMesh.commit();

#endif


  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh

  buildMesh(comm);

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;

#if 0 // Work in progress

// Refine if requested
  if(params->get<int>("h Refine Mesh", 0) != 0){

//    stk::percept::PerceptMesh eMesh(metaData, bulkData, false);
//    eMesh.printInfo("Mesh input to refiner", 0);

    // Reopen the metaData so we can refine the existing mesh
//    eMesh.reopen();
//    stk::adapt::Quad4_Quad4_4 subdivideQuads(eMesh);
//    eMesh.commit();

    stk::adapt::UniformRefiner refiner(eMesh, subdivideQuads, proc_rank_field);

    refiner.doBreak();

  }

#endif

// Rebalance if requested

#ifdef ALBANY_ZOLTAN
  if(params->get<bool>("Rebalance Mesh", false)){

    double imbalance;

    stk::mesh::Selector selector(metaData->universal_part());
    stk::mesh::Selector owned_selector(metaData->locally_owned_part());

    cout << "Before rebal nelements " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    cout << "Before rebal " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before first rebal: Imbalance threshold is = " << imbalance << endl;

    }


    // Use Zoltan to determine new partition
    Teuchos::ParameterList emptyList;

    stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, emptyList);
    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before second rebal: Imbalance threshold is = " << imbalance << endl;

    }


    // Configure Zoltan to use graph-based partitioning
    Teuchos::ParameterList graph;
    Teuchos::ParameterList lb_method;
    lb_method.set("LOAD BALANCING METHOD"      , "4");
    graph.sublist(stk::rebalance::Zoltan::default_parameters_name()) = lb_method;

    stk::rebalance::Zoltan zoltan_partitiona(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph);

    cout << "Universal part " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(selector, bulkData->buckets(metaData->element_rank())) << endl;
    cout << "Owned part " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partitiona);

    cout << "After rebal " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
    cout << "After rebal nelements " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before second rebal: Imbalance threshold is = " << imbalance << endl;

    }
  }
#endif  //ALBANY_ZOLTAN

}

template <int Dim, class traits>
void 
Albany::TmplSTKMeshStruct<Dim, traits>::DeclareParts(
              std::vector<EBSpecsStruct<Dim, traits> > ebStructArray, 
              std::vector<std::string> ssNames,
              std::vector<std::string> nsNames)
{
  // Element blocks
  for (std::size_t i=0; i<ebStructArray.size(); i++) {
    // declare each element block present in the mesh
    partVec[i] = & metaData->declare_part(ebStructArray[i].name, metaData->element_rank() );
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
    nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  }
}

template <int Dim, class traits>
void
Albany::EBSpecsStruct<Dim, traits>::Initialize(unsigned int nnelems[], double blLen[]){

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
int
Albany::EBSpecsStruct<0>::numElems(int i){ 
    return 1;
}

template<>
void
Albany::EBSpecsStruct<0>::calcElemSizes(std::vector<double> h[]){ 
     h[0][0] = 1.0;
}

template<>
void
Albany::EBSpecsStruct<0>::Initialize(unsigned int nelems[], double blLen[]){
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
    sscanf(&blkinfo[0], "begins at (%d,%d) ends at (%d,%d) length (%lf,%lf) named %s", 
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
    sscanf(&blkinfo[0], "begins at (%d,%d,%d) ends at (%d,%d,%d) length (%lf,%lf,%lf) named %s", 
      &min[0], &min[1], &min[2], &max[0], &max[1], &max[2], &blLength[0], &blLength[1], &blLength[2], buf);

    name = buf;

}

// Specializations to build the mesh for each dimension
template<>
void
Albany::TmplSTKMeshStruct<0>::buildMesh(const Teuchos::RCP<const Epetra_Comm>& comm)
{

  // Note: periodic flag just ignored for this case if it is present

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  singlePartVec[0] = partVec[0]; // Get the element block part to put the element in.
                                    // Only one block in 0D mesh

    // Declare element 1 is in that block
    stk::mesh::Entity& pt  = bulkData->declare_entity(metaData->element_rank(), 1, singlePartVec);
    // Declare node 1 is in the node part vector
    stk::mesh::Entity& node = bulkData->declare_entity(metaData->node_rank(), 1, nodePartVec);
    // Declare that the node belongs to the element "pt"
    // "node" is the zeroth node of this element
    bulkData->declare_relation(pt, node, 0);

    // No node sets or side sets in 0D

}

// Specialized for 0 D

template<>
void
Albany::TmplSTKMeshStruct<0, Albany::albany_stk_mesh_traits<0> >::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{

  SetupFieldData(comm, neq_, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh

  TmplSTKMeshStruct<0, albany_stk_mesh_traits<0> >::buildMesh(comm);

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;

}

template<>
void
Albany::TmplSTKMeshStruct<1>::buildMesh(const Teuchos::RCP<const Epetra_Comm>& comm)
{

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);

  std::vector<int> elemNumber(1);
  unsigned int ebNo;
  unsigned int rightNode=0;
  // Create elements and node IDs
  for (int i=0; i<elem_map->NumMyElements(); i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int left_node  = elem_GID;
    unsigned int right_node = left_node+1;
    if (periodic) right_node %= elem_map->NumGlobalElements();
    if (rightNode < right_node) rightNode = right_node;

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
    stk::mesh::Entity& edge  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
    // Build the left and right nodes of this element and put them in the node part Vec
    stk::mesh::Entity& lnode = bulkData->declare_entity(metaData->node_rank(), 1+left_node, nodePartVec);
    stk::mesh::Entity& rnode = bulkData->declare_entity(metaData->node_rank(), 1+right_node, nodePartVec);
    // node number 0 of this element
    bulkData->declare_relation(edge, lnode, 0);
    // node number 1 of this element
    bulkData->declare_relation(edge, rnode, 1);

    // set the coordinate values for these nodes
    double* lnode_coord = stk::mesh::field_data(*coordinates_field, lnode);
    lnode_coord[0] = x[0][elem_GID];
    double* rnode_coord = stk::mesh::field_data(*coordinates_field, rnode);
    rnode_coord[0] = x[0][elem_GID+1];

    int* p_rank = stk::mesh::field_data(*proc_rank_field, edge);
    p_rank[0] = comm->MyPID();

    // Set node sets. There are no side sets currently with 1D problems (only 2D and 3D)
    if (elem_GID==0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(lnode, singlePartVec);

    }
    if ((elem_GID+1)==(unsigned int)elem_map->NumGlobalElements()) {
      singlePartVec[0] = nsPartVec["NodeSet1"];
      bulkData->change_entity_parts(rnode, singlePartVec);

    }

// Note: Side_ranks are not currently registered for 1D elements, see 
// $TRILINOS_DIR/packages/stk/stk_mesh/stk_mesh/fem/FEMMetaData.cpp line 104.
// Skip sideset construction in 1D for this reason (GAH)

  }
}

template<>
void
Albany::TmplSTKMeshStruct<2>::buildMesh(const Teuchos::RCP<const Epetra_Comm>& comm)
{

  // STK
  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<int> elemNumber(2);
  unsigned int ebNo;

  // Create elements and node IDs
  const unsigned int nodes_x = periodic ? nelem[0] : nelem[0] + 1;
  const unsigned int mod_x   = periodic ? nelem[0] : std::numeric_limits<unsigned int>::max();
  const unsigned int mod_y   = periodic ? nelem[1] : std::numeric_limits<unsigned int>::max();

  for (int i=0; i<elem_map->NumMyElements(); i++) {

    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int x_GID = elem_GID % nelem[0]; // mesh column number
    const unsigned int y_GID = elem_GID / nelem[0]; // mesh row number
    const unsigned int lower_left  =  x_GID          + nodes_x* y_GID;
    const unsigned int lower_right = (x_GID+1)%mod_x + nodes_x* y_GID;
    const unsigned int upper_right = (x_GID+1)%mod_x + nodes_x*((y_GID+1)%mod_y);
    const unsigned int upper_left  =  x_GID          + nodes_x*((y_GID+1)%mod_y);

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

    // Declare Nodes = (Add one to IDs because STK requires 1-based
    stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_left, nodePartVec);
    stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_right, nodePartVec);
    stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_right, nodePartVec);
    stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_left, nodePartVec);

    if (triangles) { // pair of 3-node triangles

      stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+2*elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      stk::mesh::Entity& elem2 = bulkData->declare_entity(metaData->element_rank(), 1+2*elem_id+1, singlePartVec);
      bulkData->declare_relation(elem2, llnode, 0);
      bulkData->declare_relation(elem2, urnode, 1);
      bulkData->declare_relation(elem2, ulnode, 2);

      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = comm->MyPID();
      p_rank = stk::mesh::field_data(*proc_rank_field, elem2);
      p_rank[0] = comm->MyPID();


      if(!periodic){ // no sidesets if periodic
        // Triangle sideset construction
        if (x_GID==0) { // left edge of mesh, elem2 has side 2 on left boundary
  
           singlePartVec[0] = ssPartVec["SideSet0"];
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem2, side,  2 /*local side id*/); 
  
           bulkData->declare_relation(side, ulnode, 0); 
           bulkData->declare_relation(side, llnode, 1); 
  
        }
        if ((x_GID+1)==(unsigned int)nelem[0]) { // right edge of mesh, elem has side 1 on right boundary
  
           singlePartVec[0] = ssPartVec["SideSet1"];
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  1 /*local side id*/); 
  
           bulkData->declare_relation(side, lrnode, 0); 
           bulkData->declare_relation(side, urnode, 1); 
  
        }
        if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary
  
           singlePartVec[0] = ssPartVec["SideSet2"];
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  0 /*local side id*/); 
  
           bulkData->declare_relation(side, llnode, 0); 
           bulkData->declare_relation(side, lrnode, 1); 
  
        }
        if ((y_GID+1)==(unsigned int)nelem[1]) { // top edge of mesh, elem2 has side 1 on upper boundary
  
           singlePartVec[0] = ssPartVec["SideSet3"];
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem2, side,  1 /*local side id*/); 
  
           bulkData->declare_relation(side, urnode, 0); 
           bulkData->declare_relation(side, ulnode, 1); 
  
        }
      } // end if !periodic
    } // end pair of triangles
  
    else {  //4-node quad
  
      stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
      bulkData->declare_relation(elem, llnode, 0);
      bulkData->declare_relation(elem, lrnode, 1);
      bulkData->declare_relation(elem, urnode, 2);
      bulkData->declare_relation(elem, ulnode, 3);
  
      if(!periodic){ // no sidesets if periodic BC's
        // Quad sideset construction
        if (x_GID==0) { // left edge of mesh, elem has side 3 on left boundary
  
           singlePartVec[0] = ssPartVec["SideSet0"];
  
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID));
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  3 /*local side id*/); 
  
           bulkData->declare_relation(side, ulnode, 0); 
           bulkData->declare_relation(side, llnode, 1); 
  
        }
        if ((x_GID+1)==(unsigned int)nelem[0]) { // right edge of mesh, elem has side 1 on right boundary
  
           singlePartVec[0] = ssPartVec["SideSet1"];
  
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  1 /*local side id*/); 
  
           bulkData->declare_relation(side, lrnode, 0); 
           bulkData->declare_relation(side, urnode, 1); 
  
        }
        if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary
  
           singlePartVec[0] = ssPartVec["SideSet2"];
  
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  0 /*local side id*/); 
  
           bulkData->declare_relation(side, llnode, 0); 
           bulkData->declare_relation(side, lrnode, 1); 
  
        }
        if ((y_GID+1)==(unsigned int)nelem[1]) { // tope edge of mesh, elem has side 2 on upper boundary
  
           singlePartVec[0] = ssPartVec["SideSet3"];
  
           stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID);
  
           stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
           bulkData->declare_relation(elem, side,  2 /*local side id*/); 
  
           bulkData->declare_relation(side, urnode, 0); 
           bulkData->declare_relation(side, ulnode, 1); 
  
        }
      } // end if !periodic
    } // end 4 node quad

    double* llnode_coord = stk::mesh::field_data(*coordinates_field, llnode);
    llnode_coord[0] = x[0][x_GID];   llnode_coord[1] = x[1][y_GID];
    double* lrnode_coord = stk::mesh::field_data(*coordinates_field, lrnode);
    lrnode_coord[0] = x[0][x_GID+1]; lrnode_coord[1] = x[1][y_GID];
    double* urnode_coord = stk::mesh::field_data(*coordinates_field, urnode);
    urnode_coord[0] = x[0][x_GID+1]; urnode_coord[1] = x[1][y_GID+1];
    double* ulnode_coord = stk::mesh::field_data(*coordinates_field, ulnode);
    ulnode_coord[0] = x[0][x_GID]; ulnode_coord[1] = x[1][y_GID+1];

    // Nodesets

    if (x_GID==0) { // left of elem is on left bdy
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(ulnode, singlePartVec);

    }
    if ((x_GID+1)==(unsigned int)nelem[0]) { // right of elem is on right bdy
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
    if (y_GID==0) { // bottom of elem is on bottom bdy
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec);
       bulkData->change_entity_parts(lrnode, singlePartVec);
    }
    if ((y_GID+1)==(unsigned int)nelem[1]) { // top of elem is on top bdy
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec);
       bulkData->change_entity_parts(urnode, singlePartVec);
    }
  }
}

template<>
void
Albany::TmplSTKMeshStruct<3>::buildMesh(const Teuchos::RCP<const Epetra_Comm>& comm)
{

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  std::vector<int> elemNumber(3);
  unsigned int ebNo;

  const unsigned int nodes_x  = periodic ? nelem[0] : nelem[0] + 1;
  const unsigned int nodes_xy = periodic ? nodes_x * nelem[1] : nodes_x*(nelem[1] + 1);
  const unsigned int mod_x    = periodic ? nelem[0] : std::numeric_limits<unsigned int>::max();
  const unsigned int mod_y    = periodic ? nelem[1] : std::numeric_limits<unsigned int>::max();
  const unsigned int mod_z    = periodic ? nelem[2] : std::numeric_limits<unsigned int>::max();

  // Create elements and node IDs
  for (int i=0; i<elem_map->NumMyElements(); i++) {
    const unsigned int elem_GID = elem_map->GID(i);
    const unsigned int z_GID    = elem_GID / (nelem[0]*nelem[1]); // mesh column number
    const unsigned int xy_plane = elem_GID % (nelem[0]*nelem[1]); 
    const unsigned int x_GID    = xy_plane % nelem[0]; // mesh column number
    const unsigned int y_GID    = xy_plane / nelem[0]; // mesh row number

    const unsigned int lower_left  =  x_GID          + nodes_x* y_GID            + nodes_xy*z_GID;
    const unsigned int lower_right = (x_GID+1)%mod_x + nodes_x* y_GID            + nodes_xy*z_GID;
    const unsigned int upper_right = (x_GID+1)%mod_x + nodes_x*((y_GID+1)%mod_y) + nodes_xy*z_GID;
    const unsigned int upper_left  =  x_GID          + nodes_x*((y_GID+1)%mod_y) + nodes_xy*z_GID;

    const unsigned int lower_left_back  =  x_GID          + nodes_x* y_GID            + nodes_xy*((z_GID+1)%mod_z);
    const unsigned int lower_right_back = (x_GID+1)%mod_x + nodes_x* y_GID            + nodes_xy*((z_GID+1)%mod_z);
    const unsigned int upper_right_back = (x_GID+1)%mod_x + nodes_x*((y_GID+1)%mod_y) + nodes_xy*((z_GID+1)%mod_z);
    const unsigned int upper_left_back  =  x_GID          + nodes_x*((y_GID+1)%mod_y) + nodes_xy*((z_GID+1)%mod_z);

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
    stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
    stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_left, nodePartVec);
    stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_right, nodePartVec);
    stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_right, nodePartVec);
    stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_left, nodePartVec);
    stk::mesh::Entity& llnodeb = bulkData->declare_entity(metaData->node_rank(), 1+lower_left_back, nodePartVec);
    stk::mesh::Entity& lrnodeb = bulkData->declare_entity(metaData->node_rank(), 1+lower_right_back, nodePartVec);
    stk::mesh::Entity& urnodeb = bulkData->declare_entity(metaData->node_rank(), 1+upper_right_back, nodePartVec);
    stk::mesh::Entity& ulnodeb = bulkData->declare_entity(metaData->node_rank(), 1+upper_left_back, nodePartVec);
    bulkData->declare_relation(elem, llnode, 0);
    bulkData->declare_relation(elem, lrnode, 1);
    bulkData->declare_relation(elem, urnode, 2);
    bulkData->declare_relation(elem, ulnode, 3);
    bulkData->declare_relation(elem, llnodeb, 4);
    bulkData->declare_relation(elem, lrnodeb, 5);
    bulkData->declare_relation(elem, urnodeb, 6);
    bulkData->declare_relation(elem, ulnodeb, 7);

    int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
    p_rank[0] = comm->MyPID();

    double* coord;
    coord = stk::mesh::field_data(*coordinates_field, llnode);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, lrnode);
    coord[0] = x[0][x_GID+1]; coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, urnode);
    coord[0] = x[0][x_GID+1]; coord[1] = x[1][y_GID+1]; coord[2] = x[2][z_GID];
    coord = stk::mesh::field_data(*coordinates_field, ulnode);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID+1]; coord[2] = x[2][z_GID];

    coord = stk::mesh::field_data(*coordinates_field, llnodeb);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, lrnodeb);
    coord[0] = x[0][x_GID+1]; coord[1] = x[1][y_GID];   coord[2] = x[2][z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, urnodeb);
    coord[0] = x[0][x_GID+1]; coord[1] = x[1][y_GID+1]; coord[2] = x[2][z_GID+1];
    coord = stk::mesh::field_data(*coordinates_field, ulnodeb);
    coord[0] = x[0][x_GID];   coord[1] = x[1][y_GID+1]; coord[2] = x[2][z_GID+1];

/*
 * Do sidesets
 */

    if(!periodic){ // no sidesets if periodic BC's

      // Hex sideset construction

      if (x_GID==0) { // left edge of mesh, elem has side 3 on left boundary

        // Sideset construction
        // elem has side 3 (0473) on left boundary

       singlePartVec[0] = ssPartVec["SideSet0"];
  
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 
        (nelem[0] + nelem[1]) * 2 * z_GID);
  
       stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  3 /*local side id*/); 
  
       bulkData->declare_relation(side, llnode, 0); 
       bulkData->declare_relation(side, llnodeb, 4); 
       bulkData->declare_relation(side, ulnodeb, 7); 
       bulkData->declare_relation(side, ulnode, 3); 
  
      }
      if ((x_GID+1)==(unsigned int)nelem[0]) { // right edge of mesh, elem has side 1 on right boundary
  

        // elem has side 1 (1265) on right boundary

         singlePartVec[0] = ssPartVec["SideSet1"];
  
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * y_GID) + 1 +
          (nelem[0] + nelem[1]) * 2 * z_GID);
  
         stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  1 /*local side id*/); 
  
         bulkData->declare_relation(side, lrnode, 1); 
         bulkData->declare_relation(side, urnode, 2); 
         bulkData->declare_relation(side, urnodeb, 6); 
         bulkData->declare_relation(side, lrnodeb, 5); 
  
      }
      if (y_GID==0) { // bottom edge of mesh, elem has side 0 on lower boundary
  
      // elem has side 0 (0154) on bottom boundary

         singlePartVec[0] = ssPartVec["SideSet2"];
  
         stk::mesh::EntityId side_id = (stk::mesh::EntityId)(x_GID + (nelem[0] + nelem[1]) * 2 * z_GID);
  
         stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  0 /*local side id*/); 
  
         bulkData->declare_relation(side, llnode, 0); 
         bulkData->declare_relation(side, lrnode, 1); 
         bulkData->declare_relation(side, lrnodeb, 5); 
         bulkData->declare_relation(side, llnodeb, 4); 
  
      }
      if ((y_GID+1)==(unsigned int)nelem[1]) { // tope edge of mesh, elem has side 2 on upper boundary
  
       // elem has side 2 (2376) on top boundary

       singlePartVec[0] = ssPartVec["SideSet3"];
  
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(nelem[0] + (2 * nelem[1]) + x_GID +
        (nelem[0] + nelem[1]) * 2 * z_GID);
  
       stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  2 /*local side id*/); 
  
       bulkData->declare_relation(side, urnode, 2); 
       bulkData->declare_relation(side, ulnode, 3); 
       bulkData->declare_relation(side, ulnodeb, 7); 
       bulkData->declare_relation(side, urnodeb, 6); 
  
    }
    if (z_GID==0) {

        // elem has side 4 (0321) on front boundary

       singlePartVec[0] = ssPartVec["SideSet4"];
  
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] +
        x_GID + (2 * nelem[0]) * y_GID);
  
       stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  4 /*local side id*/); 
  
       bulkData->declare_relation(side, llnode, 0); 
       bulkData->declare_relation(side, ulnode, 3); 
       bulkData->declare_relation(side, urnode, 2); 
       bulkData->declare_relation(side, lrnode, 1); 

    }
    if ((z_GID+1)==(unsigned int)nelem[2]) {

        // elem has side 5 (4567) on back boundary

       singlePartVec[0] = ssPartVec["SideSet5"];
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)((nelem[0] + nelem[1]) * 2 * nelem[2] +
        x_GID + (2 * nelem[0]) * y_GID + nelem[0]);
  
       stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
       bulkData->declare_relation(elem, side,  5 /*local side id*/); 
  
       bulkData->declare_relation(side, llnodeb, 4); 
       bulkData->declare_relation(side, lrnodeb, 5); 
       bulkData->declare_relation(side, urnodeb, 6); 
       bulkData->declare_relation(side, ulnodeb, 7); 

      }
    } // end if !periodic

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
    if ((x_GID+1)==(unsigned int)nelem[0]) {

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
    if ((y_GID+1)==(unsigned int)nelem[1]) {

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
    if ((z_GID+1)==(unsigned int)nelem[2]) {

       singlePartVec[0] = nsPartVec["NodeSet5"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6

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
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("h Refine Mesh", 0, "Number of levels of uniform h refinement to apply");

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

  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("h Refine Mesh", 0, "Number of levels of uniform h refinement to apply");

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

  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Depth of Y discretization");
  validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("h Refine Mesh", 0, "Number of levels of uniform h refinement to apply");

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
