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

// Needed for rebalance 
//#include <stk_mesh/fem/DefaultFEM.hpp>
//#include <stk_rebalance/Rebalance.hpp>
//#include <stk_rebalance/Partition.hpp>
//#include <stk_rebalance/ZoltanPartition.hpp>


template<int Dim, class traits>
Albany::TmplSTKMeshStruct<Dim, traits>::TmplSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params, traits_type::size),
  numEB(params->get<int>("Element Blocks", 1)), // Read the number of element blocks. Always at least one.
  periodic(params->get("Periodic BC", false)),
  triangles(false)
{

  params->validateParameters(*this->getValidDiscretizationParameters(), 0,
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  EBSpecs.resize(numEB);

  // Read the EB extents from the parameter list and initialize the EB structs
  for(unsigned int i = 0; i < numEB; i++)

    EBSpecs[i].Initialize(i, params);

  std::vector<std::string> nsNames;

  // Construct the nodeset names

  for(int idx=0; idx < Dim*2; idx++){ // 2 nodesets per dimension (one at beginning, one at end)
    std::stringstream buf;
    buf << "NodeSet" << idx;
    nsNames.push_back(buf.str());
  }

  std::vector<std::string> ssNames;

  // Construct the sideset names

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

  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);

  // Create just enough of the mesh to figure out number of owned elements 
  // so that the problem setup can know the worksetSize

  nelem[0] = 1; // One element in the case of a single point 0D mesh (the below isn't executed)

  for(int i = 0; i < Dim; i++){ // Get the number of elements in each dimension from params

     std::stringstream buf;
     buf << i + 1 << "D Elements";
     nelem[i] = params->get<int>(buf.str());
  }

  // Calculate total number of elements
  int total_elems = nelem[0];
  for(int i = 1; i < Dim; i++)
     total_elems *= nelem[i];

  elem_map = Teuchos::rcp(new Epetra_Map(total_elems, 0, *comm)); // Distribute the elems equally

  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {
    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                               nsNames, worksetSize, partVec[0]->name(), this->interleavedOrdering));
  }
  else {

  meshSpecs.resize(numEB);

  this->allElementBlocksHaveSamePhysics=false;

  for (unsigned int eb=0; eb<numEB; eb++) {

    ebNameToIndex[partVec[eb]->name()] = eb;

    // MeshSpecs holds all info needed to set up an Albany problem

    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();
    this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                              nsNames, worksetSize, partVec[eb]->name(), this->interleavedOrdering));
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

  double scale[traits_type::size];

  // Read the scale parameters from the parameter file, one for each dimension

  for(int idx=0; idx < Dim; idx++){ // Not reached for 0D problems

    std::stringstream buf;
    buf << idx + 1 << "D Scale";

    // Read the values for "1D Scale", "2D Scale", "3D Scale"

    scale[idx] = params->get(buf.str(),     1.0);

  }

  // Format and print out information about the mesh that is being generated

  if (comm->MyPID()==0 && Dim > 0){ // Not reached for 0D problems

    std::cout <<"TmplSTKMeshStruct:: Creating " << Dim << "D cube mesh of size ";

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


  // Now, generate the axis discretization

  double h_dim;

  for(int idx=0; idx < Dim; idx++){ // Not reached for 0D problems

    x[idx].resize(nelem[idx] + 1);
    h_dim = scale[idx] / nelem[idx];

    for(unsigned int i=0; i <= nelem[idx]; i++)

      x[idx][i] = h_dim * i;

  }

  this->SetupFieldData(comm, neq_, sis, worksetSize);

  metaData->commit();

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh

  buildMesh();

  // STK
  bulkData->modification_end();
  useElementAsTopRank = true;

// Needed for rebalance 
//  stk::mesh::Selector owned_selector = metaData->locally_owned_part();
//  cout << "Before rebal " << comm->MyPID() << "  " << stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
//  // Use Zoltan to determine new partition
//  Teuchos::ParameterList emptyList;
//  stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, emptyList);
//  stk::mesh::Selector selector(metaData->universal_part());
//  stk::rebalance::rebalance(*bulkData, selector, coordinates_field, NULL, zoltan_partition);
//
//  cout << "After rebal " << comm->MyPID() << "  " << stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;

}

template <int Dim, class traits>
void Albany::TmplSTKMeshStruct<Dim, traits>::DeclareParts(
              std::vector<EBSpecsStruct<Dim> > ebStructArray, 
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

/*
  // SideSets
  for (std::size_t i=0; i<ssNames.size(); i++) {
    std::string ssn = ssNames[i];
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif
  }
*/

  // NodeSets
  for (std::size_t i=0; i<nsNames.size(); i++) {
    std::string nsn = nsNames[i];
    nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  }
}

// Template specialization functions for the different dimensions

namespace Albany { // Need to wrap all these specializations in the Albany namespace or use -fpermissive

  // Specializations to read the element block information for each dimension

  template<>
  void
  EBSpecsStruct<1>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
  {

    // Read element block specs from input file, or set defaults
  
      // Construct the name of the element block desired
      std::stringstream ss;
      ss << "Block " << i;
  
      // Get the parameter string for this block. Note the default (below) applies if there is no block
      // information in the xml file.
  
      std::string blkinfo = params->get<std::string>(ss.str(), "begins at 0 ends at 1 named Block0");
  
      std::string junk;
  
      // Parse object to break up the line
      std::stringstream parsess(blkinfo);
  
      //       "begins"  "at"          0          "ends"   "at"          1         "named"     "Bck0"
      parsess >> junk >> junk >> min[0] >> junk >> junk >> max[0] >> junk >> name;
  
  }
  
  template<>
  void
  EBSpecsStruct<2>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  
    // Read element block specs from input file, or set defaults
      char buf[256];
  
      // Construct the name of the element block desired
      std::stringstream ss;
      ss << "Block " << i;
  
      // Get the parameter string for this block. Note the default (below) applies if there is no block
      // information in the xml file.
  
      std::string blkinfo = params->get<std::string>(ss.str(), "begins at (0, 0) ends at (1, 1) named Block0");

      // Parse it
      sscanf(&blkinfo[0], "begins at (%lf,%lf) ends at (%lf,%lf) named %s", 
        &min[0], &min[1], &max[0], &max[1], buf);

      name = buf;

  }
  
  template<>
  void
  EBSpecsStruct<3>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  
    // Read element block specs from input file, or set defaults
      char buf[256];
  
      // Construct the name of the element block desired
      std::stringstream ss;
      ss << "Block " << i;
  
      // Get the parameter string for this block. Note the default (below) applies if there is no block
      // information in the xml file.
  
      std::string blkinfo = params->get<std::string>(ss.str(), "begins at (0, 0, 0) ends at (1, 1, 1) named Block0");
  
      // Parse it
      sscanf(&blkinfo[0], "begins at (%lf,%lf,%lf) ends at (%lf,%lf,%lf) named %s", 
        &min[0], &min[1], &min[2], &max[0], &max[1], &max[2], buf);

      name = buf;

  }
  
  // Specializations to build the mesh for each dimension
  template<>
  void
  TmplSTKMeshStruct<0>::buildMesh()
  {

    // Note: periodic flag just ignored for this case if it is present
  
    // STK
    std::vector<stk::mesh::Part*> nodePartVec;
    std::vector<stk::mesh::Part*> singlePartVec(1);
  
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
  
  template<>
  void
  TmplSTKMeshStruct<1>::buildMesh()
  {
  
    // STK
    std::vector<stk::mesh::Part*> nodePartVec;
    std::vector<stk::mesh::Part*> singlePartVec(1);
  
    std::vector<double> centroid(1);
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

      // Calculate centroid of element
      centroid[0] = 0.5 * (x[0][elem_GID] + x[0][elem_GID+1]);

      // find out which EB the element is in
      
      if(numEB == 1) // assume all elements are in element block if there is only one
        ebNo = 0;
      else {
        for(ebNo = 0; ebNo < numEB; ebNo++)
         // Does the centroid lie in the EB?
         if(EBSpecs[ebNo].inEB(centroid))
           break;
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
  
      // Set node and side sets
      if (elem_GID==0) {
         singlePartVec[0] = nsPartVec["NodeSet0"];
         bulkData->change_entity_parts(lnode, singlePartVec);

/*
         singlePartVec[0] = ssPartVec["SideSet0"];
         bulkData->change_entity_parts(lnode, singlePartVec);
*/
      }
      if ((elem_GID+1)==(unsigned int)elem_map->NumGlobalElements()) {
        singlePartVec[0] = nsPartVec["NodeSet1"];
        bulkData->change_entity_parts(rnode, singlePartVec);

/*
        singlePartVec[0] = ssPartVec["SideSet1"];
        bulkData->change_entity_parts(rnode, singlePartVec);
*/
      }

    }
  }
  
  template<>
  void
  TmplSTKMeshStruct<2>::buildMesh()
  {
  
    // STK
    //std::vector<stk::mesh::Part*> nodePartVec;
    //std::vector<stk::mesh::Part*> singlePartVec(1);
    stk::mesh::PartVector nodePartVec;
    stk::mesh::PartVector singlePartVec(1);
    std::vector<double> centroid(2);
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

      // Calculate centroid of element
      centroid[0] = 0.5 * (x[0][x_GID] + x[0][x_GID+1]);
      centroid[1] = 0.5 * (x[1][y_GID] + x[1][y_GID+1]);

      // find out which EB the element is in
      
      if(numEB == 1) // assume all elements are in element block if there is only one
        ebNo = 0;
      else {
        for(ebNo = 0; ebNo < numEB; ebNo++)
         // Does the centroid lie in the EB?
         if(EBSpecs[ebNo].inEB(centroid))
           break;
      }

      singlePartVec[0] = partVec[ebNo];
  
      // Declare NodesL= (Add one to IDs because STK requires 1-based
      stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_left, nodePartVec);
      stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), 1+lower_right, nodePartVec);
      stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_right, nodePartVec);
      stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), 1+upper_left, nodePartVec);
  
      if (triangles) { // pair of 3-node triangles
        stk::mesh::Entity& face  = bulkData->declare_entity(metaData->element_rank(), 1+2*elem_id, singlePartVec);
        bulkData->declare_relation(face, llnode, 0);
        bulkData->declare_relation(face, lrnode, 1);
        bulkData->declare_relation(face, urnode, 2);
        stk::mesh::Entity& face2 = bulkData->declare_entity(metaData->element_rank(), 1+2*elem_id+1, singlePartVec);
        bulkData->declare_relation(face2, llnode, 0);
        bulkData->declare_relation(face2, urnode, 1);
        bulkData->declare_relation(face2, ulnode, 2);
      }
      else {  //4-node quad
        stk::mesh::Entity& face  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
        bulkData->declare_relation(face, llnode, 0);
        bulkData->declare_relation(face, lrnode, 1);
        bulkData->declare_relation(face, urnode, 2);
        bulkData->declare_relation(face, ulnode, 3);
      }
  
      double* llnode_coord = stk::mesh::field_data(*coordinates_field, llnode);
      llnode_coord[0] = x[0][x_GID];   llnode_coord[1] = x[1][y_GID];
      double* lrnode_coord = stk::mesh::field_data(*coordinates_field, lrnode);
      lrnode_coord[0] = x[0][x_GID+1]; lrnode_coord[1] = x[1][y_GID];
      double* urnode_coord = stk::mesh::field_data(*coordinates_field, urnode);
      urnode_coord[0] = x[0][x_GID+1]; urnode_coord[1] = x[1][y_GID+1];
      double* ulnode_coord = stk::mesh::field_data(*coordinates_field, ulnode);
      ulnode_coord[0] = x[0][x_GID]; ulnode_coord[1] = x[1][y_GID+1];
  
      if (x_GID==0) {
         singlePartVec[0] = nsPartVec["NodeSet0"];
         bulkData->change_entity_parts(llnode, singlePartVec);
         bulkData->change_entity_parts(ulnode, singlePartVec);
      }
      if ((x_GID+1)==(unsigned int)nelem[0]) {
         singlePartVec[0] = nsPartVec["NodeSet1"];
         bulkData->change_entity_parts(lrnode, singlePartVec);
         bulkData->change_entity_parts(urnode, singlePartVec);
      }
      if (y_GID==0) {
         singlePartVec[0] = nsPartVec["NodeSet2"];
         bulkData->change_entity_parts(llnode, singlePartVec);
         bulkData->change_entity_parts(lrnode, singlePartVec);
      }
      if ((y_GID+1)==(unsigned int)nelem[1]) {
         singlePartVec[0] = nsPartVec["NodeSet3"];
         bulkData->change_entity_parts(ulnode, singlePartVec);
         bulkData->change_entity_parts(urnode, singlePartVec);
      }
    }
  }
  
  template<>
  void
  TmplSTKMeshStruct<3>::buildMesh()
  {
  
    std::vector<stk::mesh::Part*> nodePartVec;
    std::vector<stk::mesh::Part*> singlePartVec(1);
    std::vector<double> centroid(3);
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

      // Calculate centroid of element
      centroid[0] = 0.5 * (x[0][x_GID] + x[0][x_GID+1]);
      centroid[1] = 0.5 * (x[1][y_GID] + x[1][y_GID+1]);
      centroid[2] = 0.5 * (x[2][z_GID] + x[2][z_GID+1]);

      // find out which EB the element is in
      
      if(numEB == 1) // assume all elements are in element block if there is only one
        ebNo = 0;
      else {
        for(ebNo = 0; ebNo < numEB; ebNo++)
         // Does the centroid lie in the EB?
         if(EBSpecs[ebNo].inEB(centroid))
           break;
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
  
      
      if (x_GID==0) {
         singlePartVec[0] = nsPartVec["NodeSet0"];
         bulkData->change_entity_parts(llnode, singlePartVec);
         bulkData->change_entity_parts(ulnode, singlePartVec);
         bulkData->change_entity_parts(llnodeb, singlePartVec);
         bulkData->change_entity_parts(ulnodeb, singlePartVec);
      }
      if ((x_GID+1)==(unsigned int)nelem[0]) {
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
      if ((y_GID+1)==(unsigned int)nelem[1]) {
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
      if ((z_GID+1)==(unsigned int)nelem[2]) {
         singlePartVec[0] = nsPartVec["NodeSet5"];
         bulkData->change_entity_parts(llnodeb, singlePartVec);
         bulkData->change_entity_parts(lrnodeb, singlePartVec);
         bulkData->change_entity_parts(ulnodeb, singlePartVec);
         bulkData->change_entity_parts(urnodeb, singlePartVec);
      }
    }
  }
  
  template<>
  Teuchos::RCP<const Teuchos::ParameterList>
  TmplSTKMeshStruct<0>::getValidDiscretizationParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getValidGenericSTKParameters("ValidSTK0D_DiscParams");
  
    return validPL;
  }
  
  template<>
  Teuchos::RCP<const Teuchos::ParameterList>
  TmplSTKMeshStruct<1>::getValidDiscretizationParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getValidGenericSTKParameters("ValidSTK1D_DiscParams");
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
    validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
    validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  
    // Multiple element blocks parameters
    validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X domain");
  
    for(unsigned int i = 0; i < numEB; i++){
  
      std::stringstream ss;
      ss << "Block " << i;
  
      validPL->set<std::string>(ss.str(), "begins at 0 ends at 1 named Bck0", 
           "Beginning and ending parametric coordinates of block, block name");
  
    }
  
    return validPL;
  }
  
  template<>
  Teuchos::RCP<const Teuchos::ParameterList>
  TmplSTKMeshStruct<2>::getValidDiscretizationParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> validPL = 
      this->getValidGenericSTKParameters("ValidSTK2D_DiscParams");
  
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
    validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
    validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
    validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
    validPL->set<double>("2D Scale", 1.0, "Height of Y discretization");
    validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  
    // Multiple element blocks parameters
    validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y domain");
  
    for(unsigned int i = 0; i < numEB; i++){
  
      std::stringstream ss;
      ss << "Block " << i;
  
      validPL->set<std::string>(ss.str(), "begins at (0, 0) ends at (1, 1) named Bck0", 
           "Beginning and ending parametric coordinates of block, block name");
  
    }
  
    return validPL;
  }
  
  template<>
  Teuchos::RCP<const Teuchos::ParameterList>
  TmplSTKMeshStruct<3>::getValidDiscretizationParameters() const
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
  
    // Multiple element blocks parameters
    validPL->set<int>("Element Blocks", 1, "Number of elements blocks that span the X-Y-Z domain");
  
    for(unsigned int i = 0; i < numEB; i++){
  
      std::stringstream ss;
      ss << "Block " << i;
  
      validPL->set<std::string>(ss.str(), "begins at (0, 0, 0) ends at (1, 1, 1) named Bck0", 
           "Beginning and ending parametric coordinates of block, block name");
  
    }
  
    return validPL;
  }

}
