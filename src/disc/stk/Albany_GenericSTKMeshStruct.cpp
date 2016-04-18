//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include "Teuchos_VerboseObject.hpp"
#include "Tpetra_ComputeGatherMap.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"

#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CreateAdjacentEntities.hpp>

#include <Albany_STKNodeSharing.hpp>

// Rebalance
#ifdef ALBANY_ZOLTAN
#include <percept/stk_rebalance/Rebalance.hpp>
#include <percept/stk_rebalance/Partition.hpp>
#include <percept/stk_rebalance/ZoltanPartition.hpp>
#include <percept/stk_rebalance_utils/RebalanceUtils.hpp>
#endif

// Refinement
#ifdef ALBANY_STK_PERCEPT
#include <stk_adapt/UniformRefiner.hpp>
#include <stk_adapt/UniformRefinerPattern.hpp>
#endif

static void
printCTD(const CellTopologyData & t )
{
  std::cout << t.name ;
  std::cout << " { D = " << t.dimension ;
  std::cout << " , NV = " << t.vertex_count ;
  std::cout << " , K = 0x" << std::hex << t.key << std::dec ;
  std::cout << std::endl ;

  for ( unsigned d = 0 ; d < 4 ; ++d ) {
    for ( unsigned i = 0 ; i < t.subcell_count[d] ; ++i ) {

      const CellTopologyData_Subcell & sub = t.subcell[d][i] ;

      std::cout << "  subcell[" << d << "][" << i << "] = { " ;

      std::cout << sub.topology->name ;
      std::cout << " ," ;
      for ( unsigned j = 0 ; j < sub.topology->node_count ; ++j ) {
        std::cout << " " << sub.node[j] ;
      }
      std::cout << " }" << std::endl ;
    }
  }

  std::cout << "}" << std::endl << std::endl ;

}

Albany::GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
    int numDim_)
    : params(params_),
      adaptParams(adaptParams_),
      buildEMesh(false),
      uniformRefinementInitialized(false)
//      , out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  metaData = Teuchos::rcp(new stk::mesh::MetaData());

  buildEMesh = buildPerceptEMesh();

  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
    // eMesh needs "FAMILY_TREE" entity
    if(buildEMesh)
      entity_rank_names.push_back("FAMILY_TREE");
    metaData->initialize(numDim_, entity_rank_names);
  }

  interleavedOrdering = params->get("Interleaved Ordering",true);
  allElementBlocksHaveSamePhysics = true;
  compositeTet = params->get<bool>("Use Composite Tet 10", false);
  num_time_deriv = params->get<int>("Number Of Time Derivatives");

  requiresAutomaticAura = params->get<bool>("Use Automatic Aura", false);

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  fieldAndBulkDataSet = false;
  side_maps_present = false;
}

Albany::GenericSTKMeshStruct::~GenericSTKMeshStruct() {}

void Albany::GenericSTKMeshStruct::SetupFieldData(
    const Teuchos::RCP<const Teuchos_Comm>& commT,
    const int neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const int worksetSize)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!metaData->is_initialized(),
       std::logic_error,
       "LogicError: metaData->FEM_initialize(numDim) not yet called" << std::endl);

  neq = neq_;

  this->nodal_data_base = sis->getNodalDataBase();

  if (bulkData.is_null()) {
     const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get());
     stk::mesh::BulkData::AutomaticAuraOption auto_aura_option = stk::mesh::BulkData::NO_AUTO_AURA;
     if(requiresAutomaticAura) auto_aura_option = stk::mesh::BulkData::AUTO_AURA;
     bulkData = Teuchos::rcp(
       new stk::mesh::BulkData(*metaData,
                               *mpiComm->getRawMpiComm(),
                               auto_aura_option,
                               //worksetSize, // capability currently removed from STK_Mesh
                               false, // add_fmwk_data
                               NULL, // ConnectivityMap
                               NULL, // FieldDataManager
                               worksetSize));
  }

  // Build the container for the STK fields
  Teuchos::Array<std::string> default_solution_vector; // Empty
  Teuchos::Array<Teuchos::Array<std::string> > solution_vector;
  solution_vector.resize(num_time_deriv + 1);
  bool user_specified_solution_components = false;
  solution_vector[0] =
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", default_solution_vector);

  if(solution_vector[0].length() > 0)
     user_specified_solution_components = true;

  if(num_time_deriv >= 1){
    solution_vector[1] =
      params->get<Teuchos::Array<std::string> >("SolutionDot Vector Components", default_solution_vector);
    if(solution_vector[1].length() > 0)
       user_specified_solution_components = true;
  }

  if(num_time_deriv >= 2){
    solution_vector[2] =
      params->get<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", default_solution_vector);
    if(solution_vector[2].length() > 0)
       user_specified_solution_components = true;
  }

  Teuchos::Array<std::string> default_residual_vector; // Empty
  Teuchos::Array<std::string> residual_vector =
    params->get<Teuchos::Array<std::string> >("Residual Vector Components", default_residual_vector);

  // Build the usual Albany fields unless the user explicitly specifies the residual or solution vector layout
  if(user_specified_solution_components && (residual_vector.length() > 0)){

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<true>(params,
            metaData, bulkData, neq_, req, numDim, sis, solution_vector, residual_vector));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<false>(params,
            metaData, bulkData, neq_, req, numDim, sis, solution_vector, residual_vector));

  }

  else {

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new Albany::OrdinarySTKFieldContainer<true>(params,
            metaData, bulkData, neq_, req, numDim, sis));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::OrdinarySTKFieldContainer<false>(params,
            metaData, bulkData, neq_, req, numDim, sis));

  }

// Exodus is only for 2D and 3D. Have 1D version as well
  exoOutput = params->isType<std::string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<std::string>("Exodus Output File Name");
  exoOutputInterval = params->get<int>("Exodus Write Interval", 1);
  cdfOutput = params->isType<std::string>("NetCDF Output File Name");
  if (cdfOutput)
    cdfOutFile = params->get<std::string>("NetCDF Output File Name");

  nLat       =  params->get("NetCDF Output Number of Latitudes",100);
  nLon       =  params->get("NetCDF Output Number of Longitudes",100);
  cdfOutputInterval = params->get<int>("NetCDF Write Interval", 1);


  //get the type of transformation of STK mesh (for FELIX problems)
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh (for FELIX problems)
  felixAlpha = params->get("FELIX alpha", 0.0);
  felixL = params->get("FELIX L", 1.0);

  points_per_edge = params->get("Element Degree", 1) + 1;

  //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);

  //Does user want to write coordinates to matrix market file (e.g., for ML analysis)?
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false);

  transferSolutionToCoords = params->get<bool>("Transfer Solution to Coordinates", false);

#ifdef ALBANY_STK_PERCEPT
  // Build the eMesh if needed
  if(buildEMesh)

   eMesh = Teuchos::rcp(new stk::percept::PerceptMesh(metaData, bulkData, false));

  // Build  the requested refiners
  if(!eMesh.is_null()){

   if(buildUniformRefiner()) // cant currently build both types of refiners (FIXME)

      return;

   buildLocalRefiner();

  }
#endif

}

bool Albany::GenericSTKMeshStruct::buildPerceptEMesh(){

   // If there exists a nonempty "refine", "convert", or "enrich" string
    std::string refine = params->get<std::string>("STK Initial Refine", "");
    if(refine.length() > 0) return true;
    std::string convert = params->get<std::string>("STK Initial Enrich", "");
    if(convert.length() > 0) return true;
    std::string enrich = params->get<std::string>("STK Initial Convert", "");
    if(enrich.length() > 0) return true;

    // Or, if a percept mesh is needed to support general adaptation indicated in the "Adaptation" sublist
    if(!adaptParams.is_null()){

      std::string& method = adaptParams->get("Method", "");

      if (method == "Unif Size")
        return true;

    }

    return false;

}

bool Albany::GenericSTKMeshStruct::buildUniformRefiner(){

#ifdef ALBANY_STK_PERCEPT

    stk::adapt::BlockNamesType block_names(stk::percept::EntityRankEnd+1u);

    std::string refine = params->get<std::string>("STK Initial Refine", "");
    std::string convert = params->get<std::string>("STK Initial Convert", "");
    std::string enrich = params->get<std::string>("STK Initial Enrich", "");

    std::string convert_options = stk::adapt::UniformRefinerPatternBase::s_convert_options;
    std::string refine_options  = stk::adapt::UniformRefinerPatternBase::s_refine_options;
    std::string enrich_options  = stk::adapt::UniformRefinerPatternBase::s_enrich_options;

    // Has anything been specified?

    if(refine.length() == 0 && convert.length() == 0 && enrich.length() == 0)

       return false;

    if (refine.length())

      checkInput("refine", refine, refine_options);

    if (convert.length())

      checkInput("convert", convert, convert_options);

    if (enrich.length())

      checkInput("enrich", enrich, enrich_options);

    refinerPattern = stk::adapt::UniformRefinerPatternBase::createPattern(refine, enrich, convert, *eMesh, block_names);
    uniformRefinementInitialized = true;

    return true;

#else
    return false;
#endif

}

bool Albany::GenericSTKMeshStruct::buildLocalRefiner(){

#ifdef ALBANY_STK_PERCEPT

    if(adaptParams.is_null()) return false;

//    stk::adapt::BlockNamesType block_names = stk::adapt::BlockNamesType();
    stk::adapt::BlockNamesType block_names(stk::percept::EntityRankEnd+1u);

    std::string adapt_method = adaptParams->get<std::string>("Method", "");

    // Check if adaptation was specified
    if(adapt_method.length() == 0) return false;

    std::string pattern = adaptParams->get<std::string>("Refiner Pattern", "");

    if(pattern == "Local_Tet4_Tet4_N"){

//      refinerPattern = Teuchos::rcp(new stk::adapt::Local_Tet4_Tet4_N(*eMesh, block_names));
      refinerPattern = Teuchos::rcp(new stk::adapt::Local_Tet4_Tet4_N(*eMesh));
      return true;

    }
    else {

      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
         "Error!  Unknown local adaptation pattern in GenericSTKMeshStruct: " << pattern <<
         "!" << std::endl << "Supplied parameter list is: " << std::endl << *adaptParams
         << "\nValid patterns are: Local_Tet4_Tet4_N" << std::endl);
    }


#endif

    return false;

}

void
Albany::GenericSTKMeshStruct::cullSubsetParts(std::vector<std::string>& ssNames,
    std::map<std::string, stk::mesh::Part*>& partVec){

/*
When dealing with sideset lists, it is common to have parts that are subsets of other parts, like:
Part[ surface_12 , 18 ] {
  Supersets { {UNIVERSAL} }
  Intersection_Of { } }
  Subsets { surface_quad4_edge2d2_12 }

Part[ surface_quad4_edge2d2_12 , 19 ] {
  Supersets { {UNIVERSAL} {FEM_ROOT_CELL_TOPOLOGY_PART_Line_2} surface_12 }
  Intersection_Of { } }
  Subsets { }

This function gets rid of the subset in the list.
*/

  using std::map;

  map<std::string, stk::mesh::Part*>::iterator it;
  std::vector<stk::mesh::Part*>::const_iterator p;

  for(it = partVec.begin(); it != partVec.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it

    const stk::mesh::PartVector & subsets   = it->second->subsets();

    for ( p = subsets.begin() ; p != subsets.end() ; ++p ) {
      const std::string & n = (*p)->name();
//      std::cout << "Erasing: " << n << std::endl;
      partVec.erase(n); // erase it if it is in the base map
    }
  }

//  ssNames.clear();

  // Build the remaining data structures
  for(it = partVec.begin(); it != partVec.end(); ++it){ // loop over the parts in the map

    std::string ssn = it->first;
    ssNames.push_back(ssn);

  }
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::GenericSTKMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::GenericSTKMeshStruct::getMeshSpecs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int Albany::GenericSTKMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
    // compute numWorksets, and shrink workset size to minimize padding
    const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
    return (1 + (ebSizeMax-1) / numWorksets);
  }
}

namespace {

void only_keep_connectivity_to_specified_ranks(stk::mesh::BulkData& mesh,
                                               stk::mesh::Entity entity,
                                               stk::mesh::EntityRank keeper1,
                                               stk::mesh::EntityRank keeper2)
{
  size_t num_ranks = mesh.mesh_meta_data().entity_rank_count();

  static std::vector<stk::mesh::Entity> del_relations;
  static std::vector<stk::mesh::ConnectivityOrdinal> del_ids;

  for (stk::mesh::EntityRank r = stk::topology::NODE_RANK; r < num_ranks; ++r) {

    if (r != keeper1 && r != keeper2) {

      stk::mesh::Entity const* rels = mesh.begin(entity, r);
      stk::mesh::ConnectivityOrdinal const* ords = mesh.begin_ordinals(entity, r);
      const int num_rels = mesh.num_connectivity(entity, r);
      for (int c = 0; c < num_rels; ++c) {
        del_relations.push_back(rels[c]);
        del_ids.push_back(ords[c]);
      }
    }
  }

  for (int j = 0; j < del_relations.size(); ++j){
    stk::mesh::Entity entity = del_relations[j];
    mesh.destroy_relation(entity,entity,del_ids[j]);
  }

  del_relations.clear();
  del_ids.clear();
}

}

void Albany::GenericSTKMeshStruct::computeAddlConnectivity()
{

  if(adaptParams.is_null()) return;

  std::string& method = adaptParams->get("Method", "");

  // Mesh fracture requires full mesh connectivity, created here
  if(method == "Random"){

    stk::mesh::PartVector add_parts;
    stk::mesh::create_adjacent_entities(*bulkData, add_parts);

    stk::mesh::EntityRank sideRank = metaData->side_rank();

    std::vector<stk::mesh::Entity> element_lst;
  //  stk::mesh::get_entities(*(bulkData),stk::topology::ELEMENT_RANK,element_lst);

    stk::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();
    stk::mesh::Selector select_owned = metaData->locally_owned_part();

  /*
        stk::mesh::Selector select_owned_in_part =
        stk::mesh::Selector( metaData->universal_part() ) &
        stk::mesh::Selector( metaData->locally_owned_part() );

        stk::mesh::get_selected_entities( select_owned_in_part ,
  */

     // Loop through only on-processor elements as we are just deleting entities inside the element
     stk::mesh::get_selected_entities( select_owned,
        bulkData->buckets( stk::topology::ELEMENT_RANK ) ,
        element_lst );

    bulkData->modification_begin();

    // remove all relationships from element unless to faces(segments
    //   in 2D) or nodes
    for (int i = 0; i < element_lst.size(); ++i){
      stk::mesh::Entity element = element_lst[i];
      only_keep_connectivity_to_specified_ranks(*bulkData, element, stk::topology::NODE_RANK, sideRank);
    }

    if (bulkData->mesh_meta_data().spatial_dimension() == 3){
      // Remove extra relations from face
      std::vector<stk::mesh::Entity> face_lst;
      //stk::mesh::get_entities(*(bulkData),stk::topology::ELEMENT_RANK-1,face_lst);
      // Loop through all faces visible to this processor, as a face can be visible on two processors
      stk::mesh::get_selected_entities( select_owned_or_shared,
                                        bulkData->buckets( sideRank ) ,
                                        face_lst );

      for (int i = 0; i < face_lst.size(); ++i){
        stk::mesh::Entity face = face_lst[i];

        only_keep_connectivity_to_specified_ranks(*bulkData, face, stk::topology::ELEMENT_RANK, stk::topology::EDGE_RANK);
      }
    }

    Albany::fix_node_sharing(*bulkData);
    bulkData->modification_end();
  }

}


void Albany::GenericSTKMeshStruct::uniformRefineMesh(const Teuchos::RCP<const Teuchos_Comm>& commT){

#ifdef ALBANY_STK_PERCEPT
// Refine if requested
  if(!uniformRefinementInitialized) return;

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();


  if(!refinerPattern.is_null() && proc_rank_field){

    stk::adapt::UniformRefiner refiner(*eMesh, *refinerPattern, proc_rank_field);

    int numRefinePasses = params->get<int>("Number of Refinement Passes", 1);

    for(int pass = 0; pass < numRefinePasses; pass++){

      if(commT->getRank() == 0)
        std::cout << "Mesh refinement pass: " << pass + 1 << std::endl;

      refiner.doBreak();

    }

// printCTD(*refinerPattern->getFromTopology());
// printCTD(*refinerPattern->getToTopology());

    // Need to reset cell topology if the cell topology has changed

    if(refinerPattern->getFromTopology()->name != refinerPattern->getToTopology()->name){

      int numEB = partVec.size();

      for (int eb=0; eb<numEB; eb++) {

        meshSpecs[eb]->ctd = *refinerPattern->getToTopology();

      }
    }
  }
#endif

}


void Albany::GenericSTKMeshStruct::rebalanceInitialMeshT(const Teuchos::RCP<const Teuchos::Comm<int> >& commT){


  bool rebalance = params->get<bool>("Rebalance Mesh", false);
  bool useSerialMesh = params->get<bool>("Use Serial Mesh", false);

  if(rebalance || (useSerialMesh && commT->getSize() > 1)){

    rebalanceAdaptedMeshT(params, commT);

  }

}


void Albany::GenericSTKMeshStruct::rebalanceAdaptedMeshT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                                                        const Teuchos::RCP<const Teuchos::Comm<int> >& comm){

// Zoltan is required here

#ifdef ALBANY_ZOLTAN

    using std::cout; using std::endl;

    if(comm->getSize() <= 1)

      return;

    double imbalance;

    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

    stk::mesh::Selector selector(metaData->universal_part());
    stk::mesh::Selector owned_selector(metaData->locally_owned_part());


    if(comm->getRank() == 0){

      std::cout << "Before rebal nelements " << comm->getRank() << "  " <<
        stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(stk::topology::ELEMENT_RANK)) << endl;

      std::cout << "Before rebal " << comm->getRank() << "  " <<
        stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(stk::topology::NODE_RANK)) << endl;
    }


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, stk::topology::NODE_RANK, &selector);

    if(comm->getRank() == 0)

      std::cout << "Before rebalance: Imbalance threshold is = " << imbalance << endl;

    // Use Zoltan to determine new partition. Set the desired parameters (if any) from the input file

    Teuchos::ParameterList graph_options;

   //graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()).set("LOAD BALANCING METHOD"      , "4");
    //graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()).set("ZOLTAN DEBUG LEVEL"      , "10");

    if(params_->isSublist("Rebalance Options")){

      const Teuchos::RCP<Teuchos::ParameterList>& load_balance_method = Teuchos::sublist(params_, "Rebalance Options");

    // Set the desired parameters. The options are shown in
    // TRILINOS_ROOT/packages/stk/stk_rebalance/ZontanPartition.cpp

//      load_balance_method.set("LOAD BALANCING METHOD"      , "4");
//      load_balance_method.set("ZOLTAN DEBUG LEVEL"      , "10");

      graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()) = *load_balance_method;

    }

    const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get());

    stk::rebalance::Zoltan zoltan_partition(*bulkData, *mpiComm->getRawMpiComm(), numDim, graph_options);
    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);

    imbalance = stk::rebalance::check_balance(*bulkData, NULL,
      stk::topology::NODE_RANK, &selector);

    if(comm->getRank() == 0)
      std::cout << "After rebalance: Imbalance threshold is = " << imbalance << endl;

#if 0 // Other experiments at rebalancing

    // Configure Zoltan to use graph-based partitioning
    Teuchos::ParameterList graph;
    Teuchos::ParameterList lb_method;
    lb_method.set("LOAD BALANCING METHOD"      , "4");
    graph.sublist(stk::rebalance::Zoltan::default_parameters_name()) = lb_method;

    stk::rebalance::Zoltan zoltan_partitiona(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph);

    *out << "Universal part " << comm->MyPID() << "  " <<
      stk::mesh::count_selected_entities(selector, bulkData->buckets(metaData->element_rank())) << endl;
    *out << "Owned part " << comm->MyPID() << "  " <<
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partitiona);

    *out << "After rebal " << comm->MyPID() << "  " <<
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
    *out << "After rebal nelements " << comm->MyPID() << "  " <<
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;


    imbalance = stk::rebalance::check_balance(*bulkData, NULL,
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      *out << "Before second rebal: Imbalance threshold is = " << imbalance << endl;

    }
#endif

#endif  //ALBANY_ZOLTAN

}

void Albany::GenericSTKMeshStruct::setupMeshBlkInfo()
{
#if 0

   int nBlocks = meshSpecs.size();

   for(int i = 0; i < nBlocks; i++){

      const MeshSpecsStruct &ms = *meshSpecs[i];

      meshDynamicData[i] = Teuchos::rcp(new CellSpecs(ms.ctd, ms.worksetSize, ms.cubatureDegree,
                      numDim, neq, 0, useCompositeTet()));

   }
#endif

}

void Albany::GenericSTKMeshStruct::initializeSideSetMeshStructs (const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  if (params->isSublist ("Side Set Discretizations"))
  {
    const Teuchos::ParameterList& ssd_list = params->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& sideSets = ssd_list.get<Teuchos::Array<std::string> >("Side Sets");

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    int sideDim = this->numDim - 1;
    for (int i(0); i<sideSets.size(); ++i)
    {
      Teuchos::RCP<Albany::AbstractMeshStruct> ss_mesh;
      const std::string& ss_name = sideSets[i];
      params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(ss_name)));
      std::string method = params_ss->get<std::string>("Method");
      if (method=="SideSetSTK")
      {
        // The user said this mesh is extracted from a higher dimensional one
        TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1, std::logic_error,
                                    "Error! So far, side set mesh extraction is allowed only from STK meshes with 1 element block.\n");

        this->sideSetMeshStructs[ss_name] = Teuchos::rcp(new Albany::SideSetSTKMeshStruct(*this->meshSpecs[0], params_ss, commT));
      }
      else
      {
        if (this->sideSetMeshStructs.find(ss_name)==this->sideSetMeshStructs.end())
        {
          // We must check whether a side mesh was already created elsewhere. This happens,
          // for instance, for the basal mesh for extruded meshes.
          ss_mesh = Albany::DiscretizationFactory::createMeshStruct (params_ss,adaptParams,commT);
          this->sideSetMeshStructs[ss_name] = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(ss_mesh,false);
          TEUCHOS_TEST_FOR_EXCEPTION (this->sideSetMeshStructs[ss_name]==Teuchos::null, std::runtime_error,
                                      "Error! Could not cast side mesh to AbstractSTKMeshStruct.\n");
        }
      }

      // Checking that the side meshes have the correct dimension (in case they were loaded from file,
      // and the user mistakenly gave the wrong file name)
      TEUCHOS_TEST_FOR_EXCEPTION (sideDim!=this->sideSetMeshStructs[ss_name]->numDim, std::logic_error,
                                  "Error! Mesh on side " << ss_name << " has the wrong dimension.\n");

      // Update the side set mesh specs pointer in the mesh specs of this mesh
      this->meshSpecs[0]->sideSetMeshSpecs[ss_name] = this->sideSetMeshStructs[ss_name]->getMeshSpecs();

      // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
      typedef AbstractSTKFieldContainer::IntScalarFieldType ISFT;
      ISFT* side_to_cell_map = &this->sideSetMeshStructs[ss_name]->metaData->declare_field<ISFT> (stk::topology::ELEM_RANK, "side_to_cell_map");
      stk::mesh::put_field(*side_to_cell_map, this->sideSetMeshStructs[ss_name]->metaData->universal_part(), 1);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*side_to_cell_map, Ioss::Field::TRANSIENT);
#endif
      // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
      const int num_nodes = sideSetMeshStructs[ss_name]->getMeshSpecs()[0]->ctd.node_count;
      typedef AbstractSTKFieldContainer::IntVectorFieldType IVFT;
      IVFT* side_nodes_ids = &this->sideSetMeshStructs[ss_name]->metaData->declare_field<IVFT> (stk::topology::ELEM_RANK, "side_nodes_ids");
      stk::mesh::put_field(*side_nodes_ids, this->sideSetMeshStructs[ss_name]->metaData->universal_part(), num_nodes);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*side_nodes_ids, Ioss::Field::TRANSIENT);
#endif
    }
  }
}

void Albany::GenericSTKMeshStruct::finalizeSideSetMeshStructs (
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req,
          const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
          int worksetSize)
{
  if (this->sideSetMeshStructs.size()>0)
  {
    // Dummy sis/req if not present in the maps for a given side set.
    // This could happen if the side discretization has no requirements/states
    Teuchos::RCP<Albany::StateInfoStruct> dummy_sis = Teuchos::rcp(new Albany::StateInfoStruct());
    dummy_sis->createNodalDataBase();
    AbstractFieldContainer::FieldContainerRequirements dummy_req;

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    const Teuchos::ParameterList& ssd_list = params->sublist("Side Set Discretizations");
    for (auto it : sideSetMeshStructs)
    {
      Teuchos::RCP<Albany::SideSetSTKMeshStruct> sideMesh;
      sideMesh = Teuchos::rcp_dynamic_cast<Albany::SideSetSTKMeshStruct>(it.second,false);
      if (sideMesh!=Teuchos::null)
      {
        // SideSetSTK mesh need to build the mesh
        sideMesh->setParentMeshInfo(*this, it.first);
      }

      // We check since the basal mesh for extruded stk mesh should already have it set
      if (!it.second->fieldAndBulkDataSet)
      {
        auto it_req = side_set_req.find(it.first);
        auto it_sis = side_set_sis.find(it.first);

        auto& req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
        auto& sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);

        params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(it.first)));
        it.second->setFieldAndBulkData(commT,params_ss,neq,req,sis,worksetSize);
      }
    }
  }
}

void Albany::GenericSTKMeshStruct::buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                                                   std::map<GO,GO>& sideMap,
                                                                   std::map<GO,std::vector<GO>>& sideNodeMap)
{
  TEUCHOS_TEST_FOR_EXCEPTION (sideSetMeshStructs.find(sideSetName)==sideSetMeshStructs.end(), Teuchos::Exceptions::InvalidParameter,
                              "Error in 'buildSideNodeToSideSetCellNodeMap': side set " << sideSetName << " does not have a mesh.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> side_mesh = sideSetMeshStructs.at(sideSetName);

  // NOTE 1: the stk fields memorize maps from 2D to 3D values (since fields are defined on 2D mesh), while the input
  //         maps map 3D values to 2D ones. For instance, if node 0 of side3D 41 is mapped to node 2 of cell2D 12, the
  //         stk field values will be field(12) = [*, *, 0], while the input map will be sideNodeMap[41] = [2, *, *]
  //         This is because in Load/SaveSideSetStateField we can only use 3D data to access the map. On the other hand,
  //         the stk fields are defined in 2D, so it makes sense to have them indicized on the 2D values.
  // NOTE 2: The stk side_map maps a 2D cell id to a pair <cell3D_GID, side_lid>, where side_lid is the lid of the side
  //         within the element. The input map, instead, maps the directly side3D_GID into the cell3D_GID..

  // Extract 2D cells
  stk::mesh::Selector selector = stk::mesh::Selector(side_mesh->metaData->locally_owned_part());
  std::vector<stk::mesh::Entity> cells2D;
  stk::mesh::get_selected_entities(selector, side_mesh->bulkData->buckets(stk::topology::ELEM_RANK), cells2D);

  if (cells2D.size()==0)
  {
    // It can happen if the mesh is partitioned and this process does not own the side
    return;
  }

  const stk::topology::rank_t SIDE_RANK = metaData->side_rank();
  const int num_nodes = side_mesh->bulkData->num_nodes(cells2D[0]);
  GO* cell3D_id;
  GO* side_nodes_ids;
  GO cell2D_GID, side3D_GID;
  int side_lid;
  int num_sides;
  typedef AbstractSTKFieldContainer::IntScalarFieldType ISFT;
  typedef AbstractSTKFieldContainer::IntVectorFieldType IVFT;
  ISFT* side_to_cell_map   = this->sideSetMeshStructs[sideSetName]->metaData->get_field<ISFT> (stk::topology::ELEM_RANK, "side_to_cell_map");
  IVFT* side_nodes_ids_map = this->sideSetMeshStructs[sideSetName]->metaData->get_field<IVFT> (stk::topology::ELEM_RANK, "side_nodes_ids");
  std::vector<stk::mesh::EntityId> cell2D_nodes_ids(num_nodes), side3D_nodes_ids(num_nodes);
  const stk::mesh::Entity* side3D_nodes;
  const stk::mesh::Entity* cell2D_nodes;
  if (side_mesh->side_maps_present)
  {
    // This mesh was loaded from a file that stored the side maps.
    // Hence, we just read it and stuff the map with it
    for (const auto& cell2D : cells2D)
    {
      // Get the stk field data
      cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
      side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);
      stk::mesh::Entity cell3D = bulkData->get_entity(stk::topology::ELEM_RANK,*cell3D_id);

      num_sides = bulkData->num_sides(cell3D);
      const stk::mesh::Entity* cell_sides = bulkData->begin(cell3D,SIDE_RANK);
      side_lid = -1;
      for (int iside(0); iside<num_sides; ++iside)
      {
        side3D_nodes = bulkData->begin_nodes(cell_sides[iside]);
        for (int inode(0); inode<num_nodes; ++inode)
        {
          side3D_nodes_ids[inode] = bulkData->identifier(side3D_nodes[inode]);
        }
        if (std::is_permutation(side3D_nodes_ids.begin(),side3D_nodes_ids.end(), side_nodes_ids))
        {
          side_lid = iside;
          side3D_GID = bulkData->identifier(cell_sides[iside])-1;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION (side_lid==-1, std::logic_error, "Error! Cannot locate the right side in the cell.\n");

      sideMap[side3D_GID] = side_mesh->bulkData->identifier(cell2D)-1;
      sideNodeMap[side3D_GID].resize(num_nodes);
      cell2D_nodes = side_mesh->bulkData->begin_nodes(cell2D);
      for (int i(0); i<num_nodes; ++i)
      {
        auto it = std::find(side3D_nodes_ids.begin(),side3D_nodes_ids.end(),side_nodes_ids[i]);
        sideNodeMap[side3D_GID][std::distance(side3D_nodes_ids.begin(),it)] = i;
      }
    }

    return;
  }

  const stk::mesh::Entity* side_cells;
  for (const auto& cell2D : cells2D)
  {
    // Get the stk field data
    cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
    side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);

    // The side-id is assumed equal to the cell-id in the side mesh...
    side3D_GID = cell2D_GID = side_mesh->bulkData->identifier(cell2D)-1;
    stk::mesh::Entity side3D = bulkData->get_entity(SIDE_RANK, side3D_GID+1);

    // Safety check
    TEUCHOS_TEST_FOR_EXCEPTION (bulkData->num_elements(side3D)!=1, std::logic_error,
                                "Error! Side " << side3D_GID << " has more/less than 1 adjacent element.\n");

    side_cells = bulkData->begin_elements(side3D);
    stk::mesh::Entity cell3D = side_cells[0];

    *cell3D_id = bulkData->identifier(cell3D);

    sideMap[side3D_GID] = cell2D_GID;
    sideNodeMap[side3D_GID].resize(num_nodes);

    // Now we determine the lid of the side within the element and also the node ordering
    cell2D_nodes = side_mesh->bulkData->begin_nodes(cell2D);
    side3D_nodes = bulkData->begin_nodes(side3D);
    for (int i(0); i<num_nodes; ++i)
    {
      cell2D_nodes_ids[i] = side_mesh->bulkData->identifier(cell2D_nodes[i]);
      side3D_nodes_ids[i] = bulkData->identifier(side3D_nodes[i]);
    }

    for (int i(0); i<num_nodes; ++i)
    {
      auto it = std::find(cell2D_nodes_ids.begin(),cell2D_nodes_ids.end(),side3D_nodes_ids[i]);
      sideNodeMap[side3D_GID][i] = std::distance(cell2D_nodes_ids.begin(),it);
      side_nodes_ids[std::distance(cell2D_nodes_ids.begin(),it)] = side3D_nodes_ids[i];
    }
  }

  // Just in case this method gets called twice
  side_mesh->side_maps_present = true;
}

void Albany::GenericSTKMeshStruct::printParts(stk::mesh::MetaData *metaData)
{
  std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;
  stk::mesh::PartVector all_parts = metaData->get_parts();

  for (stk::mesh::PartVector::iterator i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part)
  {
    stk::mesh::Part* part = *i_part ;
    std::cout << "\t" << part->name() << std::endl;
  }
}

void Albany::GenericSTKMeshStruct::loadRequiredInputFields (const AbstractFieldContainer::FieldContainerRequirements& req,
                                                            const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(commT->getRank(), commT->getSize());
  out->setOutputToRootOnly(0);

  *out << "Processing field requirements...\n";

  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes, elems;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK), elems);

  Teuchos::Array<GO> nodeIndices(nodes.size()), elemIndices(elems.size());
  for (int i = 0; i < nodes.size(); ++i)
    nodeIndices[i] = bulkData->identifier(nodes[i]) - 1;
  for (int i = 0; i < elems.size(); ++i)
    elemIndices[i] = bulkData->identifier(elems[i]) - 1;


  // Creating the serial and parallel node maps
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

  Teuchos::RCP<const Tpetra_Map> nodes_map = Tpetra::createNonContigMapWithNode<LO, GO> (nodeIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());
  Teuchos::RCP<const Tpetra_Map> elems_map = Tpetra::createNonContigMapWithNode<LO, GO> (elemIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());

  // NOTE: the serial map cannot be created linearly, with GIDs from 0 to numGlobalNodes/Elems, since
  //       this may be a boundary mesh, and the GIDs may not start from 0, nor be contiguous.
  //       Therefore, we must create a root map. Moreover, we need the GIDs sorted (so that, regardless
  //       of the GID, we read the serial input files in the correct order), and we can't sort them
  //       once the map is created, so we cannot use the Tpetra utility gatherMap.

  int num_my_nodes = nodes_map->getNodeNumElements();
  int num_my_elems = elems_map->getNodeNumElements();

  // Note: I tried to use gather, but the number of elements on each process may be different.
  int num_proc = commT->getSize();
  int my_rank  = commT->getRank();
  int num_global_nodes, num_global_elems;
  Teuchos::Array<GO> allNodesToRoot(0),allElemsToRoot(0);
  for (int pid=0; pid<num_proc; ++pid)
  {
    int nb_n = num_my_nodes;
    int nb_e = num_my_elems;
    Teuchos::broadcast(*commT,pid,1,&nb_n);
    Teuchos::broadcast(*commT,pid,1,&nb_e);

    GO *tmp_nodes, *tmp_elems;
    if (pid==my_rank)
    {
      tmp_nodes = nodeIndices.getRawPtr();
      tmp_elems = elemIndices.getRawPtr();
    }
    else
    {
      tmp_nodes = new GO[nb_n];
      tmp_elems = new GO[nb_e];
    }

    Teuchos::broadcast(*commT,pid,nb_n,tmp_nodes);
    Teuchos::broadcast(*commT,pid,nb_e,tmp_elems);
    if (my_rank==0)
    {
      allNodesToRoot.insert(allNodesToRoot.begin(),tmp_nodes,tmp_nodes+nb_n);
      allElemsToRoot.insert(allElemsToRoot.begin(),tmp_elems,tmp_elems+nb_e);
    }

    if (pid!=my_rank)
    {
      delete[] tmp_nodes;
      delete[] tmp_elems;
    }
  }

  // Sorting
  std::sort(allNodesToRoot.begin(),allNodesToRoot.end());
  std::sort(allElemsToRoot.begin(),allElemsToRoot.end());

  // Removing duplicates (should do nothing for the elements)
  auto it_nodes = std::unique(allNodesToRoot.begin(),allNodesToRoot.end());
  auto it_elems = std::unique(allElemsToRoot.begin(),allElemsToRoot.end());
  allNodesToRoot.resize(std::distance(allNodesToRoot.begin(),it_nodes));    // Resize to the actual number of unique nodes
  allElemsToRoot.resize(std::distance(allElemsToRoot.begin(),it_elems));    // Resize to the actual number of unique elements

  GO node_base, elem_base;
  if (my_rank==0)
  {
    node_base = *std::min_element(allNodesToRoot.begin(), allNodesToRoot.end());
    elem_base = *std::min_element(allElemsToRoot.begin(), allElemsToRoot.end());
  }
  Teuchos::broadcast(*commT,0,1,&node_base);
  Teuchos::broadcast(*commT,0,1,&elem_base);

  num_global_nodes = allNodesToRoot.size();
  num_global_elems = allElemsToRoot.size();
  Teuchos::broadcast(*commT,0,1,&num_global_nodes);
  Teuchos::broadcast(*commT,0,1,&num_global_elems);

  Teuchos::RCP<const Tpetra_Map> serial_nodes_map, serial_elems_map;
  serial_nodes_map = Teuchos::rcp (new Tpetra_Map (num_global_nodes,allNodesToRoot(),node_base,commT,KokkosClassic::Details::getNode<KokkosNode>()));
  serial_elems_map = Teuchos::rcp (new Tpetra_Map (num_global_elems,allElemsToRoot(),elem_base,commT,KokkosClassic::Details::getNode<KokkosNode>()));

  //Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Tpetra::Details::computeGatherMap(nodes_map,out);
  //Teuchos::RCP<const Tpetra_Map> serial_elems_map = Tpetra::Details::computeGatherMap(elems_map,out);

  // Creating the Tpetra_Import object (to transfer from serial to parallel vectors)
  Tpetra_Import importOperatorNode (serial_nodes_map, nodes_map);
  Tpetra_Import importOperatorElem (serial_elems_map, elems_map);

  Teuchos::ParameterList dummyList;
  Teuchos::ParameterList* req_fields_info;
  if (params->isSublist("Required Fields Info"))
  {
    req_fields_info = &params->sublist("Required Fields Info");
  }
  else
  {
    req_fields_info = &dummyList;
  }

  int num_fields = req_fields_info->get<int>("Number Of Fields",0);
  // L.B: is this check a good idea?
  TEUCHOS_TEST_FOR_EXCEPTION (num_fields!=req.size(), std::logic_error, "Error! The number of required fields in the discretization parameter list does not match the number of requirements declared in the problem section.\n");

  std::string fname, ftype;
  for (int ifield=0; ifield<num_fields; ++ifield)
  {
    std::stringstream ss;
    ss << "Field " << ifield;
    const Teuchos::ParameterList& fparams = req_fields_info->sublist(ss.str());

    fname = fparams.get<std::string>("Field Name");
    ftype = fparams.get<std::string>("Field Type");

    // L.B: again, is this check a good idea?
    TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(),req.end(),fname)==req.end(), std::logic_error, "Error! The field " << fname << " is not listed in the problem requirements.\n");

    if (ftype=="From Mesh")
    {
      *out << "Skipping field " << fname << " since it's listed as already present in the mesh. Make sure this is true, since we can't check!\n";
      continue;
    }
    else if (ftype=="Output")
    {
      *out << "Skipping field " << fname << " since it's listed as output (computed at run time). Make sure there's an evaluator set to save it.\n";
      continue;
    }

    // Depending on the input field type, we need to use different pointers/importers/vectors
    if (ftype == "Node Scalar")
    {
      loadField (fname, fparams, importOperatorNode, nodes, commT, true, true, false);
    }
    else if (ftype == "Elem Scalar")
    {
      loadField (fname, fparams, importOperatorElem, elems, commT, false, true, false);
    }
    else if (ftype == "Node Vector")
    {
      loadField (fname, fparams, importOperatorNode, nodes, commT, true, false, false);
    }
    else if (ftype == "Elem Vector")
    {
      loadField (fname, fparams, importOperatorElem, elems, commT, false, false, false);
    }
    else if (ftype == "Node Layered Scalar")
    {
      loadField (fname, fparams, importOperatorNode, nodes, commT, true, true, true);
    }
    else if (ftype == "Elem Layered Scalar")
    {
      loadField (fname, fparams, importOperatorElem, elems, commT, false, true, true);
    }
    else if (ftype == "Node Layered Vector")
    {
      loadField (fname, fparams, importOperatorNode, nodes, commT, true, false, true);
    }
    else if (ftype == "Elem Layered Vector")
    {
      loadField (fname, fparams, importOperatorElem, elems, commT, false, false, true);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Sorry, I haven't yet implemented the case of field that are not (Layered) Scalar nor " <<
                                  "(Layered) Vector or that is not defined at nodes nor elements.\n");
    }
  }
}

void Albany::GenericSTKMeshStruct::loadField (const std::string& field_name, const Teuchos::ParameterList& params,
                                              const Tpetra_Import& importOperator, const std::vector<stk::mesh::Entity>& entities,
                                              const Teuchos::RCP<const Teuchos_Comm>& commT,
                                              bool node, bool scalar, bool layered)
{
  std::vector<double> norm_layers_coords;

  // Getting the serial and (possibly) parallel maps
  const Teuchos::RCP<const Tpetra_Map> serial_map = importOperator.getSourceMap();
  const Teuchos::RCP<const Tpetra_Map> map = importOperator.getTargetMap();

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(commT->getRank(), commT->getSize());
  out->setOutputToRootOnly(0);

  std::string temp_str;
  std::string field_type = (node ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");

  // The serial Tpetra service multivector
  Teuchos::RCP<Tpetra_MultiVector> serial_req_mvec;

  if (params.isParameter("Field Value"))
  {
    Teuchos::Array<double> values = params.get<Teuchos::Array<double> >("Field Value");

    serial_req_mvec = Teuchos::rcp(new Tpetra_MultiVector(serial_map,values.size()));

    *out << "Discarding other info about " << field_type << " field " << field_name << " and filling it with constant value " << values << "\n";

    // For debug, we allow to fill the field with a given uniform value
    for (int iv(0); iv<serial_req_mvec->getNumVectors(); ++iv)
    {
      serial_req_mvec->getVectorNonConst(iv)->putScalar(values[iv]);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!params.isParameter("File Name"), std::logic_error, "Error! No file name or constant value specified for field " << field_name << ".\n");

    std::string fname = params.get<std::string>("File Name");

    *out << "Reading " << field_type << " field " << field_name << " from file " << fname << "\n";
    // Read the input file and stuff it in the Tpetra multivector

    if (scalar)
    {
      if (layered)
      {
        readLayeredScalarFileSerial (fname,serial_req_mvec,serial_map,norm_layers_coords,commT);
      }
      else
      {
        readScalarFileSerial (fname,serial_req_mvec,serial_map,commT);
      }
    }
    else
    {
      if (layered)
      {
        readLayeredVectorFileSerial (fname,serial_req_mvec,serial_map,norm_layers_coords,commT);
      }
      else
      {
        readVectorFileSerial (fname,serial_req_mvec,serial_map,commT);
      }
    }

    if (params.isParameter("Scale Factor"))
    {
      *out << " - Scaling " << field_type << " field " << field_name << "\n";

      Teuchos::Array<double> scale_factors = params.get<Teuchos::Array<double> >("Scale Factor");
      TEUCHOS_TEST_FOR_EXCEPTION (scale_factors.size()!=serial_req_mvec->getNumVectors(), Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given scale factors vector size does not match the field dimension.\n");
      serial_req_mvec->scale (scale_factors);
    }
  }

  // Fill the (possibly) parallel vector
  Tpetra_MultiVector req_mvec(map,serial_req_mvec->getNumVectors());
  req_mvec.doImport(*serial_req_mvec,importOperator,Tpetra::INSERT);
  std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;

  for (int i(0); i<req_mvec.getNumVectors(); ++i)
    req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

  if (layered)
  {
    // Broadcast the normalized layers coordinates
    int size = norm_layers_coords.size();
    Teuchos::broadcast(*commT,0,1,&size);
    norm_layers_coords.resize(size);
    Teuchos::broadcast(*commT,0,size,norm_layers_coords.data());
    temp_str = field_name + "_NLC";
    fieldContainer->getMeshVectorStates()[temp_str] = norm_layers_coords;
  }

  //Now we have to stuff the vector in the mesh data
  typedef AbstractSTKFieldContainer::ScalarFieldType  SFT;
  typedef AbstractSTKFieldContainer::VectorFieldType  VFT;
  typedef AbstractSTKFieldContainer::TensorFieldType  TFT;

  SFT* scalar_field = 0;
  VFT* vector_field = 0;
  TFT* tensor_field = 0;

  if (scalar && !layered)
  {
    // Purely scalar field
    if (node)
      scalar_field = metaData->get_field<SFT> (stk::topology::NODE_RANK, field_name);
    else
      scalar_field = metaData->get_field<SFT> (stk::topology::ELEM_RANK, field_name);
  }
  else if ( scalar==layered )
  {
    // Either (non-layered) vector or layered scalar field
    if (node)
      vector_field = metaData->get_field<VFT> (stk::topology::NODE_RANK, field_name);
    else
      vector_field = metaData->get_field<VFT> (stk::topology::ELEM_RANK, field_name);
  }
  else
  {
    // Layered vector field
    if (node)
      tensor_field = metaData->get_field<TFT> (stk::topology::NODE_RANK, field_name);
    else
      tensor_field = metaData->get_field<TFT> (stk::topology::ELEM_RANK, field_name);
  }

  TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0 && vector_field==0 && tensor_field==0, std::logic_error,
                              "Error! Field " << field_name << " not present (perhaps is not '" << field_type << "'?).\n");

  stk::mesh::EntityId gid;
  LO lid;
  double* values;
  for (int i(0); i<entities.size(); ++i)
  {
    if (scalar_field!=0)
      values = stk::mesh::field_data(*scalar_field, entities[i]);
    else if (vector_field!=0)
      values = stk::mesh::field_data(*vector_field, entities[i]);
    else
      values = stk::mesh::field_data(*tensor_field, entities[i]);

    gid    = bulkData->identifier(entities[i]) - 1;
    lid    = map->getLocalElement((GO)(gid));
    for (int iDim(0); iDim<req_mvec_view.size(); ++iDim)
      values[iDim] = req_mvec_view[iDim][lid];
  }
}

void Albany::GenericSTKMeshStruct::readScalarFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& mvec,
                                                         const Teuchos::RCP<const Tpetra_Map>& map,
                                                         const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  // It's a scalar, so we already know MultiVector has only 1 vector
  mvec = Teuchos::rcp(new Tpetra_MultiVector(map,1));

  if (comm->getRank() != 0)
  {
    // Only process 0 will load the file...
    return;
  }

  GO numNodes;
  Teuchos::ArrayRCP<ST> nonConstView = mvec->get1dViewNonConst();

  std::ifstream ifile;
  ifile.open(fname.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

  ifile >> numNodes;
  TEUCHOS_TEST_FOR_EXCEPTION (numNodes != map->getNodeNumElements(), Teuchos::Exceptions::InvalidParameterValue,
                              "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                              "is different from the number expected (" << map->getNodeNumElements() << ").\n");

  for (GO i = 0; i < numNodes; i++)
    ifile >> nonConstView[i];

  ifile.close();
}

void Albany::GenericSTKMeshStruct::readVectorFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& mvec,
                                                         const Teuchos::RCP<const Tpetra_Map>& map,
                                                         const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  int numComponents = 0;
  if (comm->getRank() == 0)
  {
    GO numNodes;
    std::ifstream ifile;
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numComponents;

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != map->getNodeNumElements(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << map->getNodeNumElements() << ").\n");

    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numComponents));

    for (int icomp(0); icomp<numComponents; ++icomp)
    {
      Teuchos::ArrayRCP<ST> nonConstView = mvec->getVectorNonConst(icomp)->get1dViewNonConst();
      for (GO i = 0; i < numNodes; i++)
        ifile >> nonConstView[i];
    }
    ifile.close();
  }

  Teuchos::broadcast(*comm,0,1,&numComponents);
  if (comm->getRank() != 0)
    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numComponents));
}

void Albany::GenericSTKMeshStruct::readLayeredScalarFileSerial (const std::string &fname, Teuchos::RCP<Tpetra_MultiVector>& mvec,
                                                                const Teuchos::RCP<const Tpetra_Map>& map,
                                                                std::vector<double>& normalizedLayersCoords,
                                                                const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  int numLayers = 0;
  if (comm->getRank() == 0)
  {
    GO numNodes;
    std::ifstream ifile;

    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numLayers;

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != map->getNodeNumElements(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << map->getNodeNumElements() << ").\n");

    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numLayers));

    normalizedLayersCoords.resize(numLayers);
    for (int il = 0; il < numLayers; ++il)
      ifile >> normalizedLayersCoords[il];

    for (int il(0); il<numLayers; ++il)
    {
      Teuchos::ArrayRCP<ST> nonConstView = mvec->getVectorNonConst(il)->get1dViewNonConst();
      for (GO i = 0; i < numNodes; i++)
        ifile >> nonConstView[i];
    }
    ifile.close();
  }

  Teuchos::broadcast(*comm,0,1,&numLayers);
  if (comm->getRank() != 0)
    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numLayers));
}

void Albany::GenericSTKMeshStruct::readLayeredVectorFileSerial (const std::string &fname, Teuchos::RCP<Tpetra_MultiVector>& mvec,
                                                                const Teuchos::RCP<const Tpetra_Map>& map,
                                                                std::vector<double>& normalizedLayersCoords,
                                                                const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  int numVectors = 0;
  if (comm->getRank() == 0)
  {
    GO numNodes;
    int numLayers,numComponents;
    std::ifstream ifile;

    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numComponents >> numLayers;

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != map->getNodeNumElements(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << map->getNodeNumElements() << ").\n");

    numVectors = numLayers*numComponents;
    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numVectors));

    normalizedLayersCoords.resize(numLayers);
    for (int il = 0; il < numLayers; ++il)
      ifile >> normalizedLayersCoords[il];

    // Layer ordering: before switching component, we first do all the layers of the current component
    // This is because with the stk field (natural ordering) we want to keep the layer dimension last.
    // Ex: a 2D field f(i,j) would be stored at the raw array position i*num_cols+j. In our case,
    //     num_cols is the number of layers, and num_rows the number of field components
    for (int il(0); il<numLayers; ++il)
    {
      for (int icomp(0); icomp<numComponents; ++icomp)
      {
        Teuchos::ArrayRCP<ST> nonConstView = mvec->getVectorNonConst(icomp*numLayers+il)->get1dViewNonConst();
        for (GO i = 0; i < numNodes; i++)
          ifile >> nonConstView[i];
      }
    }
    ifile.close();
  }

  Teuchos::broadcast(*comm,0,1,&numVectors);
  if (comm->getRank() != 0)
    mvec = Teuchos::rcp(new Tpetra_MultiVector(map,numVectors));
}

void
Albany::GenericSTKMeshStruct::checkInput(std::string option, std::string value, std::string allowed_values){

#ifdef ALBANY_STK_PERCEPT
      std::vector<std::string> vals = stk::adapt::Util::split(allowed_values, ", ");
      for (unsigned i = 0; i < vals.size(); i++)
        {
          if (vals[i] == value)
            return;
        }

       TEUCHOS_TEST_FOR_EXCEPTION(true,
         std::runtime_error,
         "Adaptation input error in GenericSTKMeshStruct initialization: bar option: " << option << std::endl);
#endif
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");
  validPL->set<std::string>("Exodus Output File Name", "",
      "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Exodus Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
  validPL->set<int>("Exodus Write Interval", 3, "Step interval to write solution data to Exodus file");
  validPL->set<std::string>("NetCDF Output File Name", "",
      "Request NetCDF output to given file name. Requires SEACAS build");
  validPL->set<int>("NetCDF Write Interval", 1, "Step interval to write solution data to NetCDF file");
  validPL->set<int>("NetCDF Output Number of Latitudes", 1,
      "Number of samples in Latitude direction for NetCDF output. Default is 100.");
  validPL->set<int>("NetCDF Output Number of Longitudes", 1,
      "Number of samples in Longitude direction for NetCDF output. Default is 100.");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<std::string>("Cubature Rule", "", "Integration rule sent to Intrepid: GAUSS, GAUSS_RADAU_LEFT, GAUSS_RADAU_RIGHT, GAUSS_LOBATTO");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Use Automatic Aura", false, "Use automatic aura with BulkData");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<std::string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<int>("Element Degree", 1, "Element degree (points per edge - 1) in enriched Aeras mesh");
  validPL->set<bool>("Write Coordinates to MatrixMarket", false, "Writing Coordinates to MatrixMarket File"); //for writing coordinates to matrix market file
  validPL->set<double>("FELIX alpha", 0.0, "Surface boundary inclination for FELIX problems (in degrees)"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<double>("FELIX L", 1, "Domain length for FELIX problems"); //for FELIX problem that require tranformation of STK mesh

  validPL->set<bool>("Contiguous IDs", "true", "Tells Ascii mesh reader is mesh has contiguous global IDs on 1 processor."); //for FELIX problem that require tranformation of STK mesh

  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields,
                     "Fields to pick up from the restart file when restarting");
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layout of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("Residual Vector Components", defaultFields,
      "Names and layout of residual output vector written to Exodus file. Requires SEACAS build");

  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");
  validPL->set<bool>("Transfer Solution to Coordinates", false, "Copies the solution vector to the coordinates for output");

  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the composite tet 10 basis in Intrepid");

  validPL->sublist("Required Fields Info", false, "Info for the creation of the required fields in the STK mesh");

  // Uniform percept adaptation of input mesh prior to simulation

  validPL->set<std::string>("STK Initial Refine", "", "stk::percept refinement option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Enrich", "", "stk::percept enrichment option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Convert", "", "stk::percept conversion option to apply after the mesh is input");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("Number of Refinement Passes", 1, "Number of times to apply the refinement process");

  validPL->sublist("Side Set Discretizations", false, "A sublist containing info for storing side discretizations");

  return validPL;

}
