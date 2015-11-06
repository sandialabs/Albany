//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include "Teuchos_VerboseObject.hpp"
#include "Tpetra_ComputeGatherMap.hpp"

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

  requiresAutomaticAura = params->get<bool>("Use Automatic Aura", false);

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);
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
  Teuchos::Array<std::string> solution_vector =
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", default_solution_vector);

  Teuchos::Array<std::string> default_residual_vector; // Empty
  Teuchos::Array<std::string> residual_vector =
    params->get<Teuchos::Array<std::string> >("Residual Vector Components", default_residual_vector);

  // Build the usual Albany fields unless the user explicitly specifies the residual or solution vector layout
  if(solution_vector.length() == 0 && residual_vector.length() == 0){

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new Albany::OrdinarySTKFieldContainer<true>(params,
            metaData, neq_, req, numDim, sis));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::OrdinarySTKFieldContainer<false>(params,
            metaData, neq_, req, numDim, sis));

  }

  else {

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<true>(params,
            metaData, neq_, req, numDim, sis, solution_vector, residual_vector));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<false>(params,
            metaData, neq_, req, numDim, sis, solution_vector, residual_vector));

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

void Albany::GenericSTKMeshStruct::initializeSideSetMeshStructsExtraction (const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  if (params->isSublist ("Side Set Discretizations"))
  {
    const Teuchos::ParameterList& ssd_list = params->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& sideSets = ssd_list.get<Teuchos::Array<std::string> >("Side Sets");

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    for (int i(0); i<sideSets.size(); ++i)
    {
      const std::string& ss_name = sideSets[i];
      params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(ss_name)));
      if (params_ss->get<std::string>("Method")=="SideSetSTK")
      {
        // The user said this mesh is extracted from a higher dimensional one
        TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1, std::logic_error,
                                    "Error! So far, side set mesh extraction is allowed only from STK meshes with 1 element block.\n");

        this->sideSetMeshStructs[ss_name] = Teuchos::rcp(new Albany::SideSetSTKMeshStruct(*this->meshSpecs[0], params_ss, commT));
      }

      // Update the side set mesh specs pointer in the mesh specs of this mesh
      this->meshSpecs[0]->sideSetMeshSpecs[ss_name] = this->sideSetMeshStructs[ss_name]->getMeshSpecs();
    }
  }
}

void Albany::GenericSTKMeshStruct::finalizeSideSetMeshStructsExtraction (
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

    for (auto it : sideSetMeshStructs)
    {
      Teuchos::RCP<Albany::SideSetSTKMeshStruct> sideMesh;
      sideMesh = Teuchos::rcp_dynamic_cast<Albany::SideSetSTKMeshStruct>(it.second,false);
      if (sideMesh!=Teuchos::null)
      {
        sideMesh->setParentMeshInfo(*this, it.first);

        auto it_req = side_set_req.find(it.first);
        auto it_sis = side_set_sis.find(it.first);

        auto& req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
        auto& sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);

        sideMesh->setFieldAndBulkData(commT,params,neq,req,sis,worksetSize);
      }
    }
  }
}

void Albany::GenericSTKMeshStruct::printParts(stk::mesh::MetaData *metaData){

    std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;

    stk::mesh::PartVector all_parts = metaData->get_parts();

    for (stk::mesh::PartVector::iterator i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part)
    {
       stk::mesh::Part *  part = *i_part ;

       std::cout << "\t" << part->name() << std::endl;

    }

}

void Albany::GenericSTKMeshStruct::loadRequiredInputFields(
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const AbstractFieldContainer::FieldContainerRequirements& req,
          const std::vector<std::string>& missing)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes, elems;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);
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

  // NOTE: the serial map cannot be created linearly, with GIDs from 0 to numGlobalNodes/Elems, since
  //       this may be a boundary mesh, and the GIDs may not start from 0, nor be contiguous.
  //       Therefore, we must create a root map, using the Tpetra utility
  Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Tpetra::Details::computeGatherMap(nodes_map,out);
  Teuchos::RCP<const Tpetra_Map> serial_elems_map = Tpetra::Details::computeGatherMap(elems_map,out);

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
      *out << "Warning! No field type specified for field " << *it << ". We skip it and hope this does not cause problems. Note: you can list it as 'Output' if it's not meant to be an input field.\n";
      continue;
    }
    else if (ftype=="Output")
    {
      *out << "Skipping field " << *it << " since it's listed as output (computed at run time). Make sure there's an evaluator set to save it.\n";
      continue;
    }

    stk::mesh::Entity node, elem;
    stk::mesh::EntityId nodeId, elemId;
    int lid;
    double* values;

    typedef AbstractSTKFieldContainer::ScalarFieldType  SFT;
    typedef AbstractSTKFieldContainer::VectorFieldType  VFT;

    // Depending on the input field type, we need to use different pointers
    if (ftype == "Node Scalar")
    {
      // Creating the serial and (possibly) parallel Tpetra service vectors
      Tpetra_Vector serial_req_vec(serial_nodes_map);
      Tpetra_Vector req_vec(nodes_map);

      temp_str = *it + " Value";
      if (req_fields_info->isParameter(temp_str))
      {
        *out << "Discarding other info about Node Scalar field " << *it << " and filling it with constant value " << req_fields_info->get<double>(temp_str) << "\n";
        // For debug, we allow to fill the field with a given uniform value
        serial_req_vec.putScalar(req_fields_info->get<double>(temp_str));
      }
      else if (fname!="")
      {
        *out << "Reading Node Scalar field " << *it << " from file " << fname << "\n";
        // Read the input file and stuff it in the Tpetra vector
        readScalarFileSerial (fname,serial_req_vec,commT);

        temp_str = *it + " Scale Factor";
        if (req_fields_info->isParameter(temp_str))
        {
          double scale_factor = req_fields_info->get<double>(temp_str);
          serial_req_vec.scale (scale_factor);
        }
      }
      else
      {
        bool found = false;
        for (int i(0); i<missing.size(); ++i)
        {
          if (missing[i]==*it)
          {
            *out << "No file name nor constant value specified for Node Scalar field " << *it << "; initializing it to 0.\n";
            serial_req_vec.putScalar(0.);
            found = true;
            break;
          }
        }
        if (!found)
        {
          *out << "No file name nor constant value specified for Node Scalar field " << *it << "; crossing our fingers and hoping it's already in the mesh...\n";
          continue;
        }
      }

      // Fill the (possibly) parallel vector
      req_vec.doImport(serial_req_vec,importOperatorNode,Tpetra::INSERT);
      Teuchos::ArrayRCP<const ST> req_vec_view = req_vec.get1dView();

      SFT* field = metaData->get_field<SFT> (stk::topology::NODE_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is not 'Node Scalar'?).\n");

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<nodes.size(); ++i)
      {
        values = stk::mesh::field_data(*field, nodes[i]);

        nodeId    = bulkData->identifier(nodes[i]) - 1;
        lid       = nodes_map->getLocalElement((GO)(nodeId));
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
        double value = req_fields_info->get<double>(temp_str);
        *out << "Discarding other info about Elem Scalar field " << *it << " and filling it with constant value " << value << "\n";
        // For debug, we allow to fill the field with a given uniform value
        serial_req_vec.putScalar(value);
      }
      else if (fname!="")
      {
        *out << "Reading Elem Scalar field " << *it << " from file " << fname << "\n";
        // Read the input file and stuff it in the Tpetra vector
        readScalarFileSerial (fname,serial_req_vec,commT);

        temp_str = *it + " Scale Factor";
        if (req_fields_info->isParameter(temp_str))
        {
          double scale_factor = req_fields_info->get<double>(temp_str);
          serial_req_vec.scale (scale_factor);
        }
      }
      else
      {
        bool found = false;
        for (int i(0); i<missing.size(); ++i)
        {
          if (missing[i]==*it)
          {
            *out << "No file name nor constant value specified for Elem Scalar field " << *it << "; initializing it to 0.\n";
            serial_req_vec.putScalar(0.);
            found = true;
            break;
          }
        }
        if (!found)
        {
          *out << "No file name nor constant value specified for Elem Scalar field " << *it << "; crossing our fingers and hoping it's already in the mesh...\n";
          continue;
        }
      }

      // Fill the (possibly) parallel vector
      req_vec.doImport(serial_req_vec,importOperatorElem,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector view
      SFT* field = metaData->get_field<SFT>(stk::topology::ELEM_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is not 'Elem Scalar'?).\n");

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
        Teuchos::Array<double> values = req_fields_info->get<Teuchos::Array<double> >(temp_str);
        TEUCHOS_TEST_FOR_EXCEPTION (values.size()!=serial_req_mvec.getNumVectors(), Teuchos::Exceptions::InvalidParameter,
                                    "Error! The given uniform value vector size does not match the field dimension.\n");

        *out << "Discarding other info about Node Vector field " << *it << " and filling it with constant values " << values << "\n";
        // For debug, we allow to fill the field with a given uniform value
        for (int iv(0); iv<serial_req_mvec.getNumVectors(); ++iv)
        {
          serial_req_mvec.getVectorNonConst(iv)->putScalar(values[iv]);
        }
      }
      else if (fname!="")
      {
        *out << "Reading Node Vector field " << *it << " from file " << fname << "\n";
        // Read the input file and stuff it in the Tpetra multivector
        readVectorFileSerial (fname,serial_req_mvec,commT);

        temp_str = *it + " Scale Factor";
        if (req_fields_info->isParameter(temp_str))
        {
          Teuchos::Array<double> scale_factors = req_fields_info->get<Teuchos::Array<double> >(temp_str);
          TEUCHOS_TEST_FOR_EXCEPTION (scale_factors.size()!=serial_req_mvec.getNumVectors(), Teuchos::Exceptions::InvalidParameter,
                                      "Error! The given uniform value vector size does not match the field dimension.\n");
          serial_req_mvec.scale (scale_factors);
        }
      }
      else
      {
        bool found = false;
        for (int i(0); i<missing.size(); ++i)
        {
          if (missing[i]==*it)
          {
            *out << "No file name nor constant value specified for Node Vector field " << *it << "; initializing it to 0.\n";
            serial_req_mvec.putScalar(0.);
            found = true;
            break;
          }
        }
        if (!found)
        {
          *out << "No file name nor constant value specified for Node Vector field " << *it << "; crossing our fingers and hoping it's already in the mesh...\n";
          continue;
        }
      }

      // Fill the (possibly) parallel vector
      req_mvec.doImport(serial_req_mvec,importOperatorNode,Tpetra::INSERT);
      std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;

      // Extracting the mesh field (we still don't know if the field is node or cell oriented)
      VFT* field = metaData->get_field<VFT> (stk::topology::NODE_RANK, *it);

      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is not 'Node Vector'?).\n");

      for (int i(0); i<fieldDim; ++i)
        req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<nodes.size(); ++i)
      {
        values = stk::mesh::field_data(*field, nodes[i]);

        nodeId = bulkData->identifier(nodes[i]) - 1;
        lid    = nodes_map->getLocalElement((GO)(nodeId));
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
        Teuchos::Array<double> values = req_fields_info->get<Teuchos::Array<double> >(temp_str);
        TEUCHOS_TEST_FOR_EXCEPTION (values.size()!=serial_req_mvec.getNumVectors(), Teuchos::Exceptions::InvalidParameter,
                                    "Error! The given uniform value vector size does not match the field dimension.\n");

        *out << "Discarding other info about Elem Vector field " << *it << " and filling it with constant values " << values << "\n";
        // For debug, we allow to fill the field with a given uniform value
        for (int iv(0); iv<serial_req_mvec.getNumVectors(); ++iv)
        {
          serial_req_mvec.getVectorNonConst(iv)->putScalar(values[iv]);
        }
      }
      else if (fname!="")
      {
        *out << "Reading Elem Vector field " << *it << " from file " << fname << "\n";
        // Read the input file and stuff it in the Tpetra multivector
        readVectorFileSerial (fname,serial_req_mvec,commT);

        temp_str = *it + " Scale Factor";
        if (req_fields_info->isParameter(temp_str))
        {
          Teuchos::Array<double> scale_factors = req_fields_info->get<Teuchos::Array<double> >(temp_str);
          TEUCHOS_TEST_FOR_EXCEPTION (scale_factors.size()!=serial_req_mvec.getNumVectors(), Teuchos::Exceptions::InvalidParameter,
                                      "Error! The given uniform value vector size does not match the field dimension.\n");
          serial_req_mvec.scale (scale_factors);
        }
      }
      else
      {
        bool found = false;
        for (int i(0); i<missing.size(); ++i)
        {
          if (missing[i]==*it)
          {
            *out << "No file name nor constant value specified for Elem Vector field " << *it << "; initializing it to 0.\n";
            serial_req_mvec.putScalar(0.);
            found = true;
            break;
          }
        }
        if (!found)
        {
          *out << "No file name nor constant value specified for Elem Vector field " << *it << "; crossing our fingers and hoping it's already in the mesh...\n";
          continue;
        }
      }

      // Fill the (possibly) parallel vector
      req_mvec.doImport(serial_req_mvec,importOperatorElem,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector views
      VFT* field = metaData->get_field<VFT>(stk::topology::ELEM_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field " << *it << " not present (perhaps is not 'Elem Vector'?).\n");
      std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;
      for (int i(0); i<fieldDim; ++i)
        req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<elems.size(); ++i)
      {
        elemId = bulkData->identifier(elems[i]) - 1;
        lid    = elems_map->getLocalElement((GO)(elemId));

        values = stk::mesh::field_data(*field, nodes[i]);

        for (int iDim(0); iDim<fieldDim; ++iDim)
          values[iDim] = req_mvec_view[iDim][lid];
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Sorry, I haven't yet implemented the case of field that are not Scalar nor Vector or that is not defined at nodes nor elements.\n");
    }
  }
}

void Albany::GenericSTKMeshStruct::readScalarFileSerial (std::string& fname, Tpetra_MultiVector& content,
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
                                  std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << content.getLocalLength() << ")" << std::endl);

      for (GO i = 0; i < numNodes; i++)
        ifile >> content_nonConstView[i];

      ifile.close();
    }
    else
    {
      std::cout << "Warning in GenericSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
}

void Albany::GenericSTKMeshStruct::readVectorFileSerial (std::string& fname,Tpetra_MultiVector& contentVec,
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
                                  std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << contentVec.getLocalLength() << ")" << std::endl);
      TEUCHOS_TEST_FOR_EXCEPTION(numComponents != contentVec.getNumVectors(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of components in file " << fname << " (" << numComponents << ") is different from the number expected (" << contentVec.getNumVectors() << ")" << std::endl);

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
      std::cout << "Warning in GenericSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
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

  return validPL;

}
