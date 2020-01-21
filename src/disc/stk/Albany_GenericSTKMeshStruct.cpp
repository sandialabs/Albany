//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include "Teuchos_VerboseObject.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_KokkosTypes.hpp"
#include "Albany_Gather.hpp"

#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CreateAdjacentEntities.hpp>

#include <Albany_STKNodeSharing.hpp>
#include <Albany_ThyraUtils.hpp>
#include <Albany_CombineAndScatterManager.hpp>
#include <Albany_GlobalLocalIndexer.hpp>

// Expression reading
#ifdef ALBANY_PANZER_EXPR_EVAL
#include <Panzer_ExprEval_impl.hpp>
#endif

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

  for (unsigned int j = 0; j < del_relations.size(); ++j) {
    stk::mesh::Entity ent = del_relations[j];
    mesh.destroy_relation(ent,ent,del_ids[j]);
  }

  del_relations.clear();
  del_ids.clear();
}

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

}

namespace Albany
{

GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
    int numDim_)
    : buildEMesh(false),
      params(params_),
      adaptParams(adaptParams_),
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
  ignore_side_maps = false;
}

void GenericSTKMeshStruct::SetupFieldData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const int neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<StateInfoStruct>& sis,
    const int worksetSize)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!metaData->is_initialized(),
       std::logic_error,
       "LogicError: metaData->FEM_initialize(numDim) not yet called" << std::endl);

  neq = neq_;

  this->nodal_data_base = sis->getNodalDataBase();

  if (bulkData.is_null()) {
     const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get());
     stk::mesh::BulkData::AutomaticAuraOption auto_aura_option = stk::mesh::BulkData::NO_AUTO_AURA;
     if(requiresAutomaticAura) auto_aura_option = stk::mesh::BulkData::AUTO_AURA;
     bulkData = Teuchos::rcp(
       new stk::mesh::BulkData(*metaData,
                               *mpiComm->getRawMpiComm(),
                               auto_aura_option,
                               //worksetSize, // capability currently removed from STK_Mesh
                               false, // add_fmwk_data
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
        this->fieldContainer = Teuchos::rcp(new MultiSTKFieldContainer<true>(params,
            metaData, bulkData, neq, req, numDim, sis, solution_vector, residual_vector));
      else
        this->fieldContainer = Teuchos::rcp(new MultiSTKFieldContainer<false>(params,
            metaData, bulkData, neq, req, numDim, sis, solution_vector, residual_vector));

  }

  else {

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer<true>(params,
            metaData, bulkData, neq, req, numDim, sis));
      else
        this->fieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer<false>(params,
            metaData, bulkData, neq, req, numDim, sis));

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


  //get the type of transformation of STK mesh
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh
  felixAlpha = params->get("LandIce alpha", 0.0); //for LandIce problems
  felixL = params->get("LandIce L", 1.0); //for LandIce problems
  xShift = params->get("x-shift", 0.0);
  yShift = params->get("y-shift", 0.0);
  zShift = params->get("z-shift", 0.0);
  betas_BLtransform = params->get<Teuchos::Array<double> >("Betas BL Transform",  Teuchos::tuple<double>(0.0, 0.0, 0.0));

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

void GenericSTKMeshStruct::setAllPartsIO()
{
#ifdef ALBANY_SEACAS
  for (auto& it : partVec)
  {
    stk::mesh::Part& part = *it.second;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
  for (auto& it : nsPartVec)
  {
    stk::mesh::Part& part = *it.second;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
  for (auto& it : ssPartVec)
  {
    stk::mesh::Part& part = *it.second;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
#endif
}

bool GenericSTKMeshStruct::buildPerceptEMesh(){

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

bool GenericSTKMeshStruct::buildUniformRefiner(){

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

bool GenericSTKMeshStruct::buildLocalRefiner(){

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
GenericSTKMeshStruct::cullSubsetParts(std::vector<std::string>& ssNames,
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

Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> >&
GenericSTKMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

const Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> >&
GenericSTKMeshStruct::getMeshSpecs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int GenericSTKMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
    // compute numWorksets, and shrink workset size to minimize padding
    const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
    return (1 + (ebSizeMax-1) / numWorksets);
  }
}

void GenericSTKMeshStruct::computeAddlConnectivity()
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
    for (unsigned int i = 0; i < element_lst.size(); ++i){
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

      for (unsigned int i = 0; i < face_lst.size(); ++i) {
        stk::mesh::Entity face = face_lst[i];

        only_keep_connectivity_to_specified_ranks(*bulkData, face, stk::topology::ELEMENT_RANK, stk::topology::EDGE_RANK);
      }
    }

    fix_node_sharing(*bulkData);
    bulkData->modification_end();
  }

}

void GenericSTKMeshStruct::setDefaultCoordinates3d ()
{
  // If the mesh is already a 3d mesh, coordinates_field==coordinates_field3d
  if (this->numDim==3) return;

  // We make coordinates_field3d store the same coordinates as coordinates_field,
  // padding the vector of coordinates with zeros

  std::vector<stk::mesh::Entity> nodes;
  stk::mesh::get_entities(*bulkData,stk::topology::NODE_RANK,nodes);
  double* values;
  double* values3d;
  for (auto node : nodes)
  {
    values3d = stk::mesh::field_data(*this->getCoordinatesField3d(), node);
    values   = stk::mesh::field_data(*this->getCoordinatesField(), node);

    for (int iDim=0; iDim<numDim; ++iDim) {
      values3d[iDim] = values[iDim];
    }
    for (int iDim=numDim; iDim<3; ++iDim) {
      values3d[iDim] = 0.0;
    }
  }
}

void GenericSTKMeshStruct::uniformRefineMesh(const Teuchos::RCP<const Teuchos_Comm>& comm){
#ifdef ALBANY_STK_PERCEPT
// Refine if requested
  if(!uniformRefinementInitialized) return;

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();


  if(!refinerPattern.is_null() && proc_rank_field){

    stk::adapt::UniformRefiner refiner(*eMesh, *refinerPattern, proc_rank_field);

    int numRefinePasses = params->get<int>("Number of Refinement Passes", 1);

    for(int pass = 0; pass < numRefinePasses; pass++){

      if(comm->getRank() == 0)
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
#else
  // Silence compiler warnings
  (void) comm;
#endif
}

void GenericSTKMeshStruct::rebalanceInitialMeshT(const Teuchos::RCP<const Teuchos::Comm<int> >& comm){
  bool rebalance = params->get<bool>("Rebalance Mesh", false);
  bool useSerialMesh = params->get<bool>("Use Serial Mesh", false);

  if(rebalance || (useSerialMesh && comm->getSize() > 1)) {
    rebalanceAdaptedMeshT(params, comm);
  }
}

void GenericSTKMeshStruct::
rebalanceAdaptedMeshT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
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

    stk::rebalance::Zoltan zoltan_partitiona(getMpiCommFromEpetraComm(*comm), numDim, graph);

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

#else
  // Silence compiler warnings
  (void) params_;
  (void) comm;
#endif  //ALBANY_ZOLTAN
}

void GenericSTKMeshStruct::addNodeSetsFromSideSets ()
{
  TEUCHOS_TEST_FOR_EXCEPTION (this->meshSpecs[0]==Teuchos::null, std::runtime_error,
                              "Error! Mesh specs have not been initialized yet.\n");

  // This function adds a (sideset) part to nsPartVec and to the meshSpecs (nsNames)
  for (const auto& ssn_part_pair : ssPartVec)
  {
    // If a nodeset with the same name already exists, we ASSUME it contains this sideset's nodes.
    auto itns = nsPartVec.find(ssn_part_pair.first);
    if (itns!=nsPartVec.end())
      return;

    // Add the part to the node sets parts vector
    stk::mesh::Part* part = ssn_part_pair.second;
    nsPartVec[ssn_part_pair.first] = part;

    // Update the list of nodesets in the mesh specs
    this->meshSpecs[0]->nsNames.push_back(ssn_part_pair.first);

    // This list will be used later to check that the new nodesets' integrity
    m_nodesets_from_sidesets.push_back(ssn_part_pair.first);
  }
}

void GenericSTKMeshStruct::checkNodeSetsFromSideSetsIntegrity ()
{
  // For each nodeset generated from a sideset, this method checks that
  // the sides in the corresponding nodeset contain the right number of
  // nodes, that is, makes sure that 'declare_relation' was called to
  // establish the relation between the side and its node.

  for (auto ssn : m_nodesets_from_sidesets)
  {
    // Fetch the part
    auto it = ssPartVec.find(ssn);
    TEUCHOS_TEST_FOR_EXCEPTION (it==ssPartVec.end(), std::runtime_error,
                                "Error! Side set " << ssn << " not found. This error should NEVER occurr though. Bug?\n");
    stk::mesh::Part* ssPart = it->second;

    // Extract sides
    stk::mesh::Selector selector (*ssPart & (metaData->locally_owned_part() | metaData->globally_shared_part()));
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(metaData->side_rank()), sides);

    // For each side, we check that it has the right number of nodes.
    unsigned num_nodes = metaData->get_topology(*ssPart).num_nodes();
    for (const auto& side : sides)
    {
      TEUCHOS_TEST_FOR_EXCEPTION (bulkData->num_nodes(side)==num_nodes, std::runtime_error,
                                  "Error! Found a side with wrong number of nodes stored. Most likely,"
                                  "its nodes were not added to the side with 'declare_relation').\n");
    }
  }
}

void GenericSTKMeshStruct::initializeSideSetMeshSpecs (const Teuchos::RCP<const Teuchos_Comm>& comm) {
  // Loop on all mesh specs
  for (auto ms: this->getMeshSpecs() ) {
    // Loop on all side sets of the mesh
    for (auto ssName : ms->ssNames) {
      // Get the part
      stk::mesh::Part* part = metaData->get_part(ssName);
      TEUCHOS_TEST_FOR_EXCEPTION (part==nullptr, std::runtime_error, "Error! One of the stored meshSpecs claims to have sideset " + ssName +
                                                                     " which, however, is not a part of the mesh.\n");
      stk::topology stk_topo_data = metaData->get_topology( *part );
      shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data); 
      const auto* ctd = shards_ctd.getCellTopologyData();

      auto& ss_ms = ms->sideSetMeshSpecs[ssName];

      // At this point, we cannot assume there will be a discretization on this side set, so we use cubature degree=-1,
      // and the workset size of this mesh. If the user *does* add a discretization (in the Side Set Discretizations sublist),
      // he/she can specify cubature and workset size there. The method initializeSideSetMeshStructs will overwrite
      // this mesh specs anyways.
      // Note: it *may* be that the user need no cubature on this side (only node/cell fields).
      //       But if the user *does* nned cubature, we want to set a *very wrong* number, so that
      //       the code will crash somewhere, and he/sh can realize he/she needs to set cubature info somewhere

      // We allow a null ctd here, and we simply do not store the side mesh specs.
      // The reason is that we _may_ be loading a mesh that stores some empty side sets.
      // If we are using the sideset, we will probably run into some sort of errors later,
      // unless we are specifying a side discretization (which will be created _after_ this function call).
      if (ctd==nullptr) {
        Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        out->setProcRankAndSize(comm->getRank(), comm->getSize());
        out->setOutputToRootOnly(0);
        *out << "Warning! Side set '" << ssName << "' does not store a valid cell topology.\n";

        continue;
      }
      
      ss_ms.resize(1);
      ss_ms[0] = Teuchos::rcp( new MeshSpecsStruct() );
      ss_ms[0]->ctd = *ctd;
      ss_ms[0]->numDim = this->numDim-1;
    }
  }
}

void GenericSTKMeshStruct::initializeSideSetMeshStructs (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  if (params->isSublist ("Side Set Discretizations")) {
    const Teuchos::ParameterList& ssd_list = params->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& sideSets = ssd_list.get<Teuchos::Array<std::string> >("Side Sets");

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    int sideDim = this->numDim - 1;
    for (int i(0); i<sideSets.size(); ++i) {
      Teuchos::RCP<AbstractMeshStruct> ss_mesh;
      const std::string& ss_name = sideSets[i];
      params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(ss_name)));
      if (!params_ss->isParameter("Number Of Time Derivatives"))
        params_ss->set<int>("Number Of Time Derivatives",num_time_deriv);

      std::string method = params_ss->get<std::string>("Method");
      if (method=="SideSetSTK") {
        // The user said this mesh is extracted from a higher dimensional one
        TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1, std::logic_error,
                                    "Error! So far, side set mesh extraction is allowed only from STK meshes with 1 element block.\n");

        this->sideSetMeshStructs[ss_name] = Teuchos::rcp(new SideSetSTKMeshStruct(*this->meshSpecs[0], params_ss, comm));
      } else {
        // We must check whether a side mesh was already created elsewhere.
        // If the mesh already exists, we do nothing, and we ASSUME it is a valid mesh
        // This happens, for instance, for the basal mesh for extruded meshes.
        if (this->sideSetMeshStructs.find(ss_name)==this->sideSetMeshStructs.end())
        {
          ss_mesh = DiscretizationFactory::createMeshStruct (params_ss,adaptParams,comm);
          this->sideSetMeshStructs[ss_name] = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(ss_mesh,false);
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
      this->meshSpecs[0]->sideSetMeshNames.push_back(ss_name);

      // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
      typedef AbstractSTKFieldContainer::IntScalarFieldType ISFT;
      ISFT* side_to_cell_map = &this->sideSetMeshStructs[ss_name]->metaData->declare_field<ISFT> (stk::topology::ELEM_RANK, "side_to_cell_map");
      stk::mesh::put_field_on_mesh(*side_to_cell_map, this->sideSetMeshStructs[ss_name]->metaData->universal_part(), 1, nullptr);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*side_to_cell_map, Ioss::Field::TRANSIENT);
#endif
      // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
      const int num_nodes = sideSetMeshStructs[ss_name]->getMeshSpecs()[0]->ctd.node_count;
      typedef AbstractSTKFieldContainer::IntVectorFieldType IVFT;
      IVFT* side_nodes_ids = &this->sideSetMeshStructs[ss_name]->metaData->declare_field<IVFT> (stk::topology::ELEM_RANK, "side_nodes_ids");
      stk::mesh::put_field_on_mesh(*side_nodes_ids, this->sideSetMeshStructs[ss_name]->metaData->universal_part(), num_nodes, nullptr);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*side_nodes_ids, Ioss::Field::TRANSIENT);
#endif

      // If requested, we ignore the side maps already stored in the imported side mesh (if any)
      // This can be useful for side mesh of an extruded mesh, in the case it was constructed
      // as side mesh of an extruded mesh with a different ordering and/or different number
      // of layers. Notice that if that's the case, it probalby is impossible to build a new
      // set of maps, since there is no way to correctly map the side nodes to the cell nodes.
      this->sideSetMeshStructs[ss_name]->ignore_side_maps = params_ss->get<bool>("Ignore Side Maps", false);
    }
  }
}

void GenericSTKMeshStruct::finalizeSideSetMeshStructs (
          const Teuchos::RCP<const Teuchos_Comm>& comm,
          const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req,
          const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis,
          int worksetSize)
{
  if (this->sideSetMeshStructs.size()>0) {
    // Dummy sis/req if not present in the maps for a given side set.
    // This could happen if the side discretization has no requirements/states
    Teuchos::RCP<StateInfoStruct> dummy_sis = Teuchos::rcp(new StateInfoStruct());
    dummy_sis->createNodalDataBase();
    AbstractFieldContainer::FieldContainerRequirements dummy_req;

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    const Teuchos::ParameterList& ssd_list = params->sublist("Side Set Discretizations");
    for (auto it : sideSetMeshStructs) {
      Teuchos::RCP<SideSetSTKMeshStruct> sideMesh;
      sideMesh = Teuchos::rcp_dynamic_cast<SideSetSTKMeshStruct>(it.second,false);
      if (sideMesh!=Teuchos::null) {
        // SideSetSTK mesh need to build the mesh
        sideMesh->setParentMeshInfo(*this, it.first);
      }

      // We check since the basal mesh for extruded stk mesh should already have it set
      if (!it.second->fieldAndBulkDataSet) {
        auto it_req = side_set_req.find(it.first);
        auto it_sis = side_set_sis.find(it.first);

        auto& req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
        auto& sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);

        params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(it.first)));
        it.second->setFieldAndBulkData(comm,params_ss,neq,req,sis,worksetSize);  // Cell equations are also defined on the side, but not viceversa
      }
    }
  }
}

void GenericSTKMeshStruct::
buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                std::map<GO,GO>& sideMap,
                                std::map<GO,std::vector<int>>& sideNodeMap)
{
  TEUCHOS_TEST_FOR_EXCEPTION (sideSetMeshStructs.find(sideSetName)==sideSetMeshStructs.end(), Teuchos::Exceptions::InvalidParameter,
                              "Error in 'buildSideNodeToSideSetCellNodeMap': side set " << sideSetName << " does not have a mesh.\n");

  Teuchos::RCP<AbstractSTKMeshStruct> side_mesh = sideSetMeshStructs.at(sideSetName);

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

  if (cells2D.size()==0) {
    // It can happen if the mesh is partitioned and this process does not own the side
    return;
  }

  const stk::topology::rank_t SIDE_RANK = metaData->side_rank();
  const int num_nodes = side_mesh->bulkData->num_nodes(cells2D[0]);
  int* cell3D_id;
  int* side_nodes_ids;
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
  if (side_mesh->side_maps_present && !side_mesh->ignore_side_maps) {
    // This mesh was loaded from a file that stored the side maps.
    // Hence, we just read it and stuff the map with it
    // WARNING: the maps may be not be valid. This can happen if they were built with an
    //          extruded mesh with N layers, and we are now using this side mesh to build
    //          an extruded mesh with M!=N layers, or with different ordering (COLUMN/LAYER).
    //          If this is the case, you must first edit the exodus file and delete the maps
    //          or set 'Ignore Side Maps' to true in the input file
    for (const auto& cell2D : cells2D) {
      // Get the stk field data
      cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
      side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);
      stk::mesh::Entity cell3D = bulkData->get_entity(stk::topology::ELEM_RANK,*cell3D_id);

      num_sides = bulkData->num_sides(cell3D);
      const stk::mesh::Entity* cell_sides = bulkData->begin(cell3D,SIDE_RANK);
      side_lid = -1;
      for (int iside(0); iside<num_sides; ++iside) {
        side3D_nodes = bulkData->begin_nodes(cell_sides[iside]);
        for (int inode(0); inode<num_nodes; ++inode) {
          side3D_nodes_ids[inode] = bulkData->identifier(side3D_nodes[inode]);
        }
        if (std::is_permutation(side3D_nodes_ids.begin(),side3D_nodes_ids.end(), side_nodes_ids)) {
          side_lid = iside;
          side3D_GID = bulkData->identifier(cell_sides[iside])-1;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION (side_lid==-1, std::logic_error, "Error! Cannot locate the right side in the cell.\n");

      sideMap[side3D_GID] = side_mesh->bulkData->identifier(cell2D)-1;
      sideNodeMap[side3D_GID].resize(num_nodes);
      cell2D_nodes = side_mesh->bulkData->begin_nodes(cell2D);
      for (int i(0); i<num_nodes; ++i) {
        auto it = std::find(side3D_nodes_ids.begin(),side3D_nodes_ids.end(),side_nodes_ids[i]);
        sideNodeMap[side3D_GID][std::distance(side3D_nodes_ids.begin(),it)] = i;
      }
    }

    return;
  }

  const stk::mesh::Entity* side_cells;
  for (const auto& cell2D : cells2D) {
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
    for (int i(0); i<num_nodes; ++i) {
      cell2D_nodes_ids[i] = side_mesh->bulkData->identifier(cell2D_nodes[i]);
      side3D_nodes_ids[i] = bulkData->identifier(side3D_nodes[i]);
    }

    for (int i(0); i<num_nodes; ++i) {
      auto it = std::find(cell2D_nodes_ids.begin(),cell2D_nodes_ids.end(),side3D_nodes_ids[i]);
      sideNodeMap[side3D_GID][i] = std::distance(cell2D_nodes_ids.begin(),it);
      side_nodes_ids[std::distance(cell2D_nodes_ids.begin(),it)] = side3D_nodes_ids[i];
    }
  }

  // Just in case this method gets called twice
  side_mesh->side_maps_present = true;
}

void GenericSTKMeshStruct::printParts(stk::mesh::MetaData *metaData)
{
  std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;
  stk::mesh::PartVector all_parts = metaData->get_parts();

  for (auto i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part) {
    stk::mesh::Part* part = *i_part ;
    std::cout << "\t" << part->name() << std::endl;
  }
}

void GenericSTKMeshStruct::loadRequiredInputFields (const AbstractFieldContainer::FieldContainerRequirements& req,
                                                            const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(comm->getRank(), comm->getSize());
  out->setOutputToRootOnly(0);

  *out << "[GenericSTKMeshStruct] Processing field requirements...\n";

  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes, elems;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK), elems);

  Teuchos::Array<GO> nodeIndices(nodes.size()), elemIndices(elems.size());
  for (unsigned int i = 0; i < nodes.size(); ++i) {
    nodeIndices[i] = bulkData->identifier(nodes[i]) - 1;
  }
  for (unsigned int i = 0; i < elems.size(); ++i) {
    elemIndices[i] = bulkData->identifier(elems[i]) - 1;
  }

  auto nodes_vs = createVectorSpace(comm,nodeIndices);
  auto elems_vs = createVectorSpace(comm,elemIndices);

  // Check whether we need the serial map or not. The only scenario where we DO need it is if we are
  // loading a field from an ASCII file. So let's check the fields info to see if that's the case.
  Teuchos::ParameterList dummyList;
  Teuchos::ParameterList* req_fields_info;
  if (params->isSublist("Required Fields Info")) {
    req_fields_info = &params->sublist("Required Fields Info");
  } else {
    req_fields_info = &dummyList;
  }
  int num_fields = req_fields_info->get<int>("Number Of Fields",0);
  bool node_field_ascii_loads = false;
  bool elem_field_ascii_loads = false;
  std::string fname, fusage, ftype, forigin;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    Teuchos::ParameterList& fparams = req_fields_info->sublist(ss.str());

    fusage = fparams.get<std::string>("Field Usage", "Input");
    ftype  = fparams.get<std::string>("Field Type","INVALID");
    if (fusage == "Input" || fusage == "Input-Output") {
      forigin = fparams.get<std::string>("Field Origin","INVALID");
      if (forigin=="File" && fparams.isParameter("File Name")) {
        if (ftype.find("Node")!=std::string::npos) {
          node_field_ascii_loads = true;
        } else if (ftype.find("Elem")!=std::string::npos) {
          elem_field_ascii_loads = true;
        }
      }
    }
  }

  // NOTE: the serial vs cannot be created linearly, with GIDs from 0 to numGlobalNodes/Elems, since
  //       this may be a boundary mesh, and the GIDs may not start from 0, nor be contiguous.
  //       Therefore, we must create a root vs. Moreover, we need the GIDs sorted (so that, regardless
  //       of the GID, we read the serial input files in the correct order), and we can't sort them
  //       once the vs is created.

  auto serial_nodes_vs = nodes_vs;
  auto serial_elems_vs = elems_vs;
  if (node_field_ascii_loads) {
    Teuchos::Array<GO> nodes_gids = getGlobalElements(nodes_vs);
    Teuchos::Array<GO> all_nodes_gids;
    gatherV(comm,nodes_gids(),all_nodes_gids,0);
    std::sort(all_nodes_gids.begin(),all_nodes_gids.end());
    auto it = std::unique(all_nodes_gids.begin(),all_nodes_gids.end());
    all_nodes_gids.erase(it,all_nodes_gids.end());
    serial_nodes_vs = createVectorSpace(comm,all_nodes_gids);
  }
  if (elem_field_ascii_loads) {
    Teuchos::Array<GO> elems_gids = getGlobalElements(elems_vs);
    Teuchos::Array<GO> all_elems_gids;
    gatherV(comm,elems_gids(),all_elems_gids,0);
    std::sort(all_elems_gids.begin(),all_elems_gids.end());
    serial_elems_vs = createVectorSpace(comm,all_elems_gids);
  }

  // Creating the combine and scatter manager object (to transfer from serial to parallel vectors)
  auto cas_manager_node = createCombineAndScatterManager(serial_nodes_vs,nodes_vs);
  auto cas_manager_elem = createCombineAndScatterManager(serial_elems_vs,elems_vs);

  std::set<std::string> missing;
  for (auto rname : req) {
    missing.insert(rname);
  }

  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    Teuchos::ParameterList& fparams = req_fields_info->sublist(ss.str());

    // First, get the name and usage of the field, and check if it's used
    fname = fparams.get<std::string>("Field Name");
    fusage = fparams.get<std::string>("Field Usage", "Input");
    if (fusage == "Unused") {
      *out << "  - Skipping field '" << fname << "' since it's listed as unused.\n";
      continue;
    }

    // The field is used somehow. Check that it is present in the mesh
    ftype = fparams.get<std::string>("Field Type","INVALID");
    checkFieldIsInMesh(fname, ftype);
    missing.erase(fname);

    // Check if it's an output file (nothing to be done then). If not, check that the usage is a valid string
    if (fusage == "Output") {
      *out << "  - Skipping field '" << fname << "' since it's listed as output. Make sure there's an evaluator set to save it!\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (fusage!="Input" && fusage!="Input-Output", Teuchos::Exceptions::InvalidParameter,
                                  "Error! 'Field Usage' for field '" << fname << "' must be one of 'Input', 'Output', 'Input-Output' or 'Unused'.\n");
    }

    // Ok, it's an input (or input-output) field. Find out where the field comes from
    forigin = fparams.get<std::string>("Field Origin","INVALID");
    if (forigin=="Mesh") {
      *out << "  - Skipping field '" << fname << "' since it's listed as present in the mesh.\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (forigin!="File", Teuchos::Exceptions::InvalidParameter,
                                  "Error! 'Field Origin' for field '" << fname << "' must be one of 'File' or 'Mesh'.\n");
    }

    // The field is not already present (with updated values) in the mesh, and must be loaded/computed filled here.

    // Detect load type
    bool load_ascii = fparams.isParameter("File Name");
    bool load_math_expr = fparams.isParameter("Field Expression");
    bool load_value = fparams.isParameter("Field Value") || fparams.isParameter("Random Value");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_ascii && load_math_expr, std::logic_error, "Error! You cannot specify both 'File Name' and 'Field Expression' for loading a field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_ascii && load_value,     std::logic_error, "Error! You cannot specify both 'File Name' and 'Field Value' (or 'Random Value') for loading a field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_math_expr && load_value, std::logic_error, "Error! You cannot specify both 'Field Expression' and 'Field Value' (or 'Random Value') for loading a field.\n");

    // Depending on the input field type, we need to use different pointers/importers/vectors
    bool nodal, scalar, layered;
    Teuchos::RCP<CombineAndScatterManager> cas_manager;
    std::vector<stk::mesh::Entity>* entities;
    if (ftype == "Node Scalar") {
      nodal = true; scalar = true; layered = false;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Scalar") {
      nodal = false; scalar = true; layered = false;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Vector") {
      nodal = true; scalar = false; layered = false;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Vector") {
      nodal = false; scalar = false; layered = false;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Layered Scalar") {
      nodal = true; scalar = true; layered = true;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Layered Scalar") {
      nodal = false; scalar = true; layered = true;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Layered Vector") {
      nodal = true; scalar = false; layered = true;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Layered Vector") {
      nodal = false; scalar = false; layered = true;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Field '" << fname << "' has type '" << ftype << "'.\n" <<
                                  "Unfortunately, the only supported field types so fare are 'Node/Elem Scalar/Vector' and 'Node/Elem Layered Scalar/Vector'.\n");
    }

    auto serial_vs = cas_manager->getOwnedVectorSpace();
    auto vs = cas_manager->getOverlappedVectorSpace();  // It is not overlapped, it is just distributed.
    Teuchos::RCP<Thyra_MultiVector> field_mv;
    if (load_ascii) {
      loadField (fname, fparams, field_mv, *cas_manager, comm, nodal, scalar, layered, out);
    } else if (load_value) {
      fillField (fname, fparams, field_mv, vs, nodal, scalar, layered, out);
    } else if (load_math_expr) {
      computeField (fname, fparams, field_mv, vs, *entities, nodal, scalar, layered, out);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! No means were specified for loading field '" + fname + "'.\n");
    }

    auto field_mv_view = getLocalData(field_mv.getConst());

    //Now we have to stuff the vector in the mesh data
    typedef AbstractSTKFieldContainer::ScalarFieldType  SFT;
    typedef AbstractSTKFieldContainer::VectorFieldType  VFT;
    typedef AbstractSTKFieldContainer::TensorFieldType  TFT;

    SFT* scalar_field = 0;
    VFT* vector_field = 0;
    TFT* tensor_field = 0;

    stk::topology::rank_t entity_rank = nodal ? stk::topology::NODE_RANK : stk::topology::ELEM_RANK;

    if (scalar && !layered) {
      // Purely scalar field
      scalar_field = metaData->get_field<SFT> (entity_rank, fname);
    } else if ( scalar==layered ) {
      // Either (non-layered) vector or layered scalar field
      vector_field = metaData->get_field<VFT> (entity_rank, fname);
    } else {
      // Layered vector field
      tensor_field = metaData->get_field<TFT> (entity_rank, fname);
    }

    TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0 && vector_field==0 && tensor_field==0, std::logic_error,
                                "Error! Field " << fname << " not present (perhaps is not '" << ftype << "'?).\n");

    stk::mesh::EntityId gid;
    LO lid;
    double* values;
    auto indexer = createGlobalLocalIndexer(vs);
    for (unsigned int i(0); i<entities->size(); ++i) {
      if (scalar_field!=0) {
        values = stk::mesh::field_data(*scalar_field, (*entities)[i]);
      } else if (vector_field!=0) {
        values = stk::mesh::field_data(*vector_field, (*entities)[i]);
      } else {
        values = stk::mesh::field_data(*tensor_field, (*entities)[i]);
      }

      gid = bulkData->identifier((*entities)[i]) - 1;
      lid = indexer->getLocalElement(GO(gid));
      for (int iDim(0); iDim<field_mv_view.size(); ++iDim) {
        values[iDim] = field_mv_view[iDim][lid];
      }
    }
  }

  if (missing.size()>0) {
    std::string missing_list;
    for (auto i : missing) {
      missing_list += " '" + i + "',";
    }
    missing_list.erase(missing_list.size()-1);

    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                "Error! The following requirements were not found in the discretization list:" << missing_list << ".\n");
  }
}

void GenericSTKMeshStruct::
loadField (const std::string& field_name, const Teuchos::ParameterList& field_params,
           Teuchos::RCP<Thyra_MultiVector>& field_mv,
           const CombineAndScatterManager& cas_manager,
           const Teuchos::RCP<const Teuchos_Comm>& comm,
           bool node, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out)
{
  // Getting the serial and (possibly) parallel vector spaces
  auto serial_vs = cas_manager.getOwnedVectorSpace();
  auto vs        = cas_manager.getOverlappedVectorSpace();

  std::string temp_str;
  std::string field_type = (node ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");

  // The serial service multivector
  Teuchos::RCP<Thyra_MultiVector> serial_req_mvec;

  std::string fname = field_params.get<std::string>("File Name");

  *out << "  - Reading " << field_type << " field '" << field_name << "' from file '" << fname << "' ... ";
  out->getOStream()->flush();
  // Read the input file and stuff it in the Tpetra multivector

  if (scalar) {
    if (layered) {
      temp_str = field_name + "_NLC";
      auto& norm_layers_coords = fieldContainer->getMeshVectorStates()[temp_str];
      readLayeredScalarFileSerial (fname,serial_req_mvec,cas_manager.getOwnedVectorSpace(),norm_layers_coords,comm);

      // Broadcast the normalized layers coordinates
      int size = norm_layers_coords.size();
      Teuchos::broadcast(*comm,0,size,norm_layers_coords.data());
    } else {
      readScalarFileSerial (fname,serial_req_mvec,cas_manager.getOwnedVectorSpace(),comm);
    }
  } else {
    if (layered) {
      temp_str = field_name + "_NLC";
      auto& norm_layers_coords = fieldContainer->getMeshVectorStates()[temp_str];
      readLayeredVectorFileSerial (fname,serial_req_mvec,cas_manager.getOwnedVectorSpace(),norm_layers_coords,comm);

      // Broadcast the normalized layers coordinates
      int size = norm_layers_coords.size();
      Teuchos::broadcast(*comm,0,size,norm_layers_coords.data());
    } else {
      readVectorFileSerial (fname,serial_req_mvec,cas_manager.getOwnedVectorSpace(),comm);
    }
  }
  *out << "done!\n";

  if (field_params.isParameter("Scale Factor")) {
    Teuchos::Array<double> scale_factors;
    if (field_params.isType<Teuchos::Array<double>>("Scale Factor")) {
      scale_factors = field_params.get<Teuchos::Array<double> >("Scale Factor");
      TEUCHOS_TEST_FOR_EXCEPTION (scale_factors.size()!=static_cast<int>(serial_req_mvec->domain()->dim()),
                                  Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given scale factors vector size does not match the field dimension.\n");
    } else if (field_params.isType<double>("Scale Factor")) {
      scale_factors.resize(serial_req_mvec->domain()->dim());
      std::fill_n(scale_factors.begin(),scale_factors.size(),field_params.get<double>("Scale Factor"));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                 "Error! Invalid type for parameter 'Scale Factor'. Should be either 'double' or 'Array(double)'.\n");
    }

    *out << "   - Scaling " << field_type << " field '" << field_name << "' with scaling factors [" << scale_factors[0];
    for (int i=1; i<scale_factors.size(); ++i) {
      *out << " " << scale_factors[i];
    }
    *out << "]\n";

    for (int i=0; i<scale_factors.size(); ++i) {
      serial_req_mvec->col(i)->scale (scale_factors[i]);
    }
  }

  // Fill the (possibly) parallel vector
  field_mv = Thyra::createMembers(vs,serial_req_mvec->domain()->dim());
  cas_manager.scatter(*serial_req_mvec, *field_mv, CombineMode::INSERT);
}

void GenericSTKMeshStruct::
fillField (const std::string& field_name,
           const Teuchos::ParameterList& field_params,
           Teuchos::RCP<Thyra_MultiVector>& field_mv,
           const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
           bool nodal, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out)
{
  std::string temp_str;
  std::string field_type = (nodal ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");

  if (field_params.isParameter("Random Value")) {
    *out << "  - Filling " << field_type << " field '" << field_name << "' with random values.\n";

    Teuchos::Array<std::string> randomize = field_params.get<Teuchos::Array<std::string> >("Random Value");
    field_mv = Thyra::createMembers(entities_vs,randomize.size());

    if (layered) {
      *out << "    - Filling layers normalized coordinates linearly in [0,1].\n";

      temp_str = field_name + "_NLC";
      auto& norm_layers_coords = fieldContainer->getMeshVectorStates()[temp_str];
      int size = norm_layers_coords.size();
      if (size==1) {
        norm_layers_coords[0] = 1.;
      } else {
        int n_int = size-1;
        double dx = 1./n_int;
        norm_layers_coords[0] = 0.;
        for (int i=0; i<n_int; ++i) {
          norm_layers_coords[i+1] = norm_layers_coords[i]+dx;
        }
      }
    }

    // If there are components that were marked to not be randomized,
    // we look for the parameter 'Field Value' and use the corresponding entry.
    // If there is no such parameter, we fill the non random entries with zeroes.
    Teuchos::Array<double> values;
    if (field_params.isParameter("Field Value")) {
      values = field_params.get<Teuchos::Array<double> >("Field Value");
    } else {
      values.resize(randomize.size(),0.);
    }

    for (int iv=0; iv<randomize.size(); ++iv) {
      if (randomize[iv]=="false" || randomize[iv]=="no") {
        *out << "    - Using constant value " << values[iv] << " for component " << iv << ", which was marked as not random.\n";
        field_mv->col(iv)->assign(values[iv]);
      }
    }
  } else if (field_params.isParameter("Field Value")) {
    Teuchos::Array<double> values;
    if (field_params.isType<Teuchos::Array<double>>("Field Value")) {
      values = field_params.get<Teuchos::Array<double> >("Field Value");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()==0 , Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given field value array has size 0.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()==1 && !scalar , Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given field value array has size 1, but the field is not scalar.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()>1 && scalar , Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given field value array has size >1, but the field is scalar.\n");
    } else if (field_params.isType<double>("Field Value")) {
      if (scalar) {
        values.resize(1);
        values[0] = field_params.get<double>("Field Value");
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (!field_params.isParameter("Vector Dim"), std::logic_error,
                                    "Error! Cannot determine dimension of " << field_type << " field '" << field_name << "'. "
                                    "In order to fill with constant value, either specify 'Vector Dim', or make 'Field Value' an Array(double).\n");
        values.resize(field_params.get<int>("Vector Dim"));
        std::fill_n(values.begin(),values.size(),field_params.get<double>("Field Value"));
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                 "Error! Invalid type for parameter 'Field Value'. Should be either 'double' or 'Array(double)'.\n");
    }

    if (layered) {
      *out << "  - Filling " << field_type << " field '" << field_name << "' with constant value " << values << " and filling layers normalized coordinates linearly in [0,1].\n";

      temp_str = field_name + "_NLC";
      auto& norm_layers_coords = fieldContainer->getMeshVectorStates()[temp_str];
      int size = norm_layers_coords.size();
      if (size==1) {
        norm_layers_coords[0] = 1.;
      } else {
        int n_int = size-1;
        double dx = 1./n_int;
        norm_layers_coords[0] = 0.;
        for (int i=0; i<n_int; ++i) {
          norm_layers_coords[i+1] = norm_layers_coords[i]+dx;
        }
      }
    } else {
      *out << "  - Filling " << field_type << " field '" << field_name << "' with constant value " << values << ".\n";
    }

    field_mv = Thyra::createMembers(entities_vs,values.size());
    for (int iv(0); iv<field_mv->domain()->dim(); ++iv) {
      field_mv->col(iv)->assign(values[iv]);
    }
  }
}

void GenericSTKMeshStruct::
computeField (const std::string& field_name,
              const Teuchos::ParameterList& field_params,
              Teuchos::RCP<Thyra_MultiVector>& field_mv,
              const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
              const std::vector<stk::mesh::Entity>& entities,
              bool nodal, bool scalar, bool layered,
              const Teuchos::RCP<Teuchos::FancyOStream> out)
{
#ifdef ALBANY_PANZER_EXPR_EVAL
  // Only nodal fields allowed, no layered fields
  TEUCHOS_TEST_FOR_EXCEPTION(!nodal, std::logic_error, "Error! Only nodal fields can be computed from a mathematical expression.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(layered, std::logic_error, "Error! Layered fields cannot be computed from a mathematical expression.\n");

  int field_dim = 1;
  if (!scalar) {
    TEUCHOS_TEST_FOR_EXCEPTION(!field_params.isParameter("Vector Dim"), std::logic_error,
                               "Error! In order to compute the vector field '" << field_name << "' "
                               "from a mathematical expression, you must provide the parameter 'Vector Dim'.\n");
    field_dim = field_params.get<int>("Vector Dim");
  }

  // Get the expressions out of the parameter list.
  Teuchos::Array<std::string> expressions = field_params.get<Teuchos::Array<std::string>>("Field Expression");

  // NOTE: we need expressions to be of length AT LEAST equal to the field dimension.
  //       If the length L is larger than the field dimension M, then the first L-M
  //       strings are assumed to be coefficients needed for the field formula.
  //       E.g.: if we have a field of dimension 2, one could write
  //         <Parameter name="Field Expression" type="Array(string)" value="{a=1.5;b=-1;c=2;a*x^2+b*x+c;a*x+b*x+c}"/>

  int num_expr = expressions.size();
  std::string temp_str;
  std::string field_type = (nodal ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");
  TEUCHOS_TEST_FOR_EXCEPTION(num_expr<field_dim, Teuchos::Exceptions::InvalidParameter,
                             "Error! Input array for 'Field Expression' is too short. "
                             "Expected length >=" << field_dim << ". Got " << num_expr << " instead.\n");

  *out << "  - Computing " << field_type << " field '" << field_name << "' from mathematical expression(s):";
  int num_params = num_expr - field_dim;
  for (int idim=num_params; idim<num_expr; ++idim) {
    *out << " " << expressions[idim] << (idim==num_expr-1 ? "" : ";");
  }
  if (num_params>0) {
    *out << " (with";
    for (int idim=0; idim<num_params; ++idim) {
      *out << " " << expressions[idim] << (idim==num_params-1 ? "" : ";");
    }
    *out << ")";
  }
  *out << ".\n";

  // Extract coordinates of all nodes
  field_mv = Thyra::createMembers(entities_vs,field_dim);
  using exec_space = Tpetra_MultiVector::execution_space;
  using view_type = Kokkos::View<double**,DeviceView1d<double>::memory_space>;
  using layout = view_type::traits::array_layout;

  view_type x("x",entities.size(),1), y("y",entities.size(),1), z("z",entities.size(),1);
  view_type::HostMirror x_h = Kokkos::create_mirror_view(x);
  view_type::HostMirror y_h = Kokkos::create_mirror_view(y);
  view_type::HostMirror z_h = Kokkos::create_mirror_view(z);
  const auto& coordinates = *this->getCoordinatesField3d();
  double* xyz;
  for (unsigned int i=0; i<entities.size(); ++i) {
    xyz = stk::mesh::field_data(coordinates, entities[i]);

    x_h(i,0) = xyz[0];
    y_h(i,0) = xyz[1];
    z_h(i,0) = xyz[2];
  }
  Kokkos::deep_copy(x,x_h);
  Kokkos::deep_copy(y,y_h);
  Kokkos::deep_copy(z,z_h);

  // Set up the expression parser
  panzer::Expr::Eval<double**,layout,exec_space> eval;
  using const_view_type = decltype(eval)::const_view_type;
  set_cmath_functions(eval);
  eval.set("x",x);
  eval.set("y",y);
  eval.set("z",z);

  // Start by reading the parameters used in the field expression(s)
  Teuchos::any result;
  for (int iparam=0; iparam<num_params; ++iparam) {
    eval.read_string(result,expressions[iparam]+";","params");
  }

  // Parse and evaluate all the expressions
  for (int idim=0; idim<field_dim; ++idim) {
    eval.read_string(result,expressions[num_params+idim],"field expression");
    auto result_view = Teuchos::any_cast<const_view_type>(result);
    auto result_view_1d = DeviceView1d<const double>(result_view.data(),result_view.extent_int(0));
    Kokkos::deep_copy(getNonconstDeviceData(field_mv->col(idim)),result_view_1d);
  }
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Cannot read the field from a mathematical expression, since PanzerExprEval package was not found in Trilinos.\n");
#endif
}

void GenericSTKMeshStruct::
readScalarFileSerial (const std::string& fname,
                      Teuchos::RCP<Thyra_MultiVector>& mvec,
                      const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                      const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  // It's a scalar, so we already know MultiVector has only 1 vector
  mvec = Thyra::createMembers(vs,1);

  if (comm->getRank() != 0)
  {
    // Only process 0 will load the file...
    return;
  }

  GO numNodes;
  Teuchos::ArrayRCP<ST> nonConstView = getNonconstLocalData(mvec->col(0));

  std::ifstream ifile;
  ifile.open(fname.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

  ifile >> numNodes;
  TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstView.size(), Teuchos::Exceptions::InvalidParameterValue,
                              "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                              "is different from the number expected (" << nonConstView.size() << ").\n");

  for (GO i = 0; i < numNodes; i++) {
    ifile >> nonConstView[i];
  }

  ifile.close();
}

void GenericSTKMeshStruct::
readVectorFileSerial (const std::string& fname,
                      Teuchos::RCP<Thyra_MultiVector>& mvec,
                      const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                      const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  int numComponents;
  GO numNodes;
  std::ifstream ifile;
  if (comm->getRank() == 0) {
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numComponents;
  }

  Teuchos::broadcast(*comm,0,1,&numComponents);
  mvec = Thyra::createMembers(vs,numComponents);

  if (comm->getRank()==0) {
    auto nonConstView = getNonconstLocalData(mvec);
    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstView[0].size(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << nonConstView[0].size() << ").\n");

    for (int icomp=0; icomp<numComponents; ++icomp) {
      auto comp_view = nonConstView[icomp];
      for (GO i=0; i<numNodes; ++i)
        ifile >> comp_view[i];
    }
    ifile.close();
  }
}

void GenericSTKMeshStruct::
readLayeredScalarFileSerial (const std::string &fname,
                             Teuchos::RCP<Thyra_MultiVector>& mvec,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  size_t numLayers=0;
  GO numNodes;
  std::ifstream ifile;
  if (comm->getRank()==0) {
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numLayers;
  }

  Teuchos::broadcast(*comm,0,1,&numLayers);
  mvec = Thyra::createMembers(vs,numLayers);

  if (comm->getRank()==0) {
    auto nonConstView = getNonconstLocalData(mvec);
    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstView[0].size(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << nonConstView[0].size() << ").\n");
    TEUCHOS_TEST_FOR_EXCEPTION (numLayers != normalizedLayersCoords.size(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of layers in file " << fname << " (" << numLayers << ") " <<
                                "is different from the number expected (" << normalizedLayersCoords.size() << ")." <<
                                " To fix this, please specify the correct layered data dimension when you register the state.\n");

    for (size_t il = 0; il < numLayers; ++il) {
      ifile >> normalizedLayersCoords[il];
    }

    for (size_t il=0; il<numLayers; ++il) {
      for (GO i=0; i<numNodes; ++i) {
        ifile >> nonConstView[il][i];
      }
    }
    ifile.close();
  }
}

void GenericSTKMeshStruct::
readLayeredVectorFileSerial (const std::string &fname, Teuchos::RCP<Thyra_MultiVector>& mvec,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  int numVectors=0;
  int numLayers,numComponents;
  GO numNodes;
  std::ifstream ifile;
  if (comm->getRank()==0) {

    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error, "Error in GenericSTKMeshStruct: unable to open the file " << fname << ".\n");

    ifile >> numNodes >> numComponents >> numLayers;
    numVectors = numLayers*numComponents;
  }

  Teuchos::broadcast(*comm,0,1,&numVectors);
  mvec = Thyra::createMembers(vs,numVectors);

  if (comm->getRank()==0) {
    auto nonConstView = getNonconstLocalData(mvec);

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstView[0].size(), Teuchos::Exceptions::InvalidParameterValue,
                                "Error in GenericSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") " <<
                                "is different from the number expected (" << nonConstView[0].size() << ").\n");

    normalizedLayersCoords.resize(numLayers);
    for (int il=0; il<numLayers; ++il) {
      ifile >> normalizedLayersCoords[il];
    }

    // Layer ordering: before switching component, we first do all the layers of the current component
    // This is because with the stk field (natural ordering) we want to keep the layer dimension last.
    // Ex: a 2D field f(i,j) would be stored at the raw array position i*num_cols+j. In our case,
    //     num_cols is the number of layers, and num_rows the number of field components
    for (int il=0; il<numLayers; ++il) {
      for (int icomp(0); icomp<numComponents; ++icomp) {
        Teuchos::ArrayRCP<ST> col_vals = nonConstView[icomp*numLayers+il];
        for (GO i=0; i<numNodes; ++i) {
          ifile >> col_vals[i];
        }
      }
    }
    ifile.close();
  }
}

void GenericSTKMeshStruct::checkFieldIsInMesh (const std::string& fname, const std::string& ftype) const
{
  stk::topology::rank_t entity_rank;
  if (ftype.find("Node")==std::string::npos) {
    entity_rank = stk::topology::ELEM_RANK;
  } else {
    entity_rank = stk::topology::NODE_RANK;
  }

  int dim = 1;
  if (ftype.find("Vector")!=std::string::npos) {
    ++dim;
  }
  if (ftype.find("Layered")!=std::string::npos) {
    ++dim;
  }

  typedef AbstractSTKFieldContainer::ScalarFieldType  SFT;
  typedef AbstractSTKFieldContainer::VectorFieldType  VFT;
  typedef AbstractSTKFieldContainer::TensorFieldType  TFT;
  bool missing = true;
  switch (dim)
  {
    case 1:
      missing = (metaData->get_field<SFT> (entity_rank, fname)==0);
      break;
    case 2:
      missing = (metaData->get_field<VFT> (entity_rank, fname)==0);
      break;
    case 3:
      missing = (metaData->get_field<TFT> (entity_rank, fname)==0);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid field dimension.\n");
  }

  if (missing) {
    bool isFieldInMesh = false;
    auto fl = metaData->get_fields();
    auto f = fl.begin();
    for (; f != fl.end(); ++f) {
      isFieldInMesh = (fname == (*f)->name());
      if(isFieldInMesh) break;
    }
    if(isFieldInMesh) {
       TEUCHOS_TEST_FOR_EXCEPTION (missing, std::runtime_error, "Error! The field '" << fname << "' in the mesh has different rank or dimensions than the ones specified\n"
                                                        << " Rank required: " << entity_rank << ", rank of field in mesh: " << (*f)->entity_rank() << "\n"
                                                        << " Dimension required: " << dim << ", dimension of field in mesh: " << (*f)->field_array_rank()+1 << "\n");
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION (missing, std::runtime_error, "Error! The field '" << fname << "' was not found in the mesh.\n"
                                                       << "  Probably it was not registered it in the state manager (which forwards it to the mesh)\n");
  }
}

void GenericSTKMeshStruct::checkInput(std::string option, std::string value, std::string allowed_values)
{
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
#else
  // Silence compiler warnings
  (void) option;
  (void) value;
  (void) allowed_values;
#endif
}

Teuchos::RCP<Teuchos::ParameterList>
GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");
  validPL->set<std::string>("Exodus Output File Name", "",
      "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Exodus Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDot Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDotDot Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
#ifdef ALBANY_DTK
  validPL->set<std::string>("Exodus Solution DTK Name", "",
      "Name of solution dtk written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDot DTK Name", "",
      "Name of solution_dot dtk written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDotDot DTK Name", "",
      "Name of solution_dotdot dtk written to Exodus file. Requires SEACAS build");
#endif
  validPL->set<bool>("Output DTK Field to Exodus", true, "Boolean indicating whether to write dtk field to exodus file");
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
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid2");
  validPL->set<std::string>("Cubature Rule", "", "Integration rule sent to Intrepid2: GAUSS, GAUSS_RADAU_LEFT, GAUSS_RADAU_RIGHT, GAUSS_LOBATTO");
  validPL->set<int>("Workset Size", DEFAULT_WORKSET_SIZE, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Use Automatic Aura", false, "Use automatic aura with BulkData");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<std::string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for LandIce problem that require tranformation of STK mesh
  validPL->set<int>("Element Degree", 1, "Element degree (points per edge - 1) in enriched Aeras mesh");
  validPL->set<bool>("Write Coordinates to MatrixMarket", false, "Writing Coordinates to MatrixMarket File"); //for writing coordinates to matrix market file
  validPL->set<double>("LandIce alpha", 0.0, "Surface boundary inclination for LandIce problems (in degrees)"); //for LandIce problem that require tranformation of STK mesh
  validPL->set<double>("LandIce L", 1, "Domain length for LandIce problems"); //for LandIce problem that require tranformation of STK mesh

  validPL->set<double>("x-shift", 0.0, "Value by which to shift domain in positive x-direction");
  validPL->set<double>("y-shift", 0.0, "Value by which to shift domain in positive y-direction");
  validPL->set<double>("z-shift", 0.0, "Value by which to shift domain in positive z-direction");
  validPL->set<Teuchos::Array<double>>("Betas BL Transform", Teuchos::tuple<double>(0.0, 0.0, 0.0), "Beta parameters for Tanh Boundary Layer transform type");

  validPL->set<bool>("Contiguous IDs", "true", "Tells Ascii mesh reader is mesh has contiguous global IDs on 1 processor."); //for LandIce problem that require tranformation of STK mesh

  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields,
                     "Fields to pick up from the restart file when restarting");
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layout of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("SolutionDot Vector Components", defaultFields,
      "Names and layout of solution_dot output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", defaultFields,
      "Names and layout of solution_dotdot output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("Residual Vector Components", defaultFields,
      "Names and layout of residual output vector written to Exodus file. Requires SEACAS build");

  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");
  validPL->set<bool>("Transfer Solution to Coordinates", false, "Copies the solution vector to the coordinates for output");

  validPL->set<bool>("Set All Parts IO", false, "If true, all parts are marked as io parts");
  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the composite tet 10 basis in Intrepid");
  validPL->set<bool>("Build Node Sets From Side Sets",false,"Flag to build node sets from side sets");
  validPL->set<bool>("Export 3d coordinates field",false,"If true AND the mesh dimension is not already 3, export a 3d version of the coordinate field.");

  validPL->sublist("Required Fields Info", false, "Info for the creation of the required fields in the STK mesh");

  validPL->set<bool>("Ignore Side Maps", true, "If true, we ignore possible side maps already imported from the exodus file");
  validPL->sublist("Contact", false, "Sublist used to specify contact parameters");

  // Uniform percept adaptation of input mesh prior to simulation

  validPL->set<std::string>("STK Initial Refine", "", "stk::percept refinement option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Enrich", "", "stk::percept enrichment option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Convert", "", "stk::percept conversion option to apply after the mesh is input");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("Number of Refinement Passes", 1, "Number of times to apply the refinement process");

  validPL->sublist("Side Set Discretizations", false, "A sublist containing info for storing side discretizations");

  return validPL;
}

} // namespace Albany
