//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_GenericSTKMeshStruct.hpp"

#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"


#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>

// Rebalance
#ifdef ALBANY_ZOLTAN
#include <stk_rebalance/Rebalance.hpp>
#include <stk_rebalance/Partition.hpp>
#include <stk_rebalance/ZoltanPartition.hpp>
#include <stk_rebalance_utils/RebalanceUtils.hpp>
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

  metaData = new stk_classic::mesh::fem::FEMMetaData();

  buildEMesh = buildPerceptEMesh();

  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    std::vector<std::string> entity_rank_names = stk_classic::mesh::fem::entity_rank_names(numDim_);
    // eMesh needs "FAMILY_TREE" entity
    if(buildEMesh)
      entity_rank_names.push_back("FAMILY_TREE");
    metaData->FEM_initialize(numDim_, entity_rank_names);
  }

  interleavedOrdering = params->get("Interleaved Ordering",true);
  allElementBlocksHaveSamePhysics = true;
  compositeTet = params->get<bool>("Use Composite Tet 10", false);

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  bulkData = NULL;
}

Albany::GenericSTKMeshStruct::~GenericSTKMeshStruct()
{
  delete metaData;
  delete bulkData;
}

void Albany::GenericSTKMeshStruct::SetupFieldData(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!metaData->is_FEM_initialized(),
       std::logic_error,
       "LogicError: metaData->FEM_initialize(numDim) not yet called" << std::endl);

  neq = neq_;

  this->nodal_data_block = sis->getNodalDataBlock();

  if (bulkData == NULL)

     bulkData = new stk_classic::mesh::BulkData(stk_classic::mesh::fem::FEMMetaData::get_meta_data(*metaData),
                          Albany::getMpiCommFromEpetraComm(*comm), worksetSize );

#ifdef ALBANY_LCM
  // If adaptation in LCM, create a new part for boundary
  if (adaptParams.is_null() == false) {
    stk_classic::mesh::Part &
    boundary_part = metaData->declare_part("boundary");
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(boundary_part);
#endif
  }
#endif // ALBANY_LCM

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
            metaData, bulkData, neq_, req, numDim, sis));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::OrdinarySTKFieldContainer<false>(params,
            metaData, bulkData, neq_, req, numDim, sis));

  }

  else {

      if(interleavedOrdering)
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<true>(params,
            metaData, bulkData, neq_, req, numDim, sis, solution_vector, residual_vector));
      else
        this->fieldContainer = Teuchos::rcp(new Albany::MultiSTKFieldContainer<false>(params,
            metaData, bulkData, neq_, req, numDim, sis, solution_vector, residual_vector));

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

  //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);

  //Does user want to write coordinates to matrix market file (e.g., for ML analysis)?
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false);

  transferSolutionToCoords = params->get<bool>("Transfer Solution to Coordinates", false);

#ifdef ALBANY_STK_PERCEPT
  // Build the eMesh if needed
  if(buildEMesh)

   eMesh = Teuchos::rcp(new stk_classic::percept::PerceptMesh(metaData, bulkData, false));

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

    stk_classic::adapt::BlockNamesType block_names(stk_classic::percept::EntityRankEnd+1u);

    std::string refine = params->get<std::string>("STK Initial Refine", "");
    std::string convert = params->get<std::string>("STK Initial Convert", "");
    std::string enrich = params->get<std::string>("STK Initial Enrich", "");

    std::string convert_options = stk_classic::adapt::UniformRefinerPatternBase::s_convert_options;
    std::string refine_options  = stk_classic::adapt::UniformRefinerPatternBase::s_refine_options;
    std::string enrich_options  = stk_classic::adapt::UniformRefinerPatternBase::s_enrich_options;

    // Has anything been specified?

    if(refine.length() == 0 && convert.length() == 0 && enrich.length() == 0)

       return false;

    if (refine.length())

      checkInput("refine", refine, refine_options);

    if (convert.length())

      checkInput("convert", convert, convert_options);

    if (enrich.length())

      checkInput("enrich", enrich, enrich_options);

    refinerPattern = stk_classic::adapt::UniformRefinerPatternBase::createPattern(refine, enrich, convert, *eMesh, block_names);
    uniformRefinementInitialized = true;

    return true;

#else
    return false;
#endif

}

bool Albany::GenericSTKMeshStruct::buildLocalRefiner(){

#ifdef ALBANY_STK_PERCEPT

    if(adaptParams.is_null()) return false;

//    stk_classic::adapt::BlockNamesType block_names = stk_classic::adapt::BlockNamesType();
    stk_classic::adapt::BlockNamesType block_names(stk_classic::percept::EntityRankEnd+1u);

    std::string adapt_method = adaptParams->get<std::string>("Method", "");

    // Check if adaptation was specified
    if(adapt_method.length() == 0) return false;

    std::string pattern = adaptParams->get<std::string>("Refiner Pattern", "");

    if(pattern == "Local_Tet4_Tet4_N"){

//      refinerPattern = Teuchos::rcp(new stk_classic::adapt::Local_Tet4_Tet4_N(*eMesh, block_names));
      refinerPattern = Teuchos::rcp(new stk_classic::adapt::Local_Tet4_Tet4_N(*eMesh));
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
    std::map<std::string, stk_classic::mesh::Part*>& partVec){

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

  map<std::string, stk_classic::mesh::Part*>::iterator it;
  std::vector<stk_classic::mesh::Part*>::const_iterator p;

  for(it = partVec.begin(); it != partVec.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it

    const stk_classic::mesh::PartVector & subsets   = it->second->subsets();

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

int Albany::GenericSTKMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  // Resize workset size down to maximum number in an element block
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
     // compute numWorksets, and shrink workset size to minimize padding
     const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
     return (1 + (ebSizeMax-1) /  numWorksets);
  }
}

void Albany::GenericSTKMeshStruct::computeAddlConnectivity()
{

  if(adaptParams.is_null()) return;

  std::string& method = adaptParams->get("Method", "");

  // Mesh fracture requires full mesh connectivity, created here
  if(method == "Topmod" || method == "Random"){

    stk_classic::mesh::PartVector add_parts;
    stk_classic::mesh::create_adjacent_entities(*bulkData, add_parts);

    stk_classic::mesh::EntityRank elementRank = metaData->element_rank();
    stk_classic::mesh::EntityRank nodeRank = metaData->node_rank();
    stk_classic::mesh::EntityRank sideRank = metaData->side_rank();

    std::vector<stk_classic::mesh::Entity*> element_lst;
  //  stk_classic::mesh::get_entities(*(bulkData),elementRank,element_lst);

    stk_classic::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();
    stk_classic::mesh::Selector select_owned = metaData->locally_owned_part();

  /*
        stk_classic::mesh::Selector select_owned_in_part =
        stk_classic::mesh::Selector( metaData->universal_part() ) &
        stk_classic::mesh::Selector( metaData->locally_owned_part() );

        stk_classic::mesh::get_selected_entities( select_owned_in_part ,
  */

     // Loop through only on-processor elements as we are just deleting entities inside the element
     stk_classic::mesh::get_selected_entities( select_owned,
        bulkData->buckets( elementRank ) ,
        element_lst );

    bulkData->modification_begin();

      // Remove extra relations from element
    for (int i = 0; i < element_lst.size(); ++i){
        stk_classic::mesh::Entity & element = *(element_lst[i]);
        stk_classic::mesh::PairIterRelation relations = element.relations();
        std::vector<stk_classic::mesh::Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk_classic::mesh::PairIterRelation::iterator j = relations.begin();
             j != relations.end(); ++j){

          // remove all relationships from element unless to faces(segments
          //   in 2D) or nodes

          if (
              j->entity_rank() != elementRank-1 && // element to face relation
              j->entity_rank() != nodeRank
             ){

            del_relations.push_back(j->entity());
            del_ids.push_back(j->identifier());
          }
        }

      for (int j = 0; j < del_relations.size(); ++j){
        stk_classic::mesh::Entity & entity = *(del_relations[j]);
        bulkData->destroy_relation(element,entity,del_ids[j]);
      }
    }

    if (elementRank == 3){
      // Remove extra relations from face
      std::vector<stk_classic::mesh::Entity*> face_lst;
      //stk_classic::mesh::get_entities(*(bulkData),elementRank-1,face_lst);
      // Loop through all faces visible to this processor, as a face can be visible on two processors
      stk_classic::mesh::get_selected_entities( select_owned_or_shared,
                                        bulkData->buckets( elementRank-1 ) ,
                                        face_lst );
      stk_classic::mesh::EntityRank entityRank = face_lst[0]->entity_rank(); // This is rank 2 always...
  //std::cout << "element rank - 1: " << elementRank - 1 << " face rank: " << entityRank << std::endl;
      for (int i = 0; i < face_lst.size(); ++i){
        stk_classic::mesh::Entity & face = *(face_lst[i]);
        stk_classic::mesh::PairIterRelation relations = face.relations();
        std::vector<stk_classic::mesh::Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk_classic::mesh::PairIterRelation::iterator j = relations.begin();
             j != relations.end(); ++j){

          if (
              j->entity_rank() != entityRank+1 && // face to element relation
              j->entity_rank() != entityRank-1 // && // face to segment relation
  //            j->entity_rank() != sideRank     ){
             ){

            del_relations.push_back(j->entity());
            del_ids.push_back(j->identifier());
          }
        }

        for (int j = 0; j < del_relations.size(); ++j){
          stk_classic::mesh::Entity & entity = *(del_relations[j]);
          bulkData->destroy_relation(face, entity, del_ids[j]);
  //std::cout << "Deleting rank: " << entity.entity_rank() << " id: " << del_ids[j] << std::endl;
        }
      }
    }

    bulkData->modification_end();
  }

}


void Albany::GenericSTKMeshStruct::uniformRefineMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

#ifdef ALBANY_STK_PERCEPT
// Refine if requested
  if(!uniformRefinementInitialized) return;

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();


  if(!refinerPattern.is_null() && proc_rank_field){

    stk_classic::adapt::UniformRefiner refiner(*eMesh, *refinerPattern, proc_rank_field);

    int numRefinePasses = params->get<int>("Number of Refinement Passes", 1);

    for(int pass = 0; pass < numRefinePasses; pass++){

      if(comm->MyPID() == 0)
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


void Albany::GenericSTKMeshStruct::rebalanceInitialMesh(const Teuchos::RCP<const Epetra_Comm>& comm){


  bool rebalance = params->get<bool>("Rebalance Mesh", false);
  bool useSerialMesh = params->get<bool>("Use Serial Mesh", false);

  if(rebalance || (useSerialMesh && comm->NumProc() > 1)){

    rebalanceAdaptedMesh(params, comm);

  }

}

void Albany::GenericSTKMeshStruct::rebalanceAdaptedMesh(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                                                        const Teuchos::RCP<const Epetra_Comm>& comm){

// Zoltan is required here

#ifdef ALBANY_ZOLTAN

    using std::cout; using std::endl;

    if(comm->NumProc() <= 1)

      return;

    double imbalance;

    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

    stk_classic::mesh::Selector selector(metaData->universal_part());
    stk_classic::mesh::Selector owned_selector(metaData->locally_owned_part());

    if(comm->MyPID() == 0){

      std::cout << "Before rebal nelements " << comm->MyPID() << "  " <<
        stk_classic::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

      std::cout << "Before rebal " << comm->MyPID() << "  " <<
        stk_classic::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
    }


    imbalance = stk_classic::rebalance::check_balance(*bulkData, NULL,
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0)

      std::cout << "Before rebalance: Imbalance threshold is = " << imbalance << endl;

    // Use Zoltan to determine new partition. Set the desired parameters (if any) from the input file

    Teuchos::ParameterList graph_options;

   //graph_options.sublist(stk_classic::rebalance::Zoltan::default_parameters_name()).set("LOAD BALANCING METHOD"      , "4");
    //graph_options.sublist(stk_classic::rebalance::Zoltan::default_parameters_name()).set("ZOLTAN DEBUG LEVEL"      , "10");

    if(params_->isSublist("Rebalance Options")){

      const Teuchos::RCP<Teuchos::ParameterList>& load_balance_method = Teuchos::sublist(params_, "Rebalance Options");

    // Set the desired parameters. The options are shown in
    // TRILINOS_ROOT/packages/stk/stk_rebalance/ZontanPartition.cpp

//      load_balance_method.set("LOAD BALANCING METHOD"      , "4");
//      load_balance_method.set("ZOLTAN DEBUG LEVEL"      , "10");

      graph_options.sublist(stk_classic::rebalance::Zoltan::default_parameters_name()) = *load_balance_method;

    }

    stk_classic::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph_options);
    stk_classic::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);


    imbalance = stk_classic::rebalance::check_balance(*bulkData, NULL,
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0)
      std::cout << "After rebalance: Imbalance threshold is = " << imbalance << endl;

#if 0 // Other experiments at rebalancing

    // Configure Zoltan to use graph-based partitioning
    Teuchos::ParameterList graph;
    Teuchos::ParameterList lb_method;
    lb_method.set("LOAD BALANCING METHOD"      , "4");
    graph.sublist(stk_classic::rebalance::Zoltan::default_parameters_name()) = lb_method;

    stk_classic::rebalance::Zoltan zoltan_partitiona(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph);

    *out << "Universal part " << comm->MyPID() << "  " <<
      stk_classic::mesh::count_selected_entities(selector, bulkData->buckets(metaData->element_rank())) << endl;
    *out << "Owned part " << comm->MyPID() << "  " <<
      stk_classic::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    stk_classic::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partitiona);

    *out << "After rebal " << comm->MyPID() << "  " <<
      stk_classic::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
    *out << "After rebal nelements " << comm->MyPID() << "  " <<
      stk_classic::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;


    imbalance = stk_classic::rebalance::check_balance(*bulkData, NULL,
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


void Albany::GenericSTKMeshStruct::printParts(stk_classic::mesh::fem::FEMMetaData *metaData){

    std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;

    stk_classic::mesh::PartVector all_parts = metaData->get_parts();

    for (stk_classic::mesh::PartVector::iterator i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part)
    {
       stk_classic::mesh::Part *  part = *i_part ;

       std::cout << "\t" << part->name() << std::endl;

    }

}

void
Albany::GenericSTKMeshStruct::checkInput(std::string option, std::string value, std::string allowed_values){

#ifdef ALBANY_STK_PERCEPT
      std::vector<std::string> vals = stk_classic::adapt::Util::split(allowed_values, ", ");
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
  validPL->set<int>("Cubature Rule", 0, "Integration rule sent to Intrepid: 0=GAUSS, 1=GAUSS_RADAU_LEFT, 2=GAUSS_RADAU_RIGHT, 3=GAUSS_LOBATTO");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<std::string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for FELIX problem that require tranformation of STK mesh
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

  // Uniform percept adaptation of input mesh prior to simulation

  validPL->set<std::string>("STK Initial Refine", "", "stk_classic::percept refinement option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Enrich", "", "stk_classic::percept enrichment option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Convert", "", "stk_classic::percept conversion option to apply after the mesh is input");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");
  validPL->set<int>("Number of Refinement Passes", 1, "Number of times to apply the refinement process");

  return validPL;

}


