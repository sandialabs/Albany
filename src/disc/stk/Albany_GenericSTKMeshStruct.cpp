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
//#ifdef LCM_SPECULATIVE
#include <stk_adapt/UniformRefiner.hpp>
#include <stk_adapt/UniformRefinerPattern.hpp>
//#endif

Albany::GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
    int numDim_)
    : params(params_),
      adaptParams(adaptParams_),
      buildEMesh(false)
{

  metaData = new stk::mesh::fem::FEMMetaData();

  buildEMesh = buildPerceptEMesh();

  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    std::vector<std::string> entity_rank_names = stk::mesh::fem::entity_rank_names(numDim_);
    // eMesh needs "FAMILY_TREE" entity
    if(buildEMesh)
      entity_rank_names.push_back("FAMILY_TREE");
    metaData->FEM_initialize(numDim_, entity_rank_names);
  }

  interleavedOrdering = params->get("Interleaved Ordering",true);
  allElementBlocksHaveSamePhysics = true; 
  
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

  if (bulkData == NULL)
  bulkData = new stk::mesh::BulkData(stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
                          Albany::getMpiCommFromEpetraComm(*comm), worksetSize );

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
  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");

  exoOutputInterval = params->get<int>("Exodus Write Interval", 1);


  //get the type of transformation of STK mesh (for FELIX problems)
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh (for FELIX problems)
  felixAlpha = params->get("FELIX alpha", 0.0); 
  felixL = params->get("FELIX L", 1.0); 

  // Build the eMesh if needed
  if(buildEMesh)

   eMesh = Teuchos::rcp(new stk::percept::PerceptMesh(metaData, bulkData, false));

  // Build 
  if(!eMesh.is_null())

   buildUniformRefiner();

}

bool Albany::GenericSTKMeshStruct::buildPerceptEMesh(){

   // If there exists a nonempty "refine", "convert", or "enrich" string
    std::string refine = params->get<string>("STK Initial Refine", "");
    if(refine.length() > 0) return true;
    std::string convert = params->get<string>("STK Initial Enrich", "");
    if(convert.length() > 0) return true;
    std::string enrich = params->get<string>("STK Initial Convert", "");
    if(enrich.length() > 0) return true;

    // Or, if a percept mesh is needed by the "Adaptation" sublist
//    if(!adaptParams && )
//      return true;

    return false;

}

void Albany::GenericSTKMeshStruct::buildUniformRefiner(){

//#ifdef LCM_SPECULATIVE

    stk::adapt::BlockNamesType block_names(stk::percept::EntityRankEnd+1u);

    std::string refine = params->get<string>("STK Initial Refine", "");
    std::string convert = params->get<string>("STK Initial Enrich", "");
    std::string enrich = params->get<string>("STK Initial Convert", "");

    std::string convert_options = stk::adapt::UniformRefinerPatternBase::s_convert_options;
    std::string refine_options  = stk::adapt::UniformRefinerPatternBase::s_refine_options;
    std::string enrich_options  = stk::adapt::UniformRefinerPatternBase::s_enrich_options;

    // Has anything been specified?

    if(refine.length() == 0 && convert.length() == 0 && enrich.length() == 0)

       return;

    if (refine.length())

      checkInput("refine", refine, refine_options);

    if (convert.length())

      checkInput("convert", convert, convert_options);

    if (enrich.length())

      checkInput("enrich", enrich, enrich_options);

    refinerPattern = stk::adapt::UniformRefinerPatternBase::createPattern(refine, enrich, convert, *eMesh, block_names);

//#endif

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

    stk::mesh::PartVector add_parts;
    stk::mesh::create_adjacent_entities(*bulkData, add_parts);
  
    stk::mesh::EntityRank elementRank = metaData->element_rank();
    stk::mesh::EntityRank nodeRank = metaData->node_rank();
    stk::mesh::EntityRank sideRank = metaData->side_rank();
  
    std::vector<stk::mesh::Entity*> element_lst;
  //  stk::mesh::get_entities(*(bulkData),elementRank,element_lst);
    
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
        bulkData->buckets( elementRank ) ,
        element_lst );
  
    bulkData->modification_begin();
  
      // Remove extra relations from element
    for (int i = 0; i < element_lst.size(); ++i){
        stk::mesh::Entity & element = *(element_lst[i]);
        stk::mesh::PairIterRelation relations = element.relations();
        std::vector<stk::mesh::Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk::mesh::PairIterRelation::iterator j = relations.begin();
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
        stk::mesh::Entity & entity = *(del_relations[j]);
        bulkData->destroy_relation(element,entity,del_ids[j]);
      }
    }
  
    if (elementRank == 3){
      // Remove extra relations from face
      std::vector<stk::mesh::Entity*> face_lst;
      //stk::mesh::get_entities(*(bulkData),elementRank-1,face_lst);
      // Loop through all faces visible to this processor, as a face can be visible on two processors
      stk::mesh::get_selected_entities( select_owned_or_shared,
                                        bulkData->buckets( elementRank-1 ) ,
                                        face_lst );
      stk::mesh::EntityRank entityRank = face_lst[0]->entity_rank(); // This is rank 2 always...
  //std::cout << "element rank - 1: " << elementRank - 1 << " face rank: " << entityRank << std::endl;
      for (int i = 0; i < face_lst.size(); ++i){
        stk::mesh::Entity & face = *(face_lst[i]);
        stk::mesh::PairIterRelation relations = face.relations();
        std::vector<stk::mesh::Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk::mesh::PairIterRelation::iterator j = relations.begin();
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
          stk::mesh::Entity & entity = *(del_relations[j]);
          bulkData->destroy_relation(face, entity, del_ids[j]);
  //std::cout << "Deleting rank: " << entity.entity_rank() << " id: " << del_ids[j] << std::endl;
        }
      }
    }
  
    bulkData->modification_end();
  }

}

void Albany::GenericSTKMeshStruct::uniformRefineMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

//#ifdef LCM_SPECULATIVE
// Refine if requested

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();


  if(!refinerPattern.is_null() && proc_rank_field){
    bulkData->modification_begin();

    stk::adapt::UniformRefiner refiner(*eMesh, *refinerPattern, proc_rank_field);

    refiner.doBreak();
    bulkData->modification_end();
  }
//#endif

}


void Albany::GenericSTKMeshStruct::rebalanceMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

// Zoltan is required here

#ifdef ALBANY_ZOLTAN
  bool rebalance = params->get<bool>("Rebalance Mesh", false);
  bool useSerialMesh = params->get<bool>("Use Serial Mesh", false);

  if(rebalance || (useSerialMesh && comm->NumProc() > 1)){

    double imbalance;

    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

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

void Albany::GenericSTKMeshStruct::printParts(stk::mesh::fem::FEMMetaData *metaData){

    std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;

    stk::mesh::PartVector all_parts = metaData->get_parts();

    for (stk::mesh::PartVector::iterator i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part)
    {
       stk::mesh::Part *  part = *i_part ;

       std::cout << "\t" << part->name() << std::endl;
    }

}

void
Albany::GenericSTKMeshStruct::checkInput(std::string option, std::string value, std::string allowed_values){

      std::vector<std::string> vals = stk::adapt::Util::split(allowed_values, ", ");
      for (unsigned i = 0; i < vals.size(); i++)
        {
          if (vals[i] == value)
            return;
        }

       TEUCHOS_TEST_FOR_EXCEPTION(true,
         std::runtime_error,
         "Adaptation input error in GenericSTKMeshStruct initialization: bar option: " << option << std::endl);

}

Teuchos::RCP<Teuchos::ParameterList>
Albany::GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<std::string>("Exodus Output File Name", "",
      "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Exodus Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
  validPL->set<int>("Exodus Write Interval", 3, "Step interval to write solution data to Exodus file");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<double>("FELIX alpha", 0.0, "Surface boundary inclination for FELIX problems (in degrees)"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<double>("FELIX L", 1, "Domain length for FELIX problems"); //for FELIX problem that require tranformation of STK mesh

  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields, 
                     "Fields to pick up from the restart file when restarting");
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layout of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("Residual Vector Components", defaultFields,
      "Names and layout of residual output vector written to Exodus file. Requires SEACAS build");

  // Uniform percept adaptation of input mesh prior to simulation

  validPL->set<std::string>("STK Initial Refine", "", "stk::percept refinement option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Enrich", "", "stk::percept enrichment option to apply after the mesh is input");
  validPL->set<std::string>("STK Initial Convert", "", "stk::percept conversion option to apply after the mesh is input");
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");



  return validPL;
}
