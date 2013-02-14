//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_FMDBMeshStruct.hpp"
#include "mMesh.h"

#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"

#include <Shards_BasicTopologies.hpp>

#ifdef SCOREC_PARASOLID
#include "modelerParasolid.h"
#endif

#include "SCUtil.h"
#include "PUMI.h"

Albany::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(params->get<bool>("Use Serial Mesh", false))
{
  // fmdb skips mpi initialization if it's already initialized
  SCUTIL_Init(Albany::getMpiCommFromEpetraComm(*comm));
  params->validateParameters(*getValidDiscretizationParameters(),0);

  bool singlePEMeshRead;
  bool distributedMesh;

  // Determine if the mesh is read on one PE or distributed

  if(useSerialMesh || comm->NumProc() == 1)

//    singlePEMeshRead = true;
    distributedMesh = false;

  else

//    singlePEMeshRead = false;
    distributedMesh = true;

  // I think this might be the same as "Use Serial Mesh" (mesh read on PE 0 then re-partitioned to run on the
  // available processors (PE avail > 1) ?

  bool call_serial = params->get<bool>("Call serial global partition", false);

  std::string mesh_file = params->get<string>("FMDB Input File Name");
  int numPart = 1;

#if 0
  *out<<"************************************************************************\n";
  *out<<"[INPUT]\n";
  *out<<"\tdistributed mesh? ";
  if (useSerialMesh) *out<<"YES\n";
  else *out<<"NO\n";
  *out<<"\t#parts per proc: "<<numPart<<endl;
  SCUTIL_DspSysInfo();
  *out<<"************************************************************************\n\n";  
#endif

  // create a model and load
  model = NULL; // default is no model

#ifdef SCOREC_ACIS

  if(params->isParameter("Acis Model Input File Name")){ // User has an Acis model

    std::string model_file = params->get<string>("Acis Model Input File Name");
    model = new AcisModel(&model_file[0], 0);
  }
#endif
#ifdef SCOREC_PARASOLID

  if(params->isParameter("Parasolid Model Input File Name")){ // User has a Parasolid model

    std::string model_file = params->get<string>("Parasolid Model Input File Name");
    model = GM_createFromParasolidFile(&model_file[0]);
    // create element block, side set and node set  
    char* elem_name_98=new char[128];
    strcpy(elem_name_98, "Element_Block_98");
    char* elem_name_198=new char[128];
    strcpy(elem_name_198, "Element_Block_198");
    char* nodeset_name=new char[128];
    strcpy(nodeset_name, "Node_Set_1");
    char* sideset_name=new char[128];
    strcpy(sideset_name, "Side_Set_192");
    if (!strcmp(&model_file[0], "test_non_man.xmt_txt"))
    {
      GRIter gr_iter=GM_regionIter(model);
      pGeomEnt geom_rgn;
      while (geom_rgn=GRIter_next(gr_iter))
      {  
        if (GEN_tag(geom_rgn)==98)
          PUMI_Exodus_CreateElemBlk(geom_rgn, elem_name_98);
        else 
          PUMI_Exodus_CreateElemBlk(geom_rgn, elem_name_198);
      }
      GRIter_delete(gr_iter);

      GFIter gf_iter=GM_faceIter(model);
      pGeomEnt geom_face;
      while (geom_face=GFIter_next(gf_iter))
      {
        if (GEN_tag(geom_face)==1)
          PUMI_Exodus_CreateNodeSet(geom_face, nodeset_name);
        if (GEN_tag(geom_face)==192)
          PUMI_Exodus_CreateSideSet(geom_face, sideset_name);
      }
      GFIter_delete(gf_iter);
    }    
  }
#endif

  FMDB_Mesh_Create (model, mesh);

  int i, processid = getpid();

#if 0
  if (!SCUTIL_CommRank())
  {
    cout<<"Proc "<<SCUTIL_CommRank()<<">> pid "<<processid<<" Enter any digit...\n";
    cin>>i;
  }
  else
    cout<<"Proc "<<SCUTIL_CommRank()<<">> pid "<<processid<<" Waiting...\n";
#endif

  SCUTIL_Sync();

  SCUTIL_DspCurMem("INITIAL COST: ");
  SCUTIL_ResetRsrc();

//FIXME - need to look at this logic to read from a single vs. distributed file
distributedMesh = true;
  
//  if (FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], useSerialMesh))
//  if (FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], singlePEMeshRead))
  if (FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], distributedMesh))
  {
    *out<<"FAILED MESH LOADING - check mesh file or if number if input files are correct\n";
    FMDB_Mesh_Del(mesh);
    // SCUTIL_Finalize();
    ParUtil::Instance()->Finalize(0); // skip MPI_finalize 
    throw SCUtil_FAILURE;
  }

  *out<<endl;
  SCUTIL_DspRsrcDiff("MESH LOADING: ");
  FMDB_Mesh_DspStat(mesh);

  // generate node/element id for exodus compatibility
  PUMI_Exodus_Init(mesh); 

  //get mesh dim
  int mesh_dim;
  FMDB_Mesh_GetDim(mesh, &mesh_dim);

#ifdef DEBUG
  // check mesh validity
  int isValid=0;
  FMDB_Mesh_Verify(mesh, &isValid);
  if (!isValid)
  {
    PUMI_Exodus_Finalize(mesh); // should be called before mesh is deleted
    FMDB_Mesh_Del(mesh);
    // SCUTIL_Finalize();
    ParUtil::Instance()->Finalize(0); // skip MPI_finalize 
    throw SCUtil_FAILURE;
  }
#endif

  std::vector<pElemBlk> elem_blocks;
  PUMI_Exodus_GetElemBlk(mesh, elem_blocks);

  // Build a map to get the EB name given the index

  int numEB = elem_blocks.size(), EB_size;
  *out << "Found : " << numEB << " element blocks." << std::endl;
  std::vector<int> el_blocks;
  
  for (int eb=0; eb < numEB; eb++){
    string EB_name;
    PUMI_ElemBlk_GetName(elem_blocks[eb], EB_name);
    this->ebNameToIndex[EB_name] = eb;
    PUMI_ElemBlk_GetSize(mesh, elem_blocks[eb], &EB_size);
    el_blocks.push_back(EB_size);
  }

  // Set defaults for cubature and workset size, overridden in input file

  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get("Workset Size", 50);
  interleavedOrdering = params->get("Interleaved Ordering",true);
  allElementBlocksHaveSamePhysics = true; 
  hasRestartSolution = false;

  // No history available by default
  solutionFieldHistoryDepth = 0;

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  // Get number of elements per element block 
  // in calculating an upper bound on the worksetSize.

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Node sets
  std::vector<pNodeSet> node_sets;
  PUMI_Exodus_GetNodeSet(mesh, node_sets);

  std::vector<std::string> nsNames;

  for(int ns = 0; ns < node_sets.size(); ns++)
  {
    string NS_name;
    PUMI_NodeSet_GetName(node_sets[ns], NS_name);
    nsNames.push_back(NS_name);
  }

  // Side sets
  std::vector<pSideSet> side_sets;
  PUMI_Exodus_GetSideSet(mesh, side_sets);

  std::vector<std::string> ssNames;

  for(int ss = 0; ss < side_sets.size(); ss++)
  {
    string SS_name;
    PUMI_SideSet_GetName(side_sets[ss], SS_name);
    ssNames.push_back(SS_name);

  }

  // Construct MeshSpecsStruct
  vector<pMeshEnt> elements;
  if (!params->get("Separate Evaluators by Element Block",false)) 
  {
    // get elements in the first element block 
    PUMI_ElemBlk_GetElem (mesh, elem_blocks[0], elements);
    FMDB_EntTopo entTopo;
    FMDB_Ent_GetTopo(elements[0], (int*)(&entTopo));
    const CellTopologyData *ctd = getCellTopologyData(entTopo);
    string EB_name;
    PUMI_ElemBlk_GetName(elem_blocks[0], EB_name);
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(*ctd, mesh_dim, cub,
                               nsNames, ssNames, worksetSize, EB_name, 
                               this->ebNameToIndex, this->interleavedOrdering));

  }
  else {
    *out << "MULTIPLE Elem Block in FMDB: DO worksetSize[eb] max?? " << endl; 
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    int eb_size;
    std::string eb_name;
    for (int eb=0; eb<numEB; eb++) 
    {
      elements.clear();
      PUMI_ElemBlk_GetElem (mesh, elem_blocks[eb], elements);
      FMDB_EntTopo entTopo;
      FMDB_Ent_GetTopo(elements[0], (int*)(&entTopo)); // get topology of first element in element block[eb]
      const CellTopologyData *ctd = getCellTopologyData(entTopo);
      string EB_name;
      PUMI_ElemBlk_GetName(elem_blocks[eb], EB_name);
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(*ctd, mesh_dim, cub,
                                              nsNames, ssNames, worksetSize, EB_name,
                                              this->ebNameToIndex, this->interleavedOrdering));
      PUMI_ElemBlk_GetSize(mesh, elem_blocks[eb], &eb_size);
      PUMI_ElemBlk_GetName(elem_blocks[eb], eb_name);
    } // for
  } // else

  // set residual, solution field tags
  FMDB_Mesh_CreateTag (mesh, "residual", SCUtil_DBL, neq, residual_field_tag);
  FMDB_Mesh_CreateTag (mesh, "solution", SCUtil_DBL, neq, solution_field_tag);
}

Albany::FMDBMeshStruct::~FMDBMeshStruct()
{
  // delete residual, solution field tags
  FMDB_Mesh_DelTag (mesh, residual_field_tag, 1);
  FMDB_Mesh_DelTag (mesh,  solution_field_tag, 1);

  // delete exodus data
  PUMI_Exodus_Finalize(mesh);
  // delete mesh and finalize
  FMDB_Mesh_Del (mesh);
  ParUtil::Instance()->Finalize(0); // skip MPI_finalize 
}

const CellTopologyData *
Albany::FMDBMeshStruct::getCellTopologyData(const FMDB_EntTopo topo){

  switch(topo){

  case FMDB_POINT:

    return shards::getCellTopologyData< shards::Particle >();

  case FMDB_LINE:

    return shards::getCellTopologyData< shards::Line<2> >();

  case FMDB_TRI:

    return shards::getCellTopologyData< shards::Triangle<3> >();

  case FMDB_QUAD:

    return shards::getCellTopologyData< shards::Quadrilateral<4> >();

  case FMDB_TET:

    return shards::getCellTopologyData< shards::Tetrahedron<4> >();

  case FMDB_PYRAMID:

    return shards::getCellTopologyData< shards::Pyramid<5> >();

  case FMDB_HEX:

    return shards::getCellTopologyData< shards::Hexahedron<8> >();

// Not supported right now
  case FMDB_POLYGON:
  case FMDB_POLYHEDRON:
  case FMDB_PRISM:
  case FMDB_SEPTA:
  case FMDB_ALLTOPO:
  default:

    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                  std::endl << "Error FMDB mesh cell topology:  " <<
                  "Unsupported topology encountered " << topo << std::endl);
  }
}

void
Albany::FMDBMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize_)
{

  // Set the number of equation present per node. Needed by Albany_FMDBDiscretization.

  neq = neq_;

/*
  //Start STK stuff
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  proc_rank_field = & metaData->declare_field< IntScalarFieldType >( "proc_rank" );
  solution_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Solution Name", "solution"));
#ifdef ALBANY_LCM
  residual_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Residual Name", "residual"));
#endif

  stk::mesh::put_field( *coordinates_field , metaData->node_rank() , metaData->universal_part(), numDim );
  // Processor rank field, a scalar
  stk::mesh::put_field( *proc_rank_field , metaData->element_rank() , metaData->universal_part());
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
#ifdef ALBANY_LCM
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );
#endif
  
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*proc_rank_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#ifdef ALBANY_LCM
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
#endif
#endif
*/

  // Code to parse the vector of StateStructs and create STK fields
  for (std::size_t i=0; i<sis->size(); i++) {
    Albany::StateStruct& st = *((*sis)[i]);
    std::vector<int>& dim = st.dim;
    if (dim.size() == 2 && st.entity=="QuadPoint") {
      double *s_mem = new double[dim[1]]; // 1D array num QP long
      qpscalar_mem.push_back(s_mem); // save the mem for deletion
      qpscalar_name.push_back(st.name); // save the name
      qpscalar_states.push_back( new QPScalarFieldType(s_mem, dim[1]));
      cout << "NNNN qps field name " << st.name << " size : " << dim[1] << endl;
    }
    else if (dim.size() == 3 && st.entity=="QuadPoint") {
      double *v_mem = new double[dim[1]*dim[2]]; // 1D array num QP * dim long
      qpvector_mem.push_back(v_mem); // save the mem for deletion
      qpvector_name.push_back(st.name); // save the name
      qpvector_states.push_back( new QPVectorFieldType(v_mem, dim[1], dim[2]));
      cout << "NNNN qpv field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << endl;
    }
    else if (dim.size() == 4 && st.entity=="QuadPoint") {
      double *t_mem = new double[dim[1]*dim[2] * dim[3]]; // 1D array num QP * dim * dim long
      qptensor_mem.push_back(t_mem); // save the mem for deletion
      qptensor_name.push_back(st.name); // save the name
      qptensor_states.push_back( new QPTensorFieldType(t_mem, dim[1], dim[2], dim[3]));
      cout << "NNNN qpt field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << " dim[3] : " << dim[3] << endl;
    }
    else if ( dim.size() == 1 && st.entity=="ScalarValue" ) {
      scalarValue_states.push_back(st.name);
    }
    else TEUCHOS_TEST_FOR_EXCEPT(dim.size() < 2 || dim.size()>4 || st.entity!="QuadPoint");

  }
  
/*
  // Exodus is only for 2D and 3D. Have 1D version as well
  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");

  exoOutputInterval = params->get<int>("Exodus Write Interval", 1);
*/
  
}


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::FMDBMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int Albany::FMDBMeshStruct::computeWorksetSize(const int worksetSizeMax,
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

void
Albany::FMDBMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_TEST_FOR_EXCEPT(step < 0 || step >= solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
//  stk::io::process_input_request(*mesh_data, *bulkData, index);
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::FMDBMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = rcp(new Teuchos::ParameterList("Valid FMDBParams"));

  validPL->set<std::string>("FMDB Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("FMDB Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
  validPL->set<int>("FMDB Write Interval", 3, "Step interval to write solution data to Exodus file");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields, 
                     "Fields to pick up from the restart file when restarting");

  validPL->set<string>("FMDB Input File Name", "", "File Name For FMDB Mesh Input");
  validPL->set<string>("FMDB Output File Name", "", "File Name For FMDB Mesh Output");

  validPL->set<string>("Acis Model Input File Name", "", "File Name For ACIS Model Input");
  validPL->set<string>("Parasolid Model Input File Name", "", "File Name For PARASOLID Model Input");

  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for initial guess/condition.");
  validPL->set<double>("Restart Time", 1.0, "Exodus solution time to read for initial guess/condition.");
  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");

  validPL->set<int>("Number of parts", 1, "Number of parts");
  validPL->set<int>("Number of migrations", 0, "Number of migrations");
  validPL->set<int>("Number of individual migrations", 0, "Number of individual migrations");
  validPL->set<double>("Imbalance tolerance", 1.03, "Imbalance tolerance");
  validPL->set<bool>("Construct pset", false, "Construct pset");
  validPL->set<bool>("Call serial global partition", false, "Call serial global partition");

  validPL->set<string>("LB Method", "", "Method used to load balance mesh (default \"ParMETIS\")");
  validPL->set<string>("LB Approach", "", "Approach used to load balance mesh (default \"PartKway\")");

  return validPL;
}
