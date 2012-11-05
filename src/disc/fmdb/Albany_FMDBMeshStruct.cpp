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

const std::string _string_LBMethod[5] = {"RCB", "RIB", "GRAPH", "HYPERGRAPH", "ParMETIS"};
const std::string _string_LBApproach[8] = {"Partition", "Repartition", "Refine", "PartKway", 
                                 "PartGeom", "PartGeomKWay", "AdaptiveRepart", "RefineKway"};

Albany::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(params->get<bool>("Use Serial Mesh", false))
{

  SCUTIL_Init(Albany::getMpiCommFromEpetraComm(*comm));

  params->validateParameters(*getValidDiscretizationParameters(),0);

  int numMigr, numPart, numIMigr;

// Read in or initialize parameters

  numPart = params->get<int>("Number of parts", 1);

  numMigr = params->get<int>("Number of migrations", 0);

  numIMigr = params->get<int>("Number of individual migrations", 0);

  // Process load balancing method

  std::string LB_method_string = 
     params->get<string>("LB Method", "");

  if(LB_method_string == _string_LBMethod[FMDB_RCB])

    LB_method = FMDB_RCB;

  else if(LB_method_string == _string_LBMethod[FMDB_RIB])
    
    LB_method = FMDB_RIB;

  else if(LB_method_string == _string_LBMethod[FMDB_GRAPH])
    
    LB_method = FMDB_GRAPH;

  else if(LB_method_string == _string_LBMethod[FMDB_HYPERGRAPH])
    
    LB_method = FMDB_HYPERGRAPH;

  else // the default
    
    LB_method = FMDB_PARMETIS;


  // Process load balancing approach

  std::string LB_approach_string = 
     params->get<string>("LB Approach", "");

  if(LB_approach_string == _string_LBApproach[FMDB_PARTITION])

    LB_approach = FMDB_PARTITION;

  else if(LB_approach_string == _string_LBApproach[FMDB_REPARTITION])

    LB_approach = FMDB_REPARTITION;

  else if(LB_approach_string == _string_LBApproach[FMDB_REFINE])

    LB_approach = FMDB_REFINE;

  else if(LB_approach_string == _string_LBApproach[PartGeom])

    LB_approach = PartGeom;

  else if(LB_approach_string == _string_LBApproach[PartGeomKWay])

    LB_approach = PartGeomKWay;

  else if(LB_approach_string == _string_LBApproach[AdaptiveRepart])

    LB_approach = AdaptiveRepart;

  else if(LB_approach_string == _string_LBApproach[RefineKway])

    LB_approach = RefineKway;

  else  // the default

    LB_approach = PartKway;

  double imbal_tol = params->get<double>("Imbalance tolerance", 1.03);

  bool construct_pset = params->get<bool>("Construct pset", false);

  bool call_serial = params->get<bool>("Call serial global partition", false);
 
  std::string mesh_file = params->get<string>("FMDB Input File Name");

  *out<<"************************************************************************\n";
  *out<<"[INPUT]\n";
  *out<<"\tdistributed mesh? ";
  if (useSerialMesh) *out<<"YES\n";
  else *out<<"NO\n";
  *out<<"\t#parts per proc: "<<numPart<<endl;
  *out<<"\t# migration steps: "<<numMigr<<endl;
  *out<<"\t# individual migration steps: "<<numIMigr<<endl;
  *out<<"\tLB_method="<<_string_LBMethod[LB_method]<<endl;
  *out<<"\tLB_approach="<<_string_LBApproach[LB_approach]<<endl;
  *out<<setprecision(5)<<"\timbalance_tol="<<imbal_tol<<endl;
  *out<<"\tconstruct p-set? ";
  if (construct_pset) *out<<"YES\n";
  else *out<<"NO\n";

    SCUTIL_DspSysInfo();
    *out<<"************************************************************************\n\n";  

  // **************************************************************
  // TEST 1: MESH FILE I/O
  // **************************************************************

  *out<<"\n***** MESH/PART FUNCTIONS *****\n"; 

  model=NULL;

  FMDB_Mesh_Create (model, mesh);
  SCUTIL_DspCurMem("INITIAL COST: ");

  SCUTIL_ResetRsrc();

  FMDB_Mesh_SetPtnParam (mesh, LB_method, LB_approach, imbal_tol, 0);

  if (FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], useSerialMesh))
  {
    *out<<"FAILED MESH LOADING - check mesh file or if number if input files are correct\n";
    FMDB_Mesh_Del(mesh);
    SCUTIL_Finalize();
    throw SCUtil_FAILURE;
  }

  *out<<endl;
  SCUTIL_DspRsrcDiff("MESH LOADING: ");
  FMDB_Mesh_DspStat(mesh);
 
  int isValid=0;
  FMDB_Mesh_Verify(mesh, &isValid);
  if (!isValid)
  {
    FMDB_Mesh_Del(mesh);
    SCUTIL_Finalize();
    throw SCUtil_FAILURE;
  }

  if (construct_pset)
  {
    Construct_Pset(mesh);
    FMDB_Mesh_DspNumEnt(mesh);
  }

  FMDB_Mesh_GetPart(mesh, 0, part);

  // Mesh is read in and ready to go

  // Get problem dimension

  FMDB_Part_GetDim(part, &numDim);

  *out << numDim << std::endl;

exit(1);


  // Build a map to get the EB name given the index
#if 0

  for (int eb=0; eb<numEB; eb++) 

    this->ebNameToIndex[partVec[eb]->name()] = eb;

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {

    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                               nsNames, ssNames, worksetSize, partVec[0]->name(), 
                               this->ebNameToIndex, this->interleavedOrdering));

  }
  else {

    *out << "MULTIPLE Elem Block in Ioss: DO worksetSize[eb] max?? " << endl; 
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    for (int eb=0; eb<numEB; eb++) {
      const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                                                nsNames, ssNames, worksetSize, partVec[eb]->name(), 
                                                this->ebNameToIndex, this->interleavedOrdering));
      *out << "el_block_size[" << eb << "] = " << el_blocks[eb] << "   name  " << partVec[eb]->name() << endl; 
    }

  }
#endif



#if 0

  pMeshDataId double_arr_2  = MD_createMeshDataId("2 double array", SCUtil_DBL, 2);
  pMeshDataId double_arr_4  = MD_createMeshDataId("4 double array", SCUtil_DBL, 4);
  assert (FMDB_Util::Instance()->typeMeshDataId(double_arr_2)==SCUtil_DBL && FMDB_Util::Instance()->sizeMeshDataId(double_arr_2)==2);
  assert (FMDB_Util::Instance()->typeMeshDataId(double_arr_4)==SCUtil_DBL && FMDB_Util::Instance()->sizeMeshDataId(double_arr_4)==4);
 
  if (SCUTIL_CommSize()==1)
  {
    // attach double array
    for(mPart::iterall vtx_it = mesh->getPart(0)->beginall(0) ; vtx_it!=mesh->getPart(0)->endall(0) ; ++vtx_it)
    {
      double* dbl_arr = new double[2];
      dbl_arr[0] = ((double)((*vtx_it)->getId()))*1.1;
      dbl_arr[1] = ((double)((*vtx_it)->getId()))*2.2;

      EN_attachDataPtr(*vtx_it, double_arr_2 , (void*)dbl_arr);
  
      double* dbl_arr_4 = new double[4];
      for (int i=0; i<4; ++i)
        dbl_arr_4[i] = (double)((i+1)*2.2);

      EN_attachDataPtr(*vtx_it, double_arr_4 , (void*)dbl_arr_4);
    }
    MD_migrateMeshDataId(double_arr_2, FMDB_VERTEX, 1);
    MD_migrateMeshDataId(double_arr_2, FMDB_VERTEX, 0);
    MD_migrateMeshDataId(double_arr_2, FMDB_VERTEX, 1);
    MD_migrateMeshDataId(double_arr_4, FMDB_VERTEX, 1);
  }

  // **************************************************************
  // TEST 2: PART BOUNDARY CHECK
  // **************************************************************
//  CHECK("\n* Checking part boundary...", !TEST_PART_BOUNDARY (mesh), SCUTIL_CommRank());
//  SCUTIL_Sync();

  // **************************************************************
  // TEST 3: MIGRATION TESTING WITH SINGLE PART
  // **************************************************************
  CHECK("* TESTING MIGRATION ...", !TEST_MIGRATION(mesh, numPart, numMigr, call_serial), SCUTIL_CommRank());
 
  if (print_all)
  {
    for (mMesh::partIter part_it=mesh->partBegin(); part_it!=mesh->partEnd(); ++part_it)
      (*part_it)->printAll();
  }


  // **************************************************************
  // TEST 4: INDIVIDUAL MIGRATION TESTING
  // **************************************************************
  if (numIMigr)
    CHECK( "\n* TESTING INDIVIDUAL MIGRATION...", !TEST_ENT_MIGRATION(mesh, numIMigr), SCUTIL_CommRank());

  // **************************************************************
  // TEST 5: TESTING AUTO TAG MIGRATION - DOUBLE ARRAY
  // **************************************************************

  if (SCUTIL_CommSize()==1)
  {
    double* dbl_arr_back = new double[2];
    int arr_size;
    for (mMesh::partIter part_it=mesh->partBegin(); part_it!=mesh->partEnd(); ++part_it)
      for(mPart::iterall vtx_it = (*part_it)->beginall(0) ; vtx_it!=(*part_it)->endall(0) ; ++vtx_it)
      {
        assert(EN_getDataPtr(*vtx_it,  double_arr_2, (void**)&dbl_arr_back));
        assert(EN_getDataPtr(*vtx_it,  double_arr_4, (void**)&dbl_arr_back));
    }

    for (mMesh::partIter part_it=mesh->partBegin(); part_it!=mesh->partEnd(); ++part_it)
      for(mPart::iterall vtx_it = (*part_it)->beginall(0) ; vtx_it!=(*part_it)->endall(0) ; ++vtx_it)
      {
        EN_getDataPtr(*vtx_it,  double_arr_2, (void**)&dbl_arr_back);
        delete [] dbl_arr_back;
        (*vtx_it)->deleteData(double_arr_2);

        EN_getDataPtr(*vtx_it,  double_arr_4, (void**)&dbl_arr_back);
        delete [] dbl_arr_back;
        (*vtx_it)->deleteData(double_arr_4);
    }

    for (mMesh::partIter part_it=mesh->partBegin(); part_it!=mesh->partEnd(); ++part_it)
      for(mPart::iterall vtx_it = (*part_it)->beginall(0) ; vtx_it!=(*part_it)->endall(0) ; ++vtx_it)
      {
        assert(!EN_getDataPtr(*vtx_it,  double_arr_2, (void**)&dbl_arr_back));
        assert(!EN_getDataPtr(*vtx_it,  double_arr_4, (void**)&dbl_arr_back));
      }
  }
  FMDB_Mesh_Del (mesh);
  SCUTIL_Finalize();
  system ("rm -rf *part_p*sms");
  return 0;
#endif

}


void
Albany::FMDBMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{

}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::FMDBMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}


void
Albany::FMDBMeshStruct::Construct_Pset(pMeshMdl mesh)
{
  int mesh_dim, isEnd, num;
  FMDB_Mesh_GetDim(mesh, &mesh_dim);

  for (mMesh::partIter part_it=mesh->partBegin(); part_it!=mesh->partEnd(); ++part_it)
    for(mPart::iterall rgn_it = (*part_it)->beginall(mesh_dim) ; rgn_it!=(*part_it)->endall(mesh_dim) ; ++rgn_it)
       FMDB_Ent_SetWeight(*rgn_it,  2.0);

  pMeshEnt ent;
  pEntSet set;

  for (mMesh::partIter pit=mesh->partBegin(); pit!=mesh->partEnd(); ++pit)
  {
    pPart part = *pit;
    pEntSet psets[part->getId()+1];
    for (int i=0; i<part->getId()+1; ++i)
      FMDB_Set_Create (mesh, part, FMDB_PSET, psets[i]);

    vector<pEntSet> vec_set;
    FMDB_Part_GetSet(part, vec_set);
    assert (vec_set.size()==part->getId()+1);

    pPartEntIter entIter;  
    isEnd = FMDB_PartEntIter_Init (part, mesh_dim, FMDB_ALLTOPO, entIter);
    while (!isEnd)
    {
      isEnd = FMDB_PartEntIter_GetNext(entIter, ent);
      if(isEnd) break; 
      if (rand()%4==1) // random picking of partition objects up to approx. 1/4 of po-ent
        FMDB_Set_AddEnt(psets[rand()%(part->getId()+1)], ent);
    }
    FMDB_PartEntIter_Del (entIter);
  } // for mMesh::partIter

  double weight;
  for (mMesh::partIter pit=mesh->partBegin(); pit!=mesh->partEnd(); ++pit)
  {
    pPart part = *pit;
    pPartSetIter pset_iter;
    isEnd = FMDB_PartSetIter_Init (part, pset_iter); 
    while (!isEnd)
    {
      isEnd = FMDB_PartSetIter_GetNext(part, pset_iter, set);
      if(isEnd) break; 
      FMDB_Set_GetWeight(set, &weight);
      FMDB_Set_GetNumEnt (set, &num);
      assert(weight == num*2.0);
    }
    FMDB_PartSetIter_Del (pset_iter);
  }
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::FMDBMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = rcp(new Teuchos::ParameterList("Valid FMDBParams"));

  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");

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
