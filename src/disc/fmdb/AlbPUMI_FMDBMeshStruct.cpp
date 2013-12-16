//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "mMesh.h"
#include "maCallback.h"

#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TwoDArray.hpp"
#include <Shards_BasicTopologies.hpp>

#include "SCUtil.h"

#include "AdaptTypes.h"
#include "MeshAdapt.h"
#include "PWLinearSField.h"

#define DEBUG 1

struct unique_string {
   std::vector<std::string> operator()(std::vector<std::string> sveca, const std::vector<std::string> svecb){
      std::vector<std::string> outvec;

      sveca.insert(sveca.end(), svecb.begin(), svecb.end());

      std::sort(sveca.begin(), sveca.end());
      std::vector<std::string>::iterator new_end = std::unique(sveca.begin(), sveca.end());
      for(std::vector<std::string>::iterator it = sveca.begin(); it != new_end; ++it)
        outvec.push_back(*it);

      return outvec;

   }
};

static double element_size = 0.0;

int sizefieldfunc(pPart part, pSField field, void *vp){

  pMeshEnt vtx;
  double h[3], dirs[3][3], xyz[3];

  std::cout << element_size << std::endl;

  pPartEntIter vtx_iter;
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);
  while (FMDB_PartEntIter_GetNext(vtx_iter, vtx)==SCUtil_SUCCESS)
  {
    h[0] = element_size;
    h[1] = element_size;
    h[2] = element_size;

    dirs[0][0]=1.0;
    dirs[0][1]=0.;
    dirs[0][2]=0.;
    dirs[1][0]=0.;
    dirs[1][1]=1.0;
    dirs[1][2]=0.;
    dirs[2][0]=0.;
    dirs[2][1]=0.;
    dirs[2][2]=1.0;

    ((PWLsfield *)field)->setSize(vtx,dirs,h);
  }
  FMDB_PartEntIter_Del(vtx_iter);

  double beta[]={1.5,1.5,1.5};
  ((PWLsfield *)field)->anisoSmooth(beta);

  return 1;
}

AlbPUMI::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // fmdb skips mpi initialization if it's already initialized
  SCUTIL_Init(Albany::getMpiCommFromEpetraComm(*comm));
  params->validateParameters(*getValidDiscretizationParameters(),0);

  std::string mesh_file = params->get<std::string>("FMDB Input File Name");
  outputFileName = params->get<std::string>("FMDB Output File Name", "");
  outputInterval = params->get<int>("FMDB Write Interval", 1); // write every time step default
  if (params->get<bool>("Call serial global partition"))
    useDistributedMesh=false;
  else
    useDistributedMesh=true;

  compositeTet = params->get<bool>("Use Composite Tet 10", false);


#if 0
  *out<<"************************************************************************\n";
  *out<<"[INPUT]\n";
  *out<<"\tdistributed mesh? ";
  if (useDistributedMesh) *out<<"YES\n";
  else *out<<"NO\n";
  *out<<"\t#parts per proc: "<<numPart<<std::endl;
  SCUTIL_DspSysInfo();
  *out<<"************************************************************************\n\n";
#endif

  // create a model and load
  model = NULL; // default is no model

  PUMI_Geom_RegisterMesh();

  if(params->isParameter("Acis Model Input File Name")){ // User has an Acis model

    std::string model_file = params->get<std::string>("Acis Model Input File Name");
    PUMI_Geom_RegisterAcis();
    PUMI_Geom_LoadFromFile(model, model_file.c_str());
  }

  if(params->isParameter("Parasolid Model Input File Name")){ // User has a Parasolid model

    std::string model_file = params->get<std::string>("Parasolid Model Input File Name");
    PUMI_Geom_RegisterParasolid();
    PUMI_Geom_LoadFromFile(model, model_file.c_str());
  }

  if(params->isParameter("Mesh Model Input File Name")){ // User has a meshModel model

    std::string model_file = params->get<std::string>("Mesh Model Input File Name");
    PUMI_Geom_LoadFromFile(model, model_file.c_str());
  }

  if ( model == NULL && SCUTIL_CommRank() == 0 ) {

    fprintf(stderr, "ERROR: Null model is not supported\n");
    exit(1);
  }

  FMDB_Mesh_Create (model, mesh);
  FMDB_Mesh_GetGeomMdl(mesh, model); // this is needed for null model

#if 0
  int i, processid = getpid();
  if (!SCUTIL_CommRank())
  {
    cout<<"Proc "<<SCUTIL_CommRank()<<">> pid "<<processid<<" Enter any digit...\n";
    cin>>i;
  }
  else
    cout<<"Proc "<<SCUTIL_CommRank()<<">> pid "<<processid<<" Waiting...\n";
  SCUTIL_Sync();
#endif

  SCUTIL_DspCurMem("INITIAL COST: ");
  SCUTIL_ResetRsrc();

  if (FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], useDistributedMesh))
  {
    *out<<"FAILED MESH LOADING - check mesh file or if number if input files are correct\n";
    FMDB_Mesh_Del(mesh);
    // SCUTIL_Finalize();
    ParUtil::Instance()->Finalize(0); // skip MPI_finalize
    throw SCUtil_FAILURE;
  }

  *out << std::endl;
  SCUTIL_DspRsrcDiff("MESH LOADING: ");
  FMDB_Mesh_DspSize(mesh);

//  FMDB_Mesh_GetGeomMdl(mesh, model); // this is needed for null model

  if(params->isParameter("Element Block Associations")){ // User has specified associations in the input file

    // Get element block associations from input file
    Teuchos::TwoDArray< std::string > EBAssociations;

    EBAssociations = params->get<Teuchos::TwoDArray<std::string> >("Element Block Associations");

    TEUCHOS_TEST_FOR_EXCEPTION( !(2 == EBAssociations.getNumRows()),
			      Teuchos::Exceptions::InvalidParameter,
			      "Error in specifying element block associations in input file" );

    int nEBAssoc = EBAssociations.getNumCols();

    for(size_t eb = 0; eb < nEBAssoc; eb++){
      *out << "Element block \"" <<  EBAssociations(1, eb).c_str() << "\" matches mesh region : "
           << EBAssociations(0, eb).c_str() << std::endl;
    }

    FMDB_Mesh_GetGeomMdl (mesh, model);
    GRIter gr_iter = GM_regionIter(model);
    pGeomEnt geom_rgn;
    while ((geom_rgn = GRIter_next(gr_iter)) != NULL)
    {
      for(size_t eblock = 0; eblock < nEBAssoc; eblock++){
        if (GEN_tag(geom_rgn) == atoi(EBAssociations(0, eblock).c_str()))
          PUMI_Exodus_CreateElemBlk(geom_rgn, EBAssociations(1, eblock).c_str());
      }
    }
    GRIter_delete(gr_iter);
  }


  if(params->isParameter("Node Set Associations")){ // User has specified associations in the input file

    // Get node set associations from input file
    Teuchos::TwoDArray< std::string > NSAssociations;

    NSAssociations = params->get<Teuchos::TwoDArray<std::string> >("Node Set Associations");

    TEUCHOS_TEST_FOR_EXCEPTION( !(2 == NSAssociations.getNumRows()),
			      Teuchos::Exceptions::InvalidParameter,
			      "Error in specifying node set associations in input file" );

    int nNSAssoc = NSAssociations.getNumCols();

    for(size_t ns = 0; ns < nNSAssoc; ns++){
      *out << "Node set \"" << NSAssociations(1, ns).c_str() << "\" matches geometric entity : "
           << NSAssociations(0, ns).c_str() << std::endl;
    }

    GFIter gf_iter=GM_faceIter(model);
    pGeomEnt geom_face;
    while ((geom_face=GFIter_next(gf_iter)) != NULL)
    {
      for(size_t ns = 0; ns < nNSAssoc; ns++){
        if (GEN_tag(geom_face) == atoi(NSAssociations(0, ns).c_str())){
          PUMI_Exodus_CreateNodeSet(geom_face, NSAssociations(1, ns).c_str());
        }
      }
    }
    GFIter_delete(gf_iter);
    
    GEIter ge_iter=GM_edgeIter(model);
    pGeomEnt geom_edge;
    while (geom_edge=GEIter_next(ge_iter))
    {
      for (size_t ns = 0; ns < nNSAssoc; ns++){
	if (GEN_tag(geom_edge) == atoi(NSAssociations(0, ns).c_str())){
	  PUMI_Exodus_CreateNodeSet(geom_edge, NSAssociations(1, ns).c_str());
	}
      }
    }
    GEIter_delete(ge_iter);

    GVIter gv_iter=GM_vertexIter(model);
    pGeomEnt geom_vertex;
    while (geom_vertex=GVIter_next(gv_iter))
    {
      for (size_t ns = 0; ns < nNSAssoc; ns++){
	if (GEN_tag(geom_vertex) == atoi(NSAssociations(0, ns).c_str())){
	  PUMI_Exodus_CreateNodeSet(geom_vertex, NSAssociations(1, ns).c_str());
	}
      }
    }
    GVIter_delete(gv_iter);
    
  }
  
  if(params->isParameter("Side Set Associations")){ // User has specified associations in the input file

    // Get side set block associations from input file
    Teuchos::TwoDArray< std::string > SSAssociations;

    SSAssociations = params->get<Teuchos::TwoDArray<std::string> >("Side Set Associations");

    TEUCHOS_TEST_FOR_EXCEPTION( !(2 == SSAssociations.getNumRows()),
			      Teuchos::Exceptions::InvalidParameter,
			      "Error in specifying side set associations in input file" );

    int nSSAssoc = SSAssociations.getNumCols();

    for(size_t ss = 0; ss < nSSAssoc; ss++){
      *out << "Side set \"" << SSAssociations(1, ss).c_str() << "\" matches geometric face : "
           << SSAssociations(0, ss).c_str() << std::endl;
    }


    GFIter gf_iter=GM_faceIter(model);
    pGeomEnt geom_face;
    while ((geom_face = GFIter_next(gf_iter)) != NULL)
    {
      for(size_t ss = 0; ss < nSSAssoc; ss++){
        if (GEN_tag(geom_face) == atoi(SSAssociations(0, ss).c_str()))
          PUMI_Exodus_CreateSideSet(geom_face, SSAssociations(1, ss).c_str());
      }
    }
    GFIter_delete(gf_iter);
  }

  // Resize mesh after input if indicated in the input file
  if(params->isParameter("Resize Input Mesh Element Size")){ // User has indicated a desired element size in input file

    element_size = params->get<double>("Resize Input Mesh Element Size", 0.1);
    int num_iters = params->get<int>("Max Number of Mesh Adapt Iterations", 1);

      // Do basic uniform refinement
      /** Type of the size field:
          - Application - the size field will be provided by the application (default).
          - TagDriven - tag driven size field.
          - Analytical - analytical size field.  */
      /** Type of model:
          - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
      */

      meshAdapt *rdr = new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 0 );

      /** void meshAdapt::run(int niter,    // specify the maximum number of iterations
                        int flag,           // indicate if a size field function call is available
                        adaptSFunc sizefd)  // the size field function call  */

      ma::AlbanyCallback *callback = new ma::AlbanyCallback(rdr, mesh);

      rdr->run (num_iters, 1, sizefieldfunc);
      FMDB_Mesh_DspSize(mesh);
      delete rdr;
      delete callback;
  }

  // generate node/element id for exodus compatibility
  PUMI_Exodus_Init(mesh);

  //get mesh dim
  FMDB_Mesh_GetDim(mesh, &numDim);

/* mesh verification overwrites mesh entity id so commented out temporarily
   FMDB will be updated to use different id for validity check
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
*/
  std::vector<pElemBlk> elem_blocks;
  PUMI_Exodus_GetElemBlk(mesh, elem_blocks);

  // Build a map to get the EB name given the index

  int numEB = elem_blocks.size(), EB_size;
  *out <<"["<<SCUTIL_CommRank()<< "] Found : " << numEB << " element blocks." << std::endl;
  std::vector<int> el_blocks;

  for (int eb=0; eb < numEB; eb++){
    std::string EB_name;
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

  *out <<"["<<SCUTIL_CommRank()<< "] Workset size is: " << worksetSize << std::endl;

  // Node sets
  std::vector<pNodeSet> node_sets;
  PUMI_Exodus_GetNodeSet(mesh, node_sets);

  std::vector<std::string> localNsNames;

  for(int ns = 0; ns < node_sets.size(); ns++)
  {
    std::string NS_name;
    PUMI_NodeSet_GetName(node_sets[ns], NS_name);
    localNsNames.push_back(NS_name);
  }

  // Allreduce the node set names
  boost::mpi::all_reduce<std::vector<std::string> >(
                 boost::mpi::communicator(Albany::getMpiCommFromEpetraComm(*comm), boost::mpi::comm_attach),
                 localNsNames, nsNames, unique_string());

  // Side sets
  std::vector<pSideSet> side_sets;
  PUMI_Exodus_GetSideSet(mesh, side_sets);

  std::vector<std::string> localSsNames;
  int status;

  for(int ss = 0; ss < side_sets.size(); ss++)
  {
    std::string SS_name;
    status = PUMI_SideSet_GetName(side_sets[ss], SS_name);
    if(status != PUMI_SUCCESS){
       *out << "Warning: found a side set in the mesh/geom file, but cannot determine its name." << std::endl;
       *out << "     Something is probably wrong, but ignoring the side set and continuing anyway!" << std::endl;
       continue;
    }
    localSsNames.push_back(SS_name);
  }

  // Allreduce the side set names
  boost::mpi::all_reduce<std::vector<std::string> >(
                 boost::mpi::communicator(Albany::getMpiCommFromEpetraComm(*comm), boost::mpi::comm_attach),
                 localSsNames, ssNames, unique_string());

  // compute topology of the first element of the part
  FMDB_EntTopo entTopo;
  pPartEntIter elem_iter;
  pMeshEnt elem;
  pPart part;
  FMDB_Mesh_GetPart(mesh, 0, part);
  FMDB_PartEntIter_Init(part, numDim, FMDB_ALLTOPO, elem_iter);
  FMDB_PartEntIter_GetNext(elem_iter, elem); // get the first element of the part
  FMDB_PartEntIter_Del(elem_iter);
  FMDB_Ent_GetTopo(elem, (int*)(&entTopo));
  const CellTopologyData *ctd = getCellTopologyData(entTopo);

  // Allreduce the side set names
  boost::mpi::all_reduce<std::vector<std::string> >(
                 boost::mpi::communicator(Albany::getMpiCommFromEpetraComm(*comm), boost::mpi::comm_attach),
                 localSsNames, ssNames, unique_string());

  // Construct MeshSpecsStruct
  std::vector<pMeshEnt> elements;
  if (!params->get("Separate Evaluators by Element Block",false))
  {
    // get elements in the first element block
    PUMI_ElemBlk_GetElem (mesh, elem_blocks[0], elements);
    if (elements.size())
    {
      FMDB_Ent_GetTopo(elements[0], (int*)(&entTopo)); // get topology of first element in element block[eb]
      ctd = getCellTopologyData(entTopo);              // otherwise, the use the topology of the first element in the part
    }
    std::string EB_name;
    PUMI_ElemBlk_GetName(elem_blocks[0], EB_name);
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(*ctd, numDim, cub,
                               nsNames, ssNames, worksetSize, EB_name,
                               this->ebNameToIndex, this->interleavedOrdering));

  }
  else
  {
    *out <<"["<<SCUTIL_CommRank()<< "] MULTIPLE Elem Block in FMDB: DO worksetSize[eb] max?? " << std::endl;
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    int eb_size;
    std::string eb_name;
    for (int eb=0; eb<numEB; eb++)
    {
      elements.clear();
      PUMI_ElemBlk_GetElem (mesh, elem_blocks[eb], elements);
      if (elements.size())
      {
        FMDB_Ent_GetTopo(elements[0], (int*)(&entTopo)); // get topology of first element in element block[eb]
        ctd = getCellTopologyData(entTopo);
      }
      std::string EB_name;
      PUMI_ElemBlk_GetName(elem_blocks[eb], EB_name);
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(*ctd, numDim, cub,
                                              nsNames, ssNames, worksetSize, EB_name,
                                              this->ebNameToIndex, this->interleavedOrdering));
      PUMI_ElemBlk_GetSize(mesh, elem_blocks[eb], &eb_size);
      PUMI_ElemBlk_GetName(elem_blocks[eb], eb_name);
    } // for
  } // else

}

AlbPUMI::FMDBMeshStruct::~FMDBMeshStruct()
{
  // turn off auto-migration and delete residual, solution field tags
  if ( FMDB_Mesh_FindTag(mesh,"residual",residual_field_tag) == 0 ) {
    FMDB_Tag_SetAutoMigrOff (mesh, residual_field_tag, FMDB_VERTEX);
    FMDB_Mesh_DelTag (mesh, residual_field_tag, 1);
  }  
  if ( FMDB_Mesh_FindTag(mesh,"solution",solution_field_tag) == 0 ) {
    FMDB_Tag_SetAutoMigrOff (mesh, solution_field_tag, FMDB_VERTEX);
    FMDB_Mesh_DelTag (mesh,  solution_field_tag, 1);
  }

  // delete exodus data
  PUMI_Exodus_Finalize(mesh);
  // delete mesh and finalize
  FMDB_Mesh_Del (mesh);
  ParUtil::Instance()->Finalize(0); // skip MPI_finalize
}

const CellTopologyData *
AlbPUMI::FMDBMeshStruct::getCellTopologyData(const FMDB_EntTopo topo){

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
AlbPUMI::FMDBMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize_)
{

  // Set the number of equation present per node. Needed by AlbPUMI_FMDBDiscretization.

  neq = neq_;

  // set the nodal_data_block in the base clase to point at the one in stateInfoStruct
  this->nodal_data_block = sis->getNodalDataBlock();

  // create residual, solution field tags and turn on auto migration
  FMDB_Mesh_CreateTag (mesh, "residual", SCUtil_DBL, neq, residual_field_tag);
  FMDB_Mesh_CreateTag (mesh, "solution", SCUtil_DBL, neq, solution_field_tag);
  FMDB_Tag_SetAutoMigrOn (mesh, residual_field_tag, FMDB_VERTEX);
  FMDB_Tag_SetAutoMigrOn (mesh, solution_field_tag, FMDB_VERTEX);

  // Code to parse the vector of StateStructs and save the information

  // dim[0] is the number of cells in this workset
  // dim[1] is the number of QP per cell
  // dim[2] is the number of dimensions of the field
  // dim[3] is the number of dimensions of the field

  for (std::size_t i=0; i<sis->size(); i++) {
    Albany::StateStruct& st = *((*sis)[i]);
    std::vector<int>& dim = st.dim;

    if(st.aClass == Albany::StateStruct::Node) { // Data at the node points

       const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer 
               = sis->getNodalDataBlock()->getNodeContainer();

        (*nodeContainer)[st.name] = AlbPUMI::buildPUMINodeField(st.name, dim, st.output);

    }

    // qpscalars

    else if (dim.size() == 2){
      if(st.entity == Albany::StateStruct::QuadPoint) {

        qpscalar_states.push_back(Teuchos::rcp(new QPData<2>(st.name, dim)));

        std::cout << "NNNN qps field name " << st.name << " size : " << dim[1] << std::endl;
      }
/*
      else if(st.entity == Albany::StateStruct::NodePoint) {

        scalar_states.push_back(Teuchos::rcp(new NodeData<2>(st.name, dim)));

        std::cout << "NNNN Node scalar field name " << st.name << " size : " << dim[1] << std::endl;
      }
*/
    }

    // qpvectors

    else if (dim.size() == 3){
      if(st.entity == Albany::StateStruct::QuadPoint) {

        qpvector_states.push_back(Teuchos::rcp(new QPData<3>(st.name, dim)));

        std::cout << "NNNN qpv field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << std::endl;
      }
/*
      else if(st.entity == Albany::StateStruct::NodePoint) {

        vector_states.push_back(Teuchos::rcp(new NodeData<3>(st.name, dim)));

        std::cout << "NNNN Node vector field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << std::endl;
      }
*/
    }

    // qptensors

    else if (dim.size() == 4){
      if(st.entity == Albany::StateStruct::QuadPoint) {

        qptensor_states.push_back(Teuchos::rcp(new QPData<4>(st.name, dim)));

        std::cout << "NNNN qpt field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << " dim[3] : " << dim[3] << std::endl;
      }
/*
      else if(st.entity == Albany::StateStruct::NodePoint) {

        tensor_states.push_back(Teuchos::rcp(new NodeData<4>(st.name, dim)));

        std::cout << "NNNN Node tensor field name " << st.name << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << " dim[3] : " << dim[3] << std::endl;
      }
*/
    }

    // just a scalar number

    else if ( dim.size() == 1 && st.entity == Albany::StateStruct::ScalarValue) {
      // dim not used or accessed here
      scalarValue_states.push_back(Teuchos::rcp(new QPData<1>(st.name, dim)));
    }

    // anything else is an error!

    else TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "dim.size() < 2 || dim.size()>4 || " <<
         "st.entity != Albany::StateStruct::QuadPoint || " <<
         "st.entity != Albany::StateStruct::NodePoint" << std::endl);

  }

}


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
AlbPUMI::FMDBMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

Albany::AbstractMeshStruct::msType
AlbPUMI::FMDBMeshStruct::meshSpecsType()
{

  std::string str = outputFileName;
  size_t found = str.find("vtk");

  if(found != std::string::npos){

    return FMDB_VTK_MS;

  }

  found = str.find("exo");
  if(found != std::string::npos){

    return FMDB_EXODUS_MS;

  }

  TEUCHOS_TEST_FOR_EXCEPTION(true,
       std::logic_error,
       "Unrecognized output file extension given in the input file" << std::endl);

}

int AlbPUMI::FMDBMeshStruct::computeWorksetSize(const int worksetSizeMax,
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
AlbPUMI::FMDBMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_TEST_FOR_EXCEPT(step < 0 || step >= solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
//  stk::io::process_input_request(*mesh_data, *bulkData, index);
}

void AlbPUMI::FMDBMeshStruct::setupMeshBlkInfo()
{

   int nBlocks = this->meshSpecs.size();

   for(int i = 0; i < nBlocks; i++){

      const Albany::MeshSpecsStruct &ms = *meshSpecs[i];

      meshDynamicData[i] = Teuchos::rcp(new Albany::CellSpecs(ms.ctd, ms.worksetSize, ms.cubatureDegree,
                      numDim, neq, 0, useCompositeTet()));

   }

}




Teuchos::RCP<const Teuchos::ParameterList>
AlbPUMI::FMDBMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = rcp(new Teuchos::ParameterList("Valid FMDBParams"));

  validPL->set<std::string>("FMDB Solution Name", "",
      "Name of solution output vector written to output file");
  validPL->set<std::string>("FMDB Residual Name", "",
      "Name of residual output vector written to output file");
  validPL->set<int>("FMDB Write Interval", 3, "Step interval to write solution data to output file");
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

  validPL->set<std::string>("FMDB Input File Name", "", "File Name For FMDB Mesh Input");
  validPL->set<std::string>("FMDB Output File Name", "", "File Name For FMDB Mesh Output");

  validPL->set<std::string>("Acis Model Input File Name", "", "File Name For ACIS Model Input");
  validPL->set<std::string>("Parasolid Model Input File Name", "", "File Name For PARASOLID Model Input");
  validPL->set<std::string>("Mesh Model Input File Name", "", "File Name for meshModel Input");

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

  // Parameters to refine the mesh after input
  validPL->set<double>("Resize Input Mesh Element Size", 1.0, "Resize mesh element to this size at input");
  validPL->set<int>("Max Number of Resize Iterations", 0, "Max number of iteration sweeps to use during initial element resize");

  validPL->set<std::string>("LB Method", "", "Method used to load balance mesh (default \"ParMETIS\")");
  validPL->set<std::string>("LB Approach", "", "Approach used to load balance mesh (default \"PartKway\")");

  Teuchos::TwoDArray<std::string> defaultData;
  validPL->set<Teuchos::TwoDArray<std::string> >("Element Block Associations", defaultData,
      "Association between region ID and element block string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Node Set Associations", defaultData,
      "Association between face ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Side Set Associations", defaultData,
      "Association between face ID and side set string");

  return validPL;
}

