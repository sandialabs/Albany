//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TwoDArray.hpp"
#include <Shards_BasicTopologies.hpp>

#include <apfSTK.h>
#include <apfShape.h>
#include <ma.h>

class SizeFunction : public ma::IsotropicFunction {
  public:
    SizeFunction(double s) {size = s;}
    double getValue(ma::Entity*) {return size;}
  private:
    double size;
};

AlbPUMI::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  apfMesh(0)
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

  TEUCHOS_TEST_FOR_EXCEPTION(model==NULL,std::logic_error,"FMDBMeshStruct: no model" << std::endl);

  FMDB_Mesh_Create (model, mesh);

  int rc = FMDB_Mesh_LoadFromFile (mesh, &mesh_file[0], useDistributedMesh);
  TEUCHOS_TEST_FOR_EXCEPTION(rc,std::logic_error,
      "FAILED MESH LOADING - check mesh file or number of input files" << std::endl)

  FMDB_Mesh_DspSize(mesh);

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

    pumi::GRIter gr_iter = pumi::GM_regionIter(model);
    pGeomEnt geom_rgn;
    while ((geom_rgn = pumi::GRIter_next(gr_iter)) != NULL)
    {
      for(size_t eblock = 0; eblock < nEBAssoc; eblock++){
        if (GEN_tag(geom_rgn) == atoi(EBAssociations(0, eblock).c_str()))
          PUMI_Exodus_CreateElemBlk(geom_rgn, EBAssociations(1, eblock).c_str());
      }
    }
    pumi::GRIter_delete(gr_iter);
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
      *out << "Node set \"" << NSAssociations(1, ns).c_str() << "\" matches geometric face : "
        << NSAssociations(0, ns).c_str() << std::endl;
    }

    pumi::GFIter gf_iter=pumi::GM_faceIter(model);
    pGeomEnt geom_face;
    while ((geom_face=pumi::GFIter_next(gf_iter)) != NULL)
    {
      for(size_t ns = 0; ns < nNSAssoc; ns++){
        if (GEN_tag(geom_face) == atoi(NSAssociations(0, ns).c_str())){
          PUMI_Exodus_CreateNodeSet(geom_face, NSAssociations(1, ns).c_str());
        }
      }
    }
    pumi::GFIter_delete(gf_iter);

  }

  if(params->isParameter("Edge Node Set Associations")){ // User has specified associations in the input file

    // Get node set associations from input file
    Teuchos::TwoDArray< std::string > EdgeNSAssociations;

    EdgeNSAssociations = params->get<Teuchos::TwoDArray<std::string> >("Edge Node Set Associations");

    TEUCHOS_TEST_FOR_EXCEPTION( !(2 == EdgeNSAssociations.getNumRows()),
        Teuchos::Exceptions::InvalidParameter,
        "Error in specifying node set associations in input file" );

    int nEdgeNSAssoc = EdgeNSAssociations.getNumCols();

    for(size_t ns = 0; ns < nEdgeNSAssoc; ns++){
      *out << "Node set \"" << EdgeNSAssociations(1, ns).c_str() << "\" matches geometric edge : "
        << EdgeNSAssociations(0, ns).c_str() << std::endl;
    }

    pumi::GEIter ge_iter=pumi::GM_edgeIter(model);
    pGeomEnt geom_edge;
    while ((geom_edge=pumi::GEIter_next(ge_iter)) != NULL)
    {
      for(size_t ns = 0; ns < nEdgeNSAssoc; ns++){
        if (GEN_tag(geom_edge) == atoi(EdgeNSAssociations(0, ns).c_str())){
          PUMI_Exodus_CreateNodeSet(geom_edge, EdgeNSAssociations(1, ns).c_str());
        }
      }
    }
    pumi::GEIter_delete(ge_iter);
  }

  if(params->isParameter("Vertex Node Set Associations")){ // User has specified associations in the input file

    // Get node set associations from input file
    Teuchos::TwoDArray< std::string > VertexNSAssociations;

    VertexNSAssociations = params->get<Teuchos::TwoDArray<std::string> >("Vertex Node Set Associations");

    TEUCHOS_TEST_FOR_EXCEPTION( !(2 == VertexNSAssociations.getNumRows()),
        Teuchos::Exceptions::InvalidParameter,
        "Error in specifying node set associations in input file" );

    int nVertexNSAssoc = VertexNSAssociations.getNumCols();

    for(size_t ns = 0; ns < nVertexNSAssoc; ns++){
      *out << "Node set \"" << VertexNSAssociations(1, ns).c_str() << "\" matches geometric vertex : "
        << VertexNSAssociations(0, ns).c_str() << std::endl;
    }

    pumi::GVIter gv_iter=pumi::GM_vertexIter(model);
    pGeomEnt geom_vertex;
    while ((geom_vertex=pumi::GVIter_next(gv_iter)) != NULL)
    {
      for(size_t ns = 0; ns < nVertexNSAssoc; ns++){
        if (GEN_tag(geom_vertex) == atoi(VertexNSAssociations(0, ns).c_str())){
          PUMI_Exodus_CreateNodeSet(geom_vertex, VertexNSAssociations(1, ns).c_str());
        }
      }
    }
    pumi::GVIter_delete(gv_iter);
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


    pumi::GFIter gf_iter=pumi::GM_faceIter(model);
    pGeomEnt geom_face;
    while ((geom_face = pumi::GFIter_next(gf_iter)) != NULL)
    {
      for(size_t ss = 0; ss < nSSAssoc; ss++){
        if (GEN_tag(geom_face) == atoi(SSAssociations(0, ss).c_str()))
          PUMI_Exodus_CreateSideSet(geom_face, SSAssociations(1, ss).c_str());
      }
    }
    pumi::GFIter_delete(gf_iter);
  }

  apfMesh = apf::createMesh(mesh);
  bool isQuadMesh = params->get<bool>("2nd Order Mesh",false);
  if ((isQuadMesh) && (apfMesh->getShape() != apf::getLagrange(2)))
  {
    *out << "Converting straight-sided mesh to quadratic" << std::endl;
    changeMeshShape(apfMesh,apf::getLagrange(2));
  }

  // Resize mesh after input if indicated in the input file
  if(params->isParameter("Resize Input Mesh Element Size")){ // User has indicated a desired element size in input file
      SizeFunction sizeFunction(params->get<double>("Resize Input Mesh Element Size", 0.1));
      int num_iters = params->get<int>("Max Number of Mesh Adapt Iterations", 1);
      ma::Input* input = ma::configure(apfMesh,&sizeFunction);
      input->maximumIterations = num_iters;
      input->shouldSnap = false;
      ma::adapt(input);
      FMDB_Mesh_DspSize(mesh);
  }

  //get mesh dim
  FMDB_Mesh_GetDim(mesh, &numDim);

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

  cubatureDegree = params->get("Cubature Degree", 1);
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

  for(int ns = 0; ns < node_sets.size(); ns++)
  {
    std::string NS_name;
    PUMI_NodeSet_GetName(node_sets[ns], NS_name);
    nsNames.push_back(NS_name);
  }

  // Side sets
  std::vector<pSideSet> side_sets;
  PUMI_Exodus_GetSideSet(mesh, side_sets);

  int status;

  for(int ss = 0; ss < side_sets.size(); ss++)
  {
    std::string SS_name;
    PUMI_SideSet_GetName(side_sets[ss], SS_name);
    ssNames.push_back(SS_name);
  }

  // Construct MeshSpecsStruct
  std::vector<pMeshEnt> elements;
  const CellTopologyData* ctd = getCellTopology(apfMesh);
  if (!params->get("Separate Evaluators by Element Block",false))
  {
    // get elements in the first element block
    std::string EB_name;
    PUMI_ElemBlk_GetName(elem_blocks[0], EB_name);
    this->meshSpecs[0] = Teuchos::rcp(
        new Albany::MeshSpecsStruct(
          *ctd, numDim, cubatureDegree,
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
      std::string EB_name;
      PUMI_ElemBlk_GetName(elem_blocks[eb], EB_name);
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(*ctd, numDim, cubatureDegree,
                                              nsNames, ssNames, worksetSize, EB_name,
                                              this->ebNameToIndex, this->interleavedOrdering));
    } // for
  } // else

}

AlbPUMI::FMDBMeshStruct::~FMDBMeshStruct()
{
  apfMesh->destroyNative();
  apf::destroyMesh(apfMesh);
  ParUtil::Instance()->Finalize(0); // skip MPI_finalize
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

  using Albany::StateStruct;

  // Set the number of equation present per node. Needed by AlbPUMI_FMDBDiscretization.

  neq = neq_;

  Teuchos::Array<std::string> defaultLayout;
  solVectorLayout = 
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", defaultLayout);

  if (solVectorLayout.size() == 0) { 
    int valueType;
    if (neq==1)
      valueType = apf::SCALAR;
    else if (neq==3)
      valueType = apf::VECTOR;
    else
    {
      assert(neq==9);
      valueType = apf::MATRIX;
    }
    apf::createFieldOn(apfMesh,"residual",valueType);
    apf::createFieldOn(apfMesh,"solution",valueType);
  }
  else 
    splitFields(solVectorLayout);

  solutionInitialized = false;
  residualInitialized = false;

  // Code to parse the vector of StateStructs and save the information

  // dim[0] is the number of cells in this workset
  // dim[1] is the number of QP per cell
  // dim[2] is the number of dimensions of the field
  // dim[3] is the number of dimensions of the field

  std::set<std::string> nameSet;

  for (std::size_t i=0; i<sis->size(); i++) {
    StateStruct& st = *((*sis)[i]);

    if ( ! nameSet.insert(st.name).second)
      continue; //ignore duplicates

    std::vector<int>& dim = st.dim;

    if(st.entity == StateStruct::NodalData) { // Data at the node points

       const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer
               = sis->getNodalDataBlock()->getNodeContainer();

        (*nodeContainer)[st.name] = AlbPUMI::buildPUMINodeField(st.name, dim, st.output);

    }

    // qpscalars

    else if (dim.size() == 2){
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode) {

        qpscalar_states.push_back(Teuchos::rcp(new QPData<double, 2>(st.name, dim, st.output)));

        if ( ! PCU_Comm_Self())
          std::cout << "AlbPUMI::FMDBMeshStruct qps field name " << st.name
            << " size : " << dim[1] << std::endl;
      }
    }

    // qpvectors

    else if (dim.size() == 3){
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode) {

        qpvector_states.push_back(Teuchos::rcp(new QPData<double, 3>(st.name, dim, st.output)));

        if ( ! PCU_Comm_Self())
          std::cout << "AlbPUMI::FMDBMeshStruct qpv field name " << st.name
            << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << std::endl;
      }
    }

    // qptensors

    else if (dim.size() == 4){
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode) {

        qptensor_states.push_back(Teuchos::rcp(new QPData<double, 4>(st.name, dim, st.output)));

        if ( ! PCU_Comm_Self())
          std::cout << "AlbPUMI::FMDBMeshStruct qpt field name " << st.name
            << " dim[1] : " << dim[1] << " dim[2] : " << dim[2] << " dim[3] : " << dim[3] << std::endl;
      }
    }

    // just a scalar number

    else if ( dim.size() == 1 && st.entity == Albany::StateStruct::WorksetValue) {
      // dim not used or accessed here
      scalarValue_states.push_back(Teuchos::rcp(new QPData<double, 1>(st.name, dim, st.output)));
    }

    // anything else is an error!

    else TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "dim.size() < 2 || dim.size()>4 || " <<
         "st.entity != Albany::StateStruct::QuadPoint || " <<
         "st.entity != Albany::StateStruct::ElemNode || " <<
         "st.entity != Albany::StateStruct::NodalData" << std::endl);

  }

}

void
AlbPUMI::FMDBMeshStruct::splitFields(Teuchos::Array<std::string> fieldLayout)
{ // user is breaking up or renaming solution & residual fields

  TEUCHOS_TEST_FOR_EXCEPTION((fieldLayout.size() % 2), std::logic_error,
      "Error in input file: specification of solution vector layout is incorrect\n");

  int valueType;

  for (std::size_t i=0; i < fieldLayout.size(); i+=2) {

    if (fieldLayout[i+1] == "S")
      valueType = apf::SCALAR;
    else if (fieldLayout[i+1] == "V")
      valueType = apf::VECTOR;
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Error in input file: specification of solution vector layout is incorrect\n");

    apf::createFieldOn(apfMesh,fieldLayout[i].c_str(),valueType);
    apf::createFieldOn(apfMesh,fieldLayout[i].append("Res").c_str(),valueType);
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
  validPL->set<int>("Cubature Degree", 1, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields,
      "Fields to pick up from the restart file when restarting");
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layouts of solution vector components");
  validPL->set<bool>("2nd Order Mesh", false, "Flag to indicate 2nd order Lagrange shape functions");
  
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
      "Association between geometric face ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Edge Node Set Associations", defaultData,
      "Association between geometric edge ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Vertex Node Set Associations", defaultData,
      "Association between geometric edge ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Side Set Associations", defaultData,
      "Association between face ID and side set string");

  return validPL;
}

