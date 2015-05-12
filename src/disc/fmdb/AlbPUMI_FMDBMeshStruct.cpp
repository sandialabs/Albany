//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>

#include <boost/mpi/exception.hpp>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TwoDArray.hpp"
#include <Shards_BasicTopologies.hpp>

#include <gmi_mesh.h>
#ifdef SCOREC_SIMMODEL
#include <gmi_sim.h>
#include <SimUtil.h>
#endif
#include <apfShape.h>
#include <ma.h>
#include <PCU.h>

class SizeFunction : public ma::IsotropicFunction {
  public:
    SizeFunction(double s) {size = s;}
    double getValue(ma::Entity*) {return size;}
  private:
    double size;
};

static void loadSets(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    apf::StkModels& sets,
    const char* param_name,
    int geom_dim,
    int mesh_dim)
{
  // User has specified associations in the input file
  if(params->isParameter(param_name)) {
    // Get element block associations from input file
    Teuchos::TwoDArray< std::string > pairs;
    pairs = params->get<Teuchos::TwoDArray<std::string> >(param_name);
    int npairs = pairs.getNumCols();
    size_t start = sets[mesh_dim].getSize();
    sets[mesh_dim].setSize(start + npairs);
    for(int i = 0; i < npairs; ++i) {
      apf::StkModel& set = sets[mesh_dim][start + i];
      set.dim = geom_dim;
      set.apfTag = atoi(pairs(0, i).c_str());
      set.stkName = pairs(1, i);
    }
  }
}

AlbPUMI::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<const Teuchos_Comm>& commT) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  PCU_Comm_Init();
  params->validateParameters(*getValidDiscretizationParameters(),0);

  std::string mesh_file = params->get<std::string>("FMDB Input File Name");
  outputFileName = params->get<std::string>("FMDB Output File Name", "");
  outputInterval = params->get<int>("FMDB Write Interval", 1); // write every time step default
  useNullspaceTranslationOnly = params->get<bool>("Use Nullspace Translation Only", false);

  compositeTet = false;

  gmi_register_mesh();

  std::string model_file;
  if(params->isParameter("Mesh Model Input File Name"))
    model_file = params->get<std::string>("Mesh Model Input File Name");

#ifdef SCOREC_SIMMODEL
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();

  if (params->isParameter("Acis Model Input File Name"))
    model_file = params->get<std::string>("Parasolid Model Input File Name");

  if(params->isParameter("Parasolid Model Input File Name"))
    model_file = params->get<std::string>("Parasolid Model Input File Name");
#endif

  mesh = apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str());
  model = mesh->getModel();

  mesh->verify();

  int d = mesh->getDimension();
  loadSets(params, sets, "Element Block Associations",   d,     d);
  loadSets(params, sets, "Node Set Associations",        d - 1, 0);
  loadSets(params, sets, "Edge Node Set Associations",   1,     0);
  loadSets(params, sets, "Vertex Node Set Associations", 0,     0);
  loadSets(params, sets, "Side Set Associations",        d - 1, d - 1);

  bool isQuadMesh = params->get<bool>("2nd Order Mesh",false);
  if (isQuadMesh)
    assert(mesh->getShape() == apf::getLagrange(2));

  // Resize mesh after input if indicated in the input file
  // User has indicated a desired element size in input file
  if(params->isParameter("Resize Input Mesh Element Size")){
      SizeFunction sizeFunction(params->get<double>(
            "Resize Input Mesh Element Size", 0.1));
      int num_iters = params->get<int>(
          "Max Number of Mesh Adapt Iterations", 1);
      ma::Input* input = ma::configure(mesh,&sizeFunction);
      input->maximumIterations = num_iters;
      input->shouldSnap = false;
      ma::adapt(input);
  }

  numDim = mesh->getDimension();

  // Build a map to get the EB name given the index

  int numEB = sets[d].getSize(), EB_size;
  std::vector<int> el_blocks;

  for (int eb=0; eb < numEB; eb++){
    apf::StkModel& set = sets[d][eb];
    std::string EB_name = set.stkName;
    apf::ModelEntity* me = mesh->findModelEntity(set.dim, set.apfTag);
    this->ebNameToIndex[EB_name] = eb;
    EB_size = apf::countEntitiesOn(mesh, me, numDim);
    el_blocks.push_back(EB_size);
  }

  // Set defaults for cubature and workset size, overridden in input file

  cubatureDegree = params->get("Cubature Degree", 3);
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
  for(size_t ns = 0; ns < sets[0].getSize(); ns++) {
    nsNames.push_back(sets[0][ns].stkName);
  }

  // Side sets
  for(size_t ss = 0; ss < sets[d - 1].getSize(); ss++) {
    ssNames.push_back(sets[d - 1][ss].stkName);
  }

  // Construct MeshSpecsStruct
  const CellTopologyData* ctd = apf::getCellTopology(mesh);
  if (!params->get("Separate Evaluators by Element Block",false))
  {
    // get elements in the first element block
    std::string EB_name = sets[d][0].stkName;
    this->meshSpecs[0] = Teuchos::rcp(
        new Albany::MeshSpecsStruct(
          *ctd, numDim, cubatureDegree,
          nsNames, ssNames, worksetSize, EB_name,
          this->ebNameToIndex, this->interleavedOrdering));
  }
  else
  {
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    int eb_size;
    std::string eb_name;
    for (int eb=0; eb<numEB; eb++)
    {
      std::string EB_name = sets[d][eb].stkName;
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          *ctd, numDim, cubatureDegree, nsNames, ssNames, worksetSize, EB_name,
          this->ebNameToIndex, this->interleavedOrdering, true));
    } // for
  } // else

}

AlbPUMI::FMDBMeshStruct::~FMDBMeshStruct()
{
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  PCU_Comm_Free();
#ifdef SCOREC_SIMMODEL
  gmi_sim_stop();
  Sim_unregisterAllKeys();
#endif
}

void
AlbPUMI::FMDBMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize_,
                  const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& /*side_set_sis*/)
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
    apf::createFieldOn(mesh,"residual",valueType);
    apf::createFieldOn(mesh,"solution",valueType);
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
    std::vector<PHX::DataLayout::size_type>& dim = st.dim;
    if(st.entity == StateStruct::NodalData) { // Data at the node points
       const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer
               = sis->getNodalDataBase()->getNodeContainer();
        (*nodeContainer)[st.name] = AlbPUMI::buildPUMINodeField(st.name, dim, st.output);
    }
    else if (dim.size() == 2) {
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qpscalar_states.push_back(Teuchos::rcp(new QPData<double, 2>(st.name, dim, st.output)));
    }
    else if (dim.size() == 3) {
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qpvector_states.push_back(Teuchos::rcp(new QPData<double, 3>(st.name, dim, st.output)));
    }
    else if (dim.size() == 4){
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qptensor_states.push_back(Teuchos::rcp(new QPData<double, 4>(st.name, dim, st.output)));
    }
    else if ( dim.size() == 1 && st.entity == Albany::StateStruct::WorksetValue)
      scalarValue_states.push_back(Teuchos::rcp(new QPData<double, 1>(st.name, dim, st.output)));
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

    apf::createFieldOn(mesh,fieldLayout[i].c_str(),valueType);
    apf::createFieldOn(mesh,fieldLayout[i].append("Res").c_str(),valueType);
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
  if(outputFileName.find("exo") != std::string::npos)
    return FMDB_EXODUS_MS;
  else
    return FMDB_VTK_MS;
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

  validPL->set<int>("FMDB Write Interval", 3, "Step interval to write solution data to output file");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layouts of solution vector components");
  validPL->set<bool>("2nd Order Mesh", false, "Flag to indicate 2nd order Lagrange shape functions");

  validPL->set<std::string>("FMDB Input File Name", "", "File Name For FMDB Mesh Input");
  validPL->set<std::string>("FMDB Output File Name", "", "File Name For FMDB Mesh Output");

  validPL->set<std::string>("Acis Model Input File Name", "", "File Name For ACIS Model Input");
  validPL->set<std::string>("Parasolid Model Input File Name", "", "File Name For PARASOLID Model Input");
  validPL->set<std::string>("Mesh Model Input File Name", "", "File Name for meshModel Input");

  validPL->set<double>("Imbalance tolerance", 1.03, "Imbalance tolerance");

  // Parameters to refine the mesh after input
  validPL->set<double>("Resize Input Mesh Element Size", 1.0, "Resize mesh element to this size at input");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1);

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

  validPL->set<bool>("Use Nullspace Translation Only", false,
                     "Temporary hack to get MueLu (possibly) working for us");

  return validPL;
}

