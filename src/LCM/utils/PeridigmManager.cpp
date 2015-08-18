/*! \file PeridigmManager.cpp */

#include "PeridigmManager.hpp"
#include "Peridigm_ProximitySearch.hpp"
#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include "Phalanx_DataLayout.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "PHAL_Dimension.hpp"
#include <boost/math/special_functions/fpclassify.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Epetra_Export.h>

const Teuchos::RCP<LCM::PeridigmManager>& LCM::PeridigmManager::self() {
  static Teuchos::RCP<PeridigmManager> peridigmManager;
  return peridigmManager;
}

void LCM::PeridigmManager::initializeSingleton(
  const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  if ( ! params->sublist("Problem").isSublist("Peridigm Parameters"))
    return;
  Teuchos::RCP<LCM::PeridigmManager>* ston =
    const_cast<Teuchos::RCP<LCM::PeridigmManager>*>(&self());
  *ston = Teuchos::rcp(new PeridigmManager());
}

LCM::PeridigmManager::PeridigmManager() : hasPeridynamics(false), enableOptimizationBasedCoupling(false), previousTime(0.0), currentTime(0.0), timeStep(0.0), cubatureDegree(-1)
{}

void LCM::PeridigmManager::initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
  Teuchos::RCP<Albany::AbstractDiscretization> disc,
  const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  if(!params->sublist("Problem").isSublist("Peridigm Parameters")){
    hasPeridynamics = false;
    return;
  }

  teuchosComm = comm;
  peridigmParams = Teuchos::RCP<Teuchos::ParameterList>(new Teuchos::ParameterList(params->sublist("Problem").sublist("Peridigm Parameters", true)));
  Teuchos::ParameterList& problemParams = params->sublist("Problem");
  Teuchos::ParameterList& discretizationParams = params->sublist("Discretization");
  cubatureDegree = discretizationParams.get<int>("Cubature Degree", 2);

  if(peridigmParams->isSublist("Optimization Based Coupling")){
    enableOptimizationBasedCoupling = true;
  }

  // Read the material data base file, if any
  Teuchos::RCP<QCAD::MaterialDatabase> materialDataBase;
  if(problemParams.isType<std::string>("MaterialDB Filename")){
    std::string filename = problemParams.get<std::string>("MaterialDB Filename");
    materialDataBase = Teuchos::rcp(new QCAD::MaterialDatabase(filename, teuchosComm));
  }

  stkDisc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(stkDisc.is_null(), "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for STK meshes.\n\n");
  metaData = Teuchos::rcpFromRef(stkDisc->getSTKMetaData());
  bulkData = Teuchos::rcpFromRef(stkDisc->getSTKBulkData());
  TEUCHOS_TEST_FOR_EXCEPT_MSG(metaData->spatial_dimension() != 3, "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for three-dimensional meshes.\n\n");

  // Store the cell topology for each element mesh part
  std::map<std::string,CellTopologyData> partCellTopologyData; 

  const stk::mesh::PartVector& stkParts = metaData->get_parts();
  stk::mesh::PartVector stkElementBlocks;
  for(stk::mesh::PartVector::const_iterator it = stkParts.begin(); it != stkParts.end(); ++it){
    stk::mesh::Part* const part = *it;
    if(!stk::mesh::is_auto_declared_part(*part) && part->primary_entity_rank() == stk::topology::ELEMENT_RANK){
      stkElementBlocks.push_back(part);
      partCellTopologyData[part->name()] = *metaData->get_cell_topology(*part).getCellTopologyData();
    }
  }

  //   const stk::mesh::FieldVector &fields = metaData->get_fields();
  //   for(unsigned int i=0 ; i<fields.size() ; ++i)
  //     std::cout << "DJL DEBUGGING STK field " << *fields[i] << std::endl;

  stk::mesh::Field<double,stk::mesh::Cartesian3d>* coordinatesField = 
      metaData->get_field< stk::mesh::Field<double,stk::mesh::Cartesian3d>>(stk::topology::NODE_RANK, "coordinates");
  TEUCHOS_TEST_FOR_EXCEPT_MSG(coordinatesField == 0, "\n\n**** Error in PeridigmManager::initialize(), unable to access coordinates field.\n\n");

  stk::mesh::Field<double,stk::mesh::Cartesian3d>* volumeField = 
      metaData->get_field< stk::mesh::Field<double,stk::mesh::Cartesian3d>>(stk::topology::ELEMENT_RANK, "volume");

  // Create a selector to select everything in the universal part that is either locally owned or globally shared
  stk::mesh::Selector selector = 
      stk::mesh::Selector( metaData->universal_part() ) & ( stk::mesh::Selector( metaData->locally_owned_part() ) | stk::mesh::Selector( metaData->globally_shared_part() ) );

  // Select element mesh entities that match the selector
  std::vector<stk::mesh::Entity> elements;
  stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), elements);

  // List of blocks for peridynamics, partial stress, and standard FEM
  std::vector<std::string> peridynamicsBlocks, peridynamicPartialStressBlocks, classicalContinuumMechanicsBlocks;

  int numPartialStressIds(0);

  // Bookkeeping so that partial stress nodes on the Peridigm side are guaranteed to have ids that don't exist in the Albany discretization
  int maxAlbanyElementId(0), maxAlbanyNodeId(0);

  // Create a list of all node GIDs in the Albany disc. for partial stress blocks
  // This will be used to determine neighor PD particles for each node
  // as needed in boundary conditions, etc.
  std::vector<int> albanyPartialStressNodeGIDs;
  std::map<int,int> albanyPartialStressNodeGlobalToLocalID;

  // Store the global node id for each sphere element that will be used for "Peridynamics" materials
  // Store necessary information for each Gauss point in a solid element for "Peridynamic Partial Stress" materials
  for(unsigned int iBlock=0 ; iBlock<stkElementBlocks.size() ; iBlock++){

    // Determine the block id under the assumption that the block names follow the format "block_1", "block_2", etc.
    // Older versions of stk did not have the ability to return the block id directly, I think newer versions of stk can do this however
    const std::string blockName = stkElementBlocks[iBlock]->name();
    size_t loc = blockName.find_last_of('_');
    TEUCHOS_TEST_FOR_EXCEPT_MSG(loc == string::npos, "\n**** Parse error in PeridigmManager::initialize(), invalid block name: " + blockName + "\n");
    stringstream blockIDSS(blockName.substr(loc+1, blockName.size()));
    int bId;
    blockIDSS >> bId;
    blockNameToBlockId[blockName] = bId;

    // Create a selector for all locally-owned elements in the block
    stk::mesh::Selector selector = 
        stk::mesh::Selector( *stkElementBlocks[iBlock] ) & stk::mesh::Selector( metaData->locally_owned_part() );

    // Select the mesh entities that match the selector
    std::vector<stk::mesh::Entity> elementsInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), elementsInElementBlock);
    std::vector<stk::mesh::Entity> nodesInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::NODE_RANK), nodesInElementBlock);
    // Determine the material model assigned to this block
    std::string materialModelName;
    if(!materialDataBase.is_null())
      materialModelName = materialDataBase->getElementBlockSublist(blockName, "Material Model").get<std::string>("Model Name");

    // Sphere elements with the "Peridynamics" material model
    if(materialModelName == "Peridynamics"){
      peridynamicsBlocks.push_back(blockName);
      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	int numNodes = bulkData->num_nodes(elementsInElementBlock[iElement]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(numNodes != 1,
				    "\n\n**** Error in PeridigmManager::initialize(), \"Peridynamics\" material model may be assigned only to sphere elements.  Multiple nodes per element detected..\n\n");
	const stk::mesh::Entity* nodes = bulkData->begin_nodes(elementsInElementBlock[iElement]);
	int globalId = bulkData->identifier(nodes[0]) - 1;
	int localId = static_cast<int>(peridigmNodeGlobalIds.size());
	peridigmNodeGlobalIds.push_back(globalId);
	peridigmGlobalIdToPeridigmLocalId[globalId] = localId;
      }
    }
    // Standard solid elements with the "Peridynamic Partial Stress" material model
    else if(materialModelName == "Peridynamic Partial Stress"){
      for(unsigned int iNode=0; iNode<nodesInElementBlock.size(); iNode++){
        const int nodeLocalID = albanyPartialStressNodeGIDs.size();
        const int nodeGlobalID = bulkData->identifier(nodesInElementBlock[iNode])-1;
        albanyPartialStressNodeGIDs.push_back(nodeGlobalID);
        albanyPartialStressNodeGlobalToLocalID[nodeGlobalID] = nodeLocalID;
      }
      peridynamicPartialStressBlocks.push_back(blockName);
      CellTopologyData& cellTopologyData = partCellTopologyData[blockName];
      shards::CellTopology cellTopology(&cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType>> cubature = cubFactory.create(cellTopology, cubatureDegree);
      const int numQPts = cubature->getNumPoints();
      numPartialStressIds += numQPts * elementsInElementBlock.size();
    }
    // Standard solid elements with a classical continum mechanics model
    else{
      classicalContinuumMechanicsBlocks.push_back(blockName);
    }

    // Track the max element and node id in the Albany discretization
    for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
      int elementId = bulkData->identifier(elementsInElementBlock[iElement]) - 1;
      if(elementId > maxAlbanyElementId)
        maxAlbanyElementId = elementId;
      int numNodes = bulkData->num_nodes(elementsInElementBlock[iElement]);
      const stk::mesh::Entity* nodes = bulkData->begin_nodes(elementsInElementBlock[iElement]);
      for(int i=0 ; i<numNodes ; ++i){
        int nodeId = bulkData->identifier(nodes[i]) - 1;
        if(nodeId > maxAlbanyNodeId)
          maxAlbanyNodeId = nodeId;
      }
    }
  }

  // Determine the Peridigm node ids for the Gauss points in the partial stress elements

  int numProc = teuchosComm->getSize();
  int pid = teuchosComm->getRank();

  // Find the minimum global id across all processors
  int lowestPossiblePartialStressId = maxAlbanyElementId + 1;
  if(maxAlbanyNodeId > lowestPossiblePartialStressId)
    lowestPossiblePartialStressId = maxAlbanyNodeId + 1;
  vector<int> localVal(1), globalVal(1);
  localVal[0] = lowestPossiblePartialStressId;
  Teuchos::reduceAll(*teuchosComm, Teuchos::REDUCE_MAX, 1, &localVal[0], &globalVal[0]);
  lowestPossiblePartialStressId = globalVal[0];

  int minPeridigmPartialStressId = lowestPossiblePartialStressId;

  for(int iProc=0 ; iProc<numProc ; iProc++){

    // Let all processors know how many partial stress nodes are on processor iProc
    localVal[0] = 0;
    if(pid == iProc)
      localVal[0] = numPartialStressIds;
    Teuchos::reduceAll(*teuchosComm, Teuchos::REDUCE_MAX, 1, &localVal[0], &globalVal[0]);

    // Adjust the min partial stress id such that processors will not end up with the same global ids
    if(pid > iProc)
      minPeridigmPartialStressId += globalVal[0];
  }

  std::vector<int> peridigmPartialStressLocalIds;
  for(int i=0 ; i<numPartialStressIds ; i++){
    int peridigmGlobalId = minPeridigmPartialStressId + i;
    int localId = static_cast<int>(peridigmNodeGlobalIds.size());
    peridigmNodeGlobalIds.push_back(peridigmGlobalId);
    peridigmPartialStressLocalIds.push_back(localId);
    peridigmGlobalIdToPeridigmLocalId[peridigmGlobalId] = localId;
  }

  // Write block information to stdout
  std::cout << "\n---- PeridigmManager ----";
  std::cout << "\n  peridynamics blocks:";
  for(unsigned int i=0 ; i<peridynamicsBlocks.size() ; ++i)
    std::cout << " " << peridynamicsBlocks[i];
  std::cout << "\n  peridynamic partial stress blocks:";
  for(unsigned int i=0 ; i<peridynamicPartialStressBlocks.size() ; ++i)
    std::cout << " " << peridynamicPartialStressBlocks[i];
  std::cout << "\n  classical continuum mechanics blocks:";
  for(unsigned int i=0 ; i<classicalContinuumMechanicsBlocks.size() ; ++i)
    std::cout << " " << classicalContinuumMechanicsBlocks[i];
  std::cout << "\n  max Albany element id: " << maxAlbanyElementId << std::endl;
  std::cout << "  max Albany node id: " << maxAlbanyNodeId << std::endl;
  std::cout << "  min Peridigm partial stress id: " << minPeridigmPartialStressId << std::endl;
  std::cout << "  number of Peridigm partial stress material points: " << numPartialStressIds << std::endl;
  if(enableOptimizationBasedCoupling){
    std::cout << "  enable optimization-based coupling: true\n" << std::endl;
  }
  else{
    std::cout << "  enable optimization-based coupling: false\n" << std::endl;
  }

  // Bail if there are no sphere elements or partial stress elements
  if(peridynamicsBlocks.size() == 0 && peridynamicPartialStressBlocks.size() == 0){
    hasPeridynamics = false;
    return;
  }
  else{
    hasPeridynamics = true;
  }

  std::vector<double> initialX(3*peridigmNodeGlobalIds.size());
  std::vector<double> cellVolume(peridigmNodeGlobalIds.size());
  std::vector<int> blockId(peridigmNodeGlobalIds.size());
  std::vector<double> initialNodeX(3*albanyPartialStressNodeGIDs.size());
  std::vector<int> nodeBlockId(albanyPartialStressNodeGIDs.size());

  // loop over the element blocks and store the initial positions, volume, and block_id
  int peridigmPartialStressIndex = 0;
  for(unsigned int iBlock=0 ; iBlock<stkElementBlocks.size() ; iBlock++){

    const std::string blockName = stkElementBlocks[iBlock]->name();
    int bId = blockNameToBlockId[blockName];

    // Create a selector for all locally-owned elements in the block
    stk::mesh::Selector selector = 
        stk::mesh::Selector( *stkElementBlocks[iBlock] ) & stk::mesh::Selector( metaData->locally_owned_part() );

    // Select the mesh entities that match the selector
    std::vector<stk::mesh::Entity> elementsInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), elementsInElementBlock);
    std::vector<stk::mesh::Entity> nodesInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::NODE_RANK), nodesInElementBlock);

    // Determine the material model assigned to this block
    std::string materialModelName;
    if(!materialDataBase.is_null())
      materialModelName = materialDataBase->getElementBlockSublist(blockName, "Material Model").get<std::string>("Model Name");

    if(materialModelName == "Peridynamics"){
      TEUCHOS_TEST_FOR_EXCEPT_MSG(volumeField == 0, "\n\n**** Error in PeridigmManager::initialize(), unable to access volume field.\n\n");
      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	int numNodes = bulkData->num_nodes(elementsInElementBlock[iElement]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(numNodes != 1, "\n\n**** Error in PeridigmManager::initialize(), \"Peridynamics\" material model may be assigned only to sphere elements.\n\n");
	const stk::mesh::Entity* node = bulkData->begin_nodes(elementsInElementBlock[iElement]);
	int globalId = bulkData->identifier(node[0]) - 1;
	int localId = peridigmGlobalIdToPeridigmLocalId[globalId];
	TEUCHOS_TEST_FOR_EXCEPT_MSG(localId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	blockId[localId] = bId;
	double* exodusVolume = stk::mesh::field_data(*volumeField, elementsInElementBlock[iElement]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(exodusVolume == 0, "\n\n**** Error in PeridigmManager::initialize(), failed to access element's volume field.\n\n");
	cellVolume[localId] = exodusVolume[0];
	double* exodusCoordinates = stk::mesh::field_data(*coordinatesField, node[0]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(exodusCoordinates == 0, "\n\n**** Error in PeridigmManager::initialize(), failed to access element's coordinates field.\n\n");
	initialX[localId*3]   = exodusCoordinates[0];
	initialX[localId*3+1] = exodusCoordinates[1];
	initialX[localId*3+2] = exodusCoordinates[2];
	sphereElementGlobalNodeIds.push_back(globalId);
      }
    }

    else if(materialModelName == "Peridynamic Partial Stress"){

      for(unsigned int iNode=0; iNode<nodesInElementBlock.size(); iNode++){
        const int nodeGlobalID = bulkData->identifier(nodesInElementBlock[iNode])-1;
        const int nodeLocalID = albanyPartialStressNodeGlobalToLocalID[nodeGlobalID];
        double* exodusCoordinates = stk::mesh::field_data(*coordinatesField, nodesInElementBlock[iNode]);
        initialNodeX[nodeLocalID*3]   = exodusCoordinates[0];
        initialNodeX[nodeLocalID*3+1] = exodusCoordinates[1];
        initialNodeX[nodeLocalID*3+2] = exodusCoordinates[2];
        nodeBlockId[nodeLocalID] = bId;
      }

      CellTopologyData& cellTopologyData = partCellTopologyData[blockName];
      shards::CellTopology cellTopology(&cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType>> cubature = cubFactory.create(cellTopology, cubatureDegree);
      const int numDim = cubature->getDimension();
      const int numQuadPoints = cubature->getNumPoints();
      const int numNodes = cellTopology.getNodeCount();
      const int numCells = 1;

      // Get the quadrature points and weights
      Intrepid::FieldContainer<RealType> quadratureRefPoints;
      Intrepid::FieldContainer<RealType> quadratureRefWeights;
      quadratureRefPoints.resize(numQuadPoints, numDim);
      quadratureRefWeights.resize(numQuadPoints);
      cubature->getCubature(quadratureRefPoints, quadratureRefWeights);

      // Container for the Jacobians, Jacobian determinants, and weighted measures
      Intrepid::FieldContainer<RealType> jacobians;
      Intrepid::FieldContainer<RealType> jacobianDeterminants;
      Intrepid::FieldContainer<RealType> weightedMeasures;
      jacobians.resize(numCells, numQuadPoints, numDim, numDim);
      jacobianDeterminants.resize(numCells, numQuadPoints);
      weightedMeasures.resize(numCells, numQuadPoints);

      // Create data structures for passing information to/from Intrepid.

      typedef PHX::KokkosViewFactory<RealType, PHX::Device> ViewFactory;

      // Physical points, which are the physical (x, y, z) values of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim>> physPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> physPoints("Physical Points", physPointsLayout);
      physPoints.setFieldData(ViewFactory::buildView(physPoints.fieldTag()));

      // Reference points, which are the natural coordinates of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim>> refPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> refPoints("Reference Points", refPointsLayout);
      refPoints.setFieldData(ViewFactory::buildView(refPoints.fieldTag()));

      // Cell workset, which is the set of nodes for the given element
      Teuchos::RCP< PHX::MDALayout<Cell, Node, Dim>> cellWorksetLayout = Teuchos::rcp(new PHX::MDALayout<Cell, Node, Dim>(numCells, numNodes, numDim));
      PHX::MDField<RealType, Cell, Node, Dim> cellWorkset("Cell Workset", cellWorksetLayout);
      cellWorkset.setFieldData(ViewFactory::buildView(cellWorkset.fieldTag()));

      // Copy the reference points from the Intrepid::FieldContainer to a PHX::MDField
      for(int qp=0 ; qp<numQuadPoints ; ++qp){
        for(int dof=0 ; dof<3 ; ++dof){
          refPoints(0, qp, dof) = quadratureRefPoints(qp, dof);
        }
      }

      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	int numNodesInElement = bulkData->num_nodes(elementsInElementBlock[iElement]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(numNodesInElement != numNodes, "\n\n**** Error in PeridigmManager::initialize(), bulkData->num_nodes() != numNodes.\n\n");
	const stk::mesh::Entity* node = bulkData->begin_nodes(elementsInElementBlock[iElement]);
	for(int i=0 ; i<numNodes ; i++){
	  double* coordinates = stk::mesh::field_data(*coordinatesField, node[i]);
	  for(int dof=0 ; dof<3 ; ++dof)
	    cellWorkset(0, i, dof) = coordinates[dof];
	}

	// Determine the global (x,y,z) coordinates of the quadrature points
  	Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, cellTopology);

	// Determine the weighted integration measures, which are the volumes that will be assigned to the peridynamic material points
 	Intrepid::CellTools<RealType>::setJacobian(jacobians, refPoints, cellWorkset, cellTopology);
 	Intrepid::CellTools<RealType>::setJacobianDet(jacobianDeterminants, jacobians);
 	Intrepid::FunctionSpaceTools::computeCellMeasure<RealType>(weightedMeasures, jacobianDeterminants, quadratureRefWeights);

	// Bookkeeping for use downstream
	PartialStressElement partialStressElement;
	partialStressElement.albanyElement = elementsInElementBlock[iElement];
	partialStressElement.cellTopologyData = cellTopologyData;

	for(unsigned int i=0 ; i<numNodes ; ++i){
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 0) );
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 1) );
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 2) );
	}

	for(unsigned int qp=0 ; qp<numQuadPoints ; ++qp){
	  int localId = peridigmPartialStressLocalIds[peridigmPartialStressIndex++];
	  int globalId = peridigmNodeGlobalIds[localId];
	  blockId[localId] = bId;
	  cellVolume[localId] = weightedMeasures(0, qp);
	  initialX[localId*3]   = physPoints(0, qp, 0);
	  initialX[localId*3+1] = physPoints(0, qp, 1);
	  initialX[localId*3+2] = physPoints(0, qp, 2);
	  partialStressElement.peridigmGlobalIds.push_back(globalId);
	}

	partialStressElements.push_back(partialStressElement);
      }
    }
  }

  // Create a vector for storing the previous solution (from last converged load step)
  previousSolutionPositions = std::vector<double>(initialX.size());
  for(unsigned int i=0 ; i<initialX.size() ; ++i)
    previousSolutionPositions[i] = initialX[i];

  // Create a Peridigm discretization
  const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* >(teuchosComm.get());
  TEUCHOS_TEST_FOR_EXCEPT_MSG(mpiComm == 0, "\n\n**** Error in PeridigmManager::initialize(), failed to dynamically cast comm object to Teuchos::MpiComm<int>.\n");
  peridynamicDiscretization = Teuchos::rcp<PeridigmNS::Discretization>(new PeridigmNS::AlbanyDiscretization(*mpiComm->getRawMpiComm(),
    peridigmParams,
    static_cast<int>(peridigmNodeGlobalIds.size()),
    &peridigmNodeGlobalIds[0],
    &initialX[0],
    &cellVolume[0],
    &blockId[0],
    static_cast<int>(albanyPartialStressNodeGIDs.size()),
    &albanyPartialStressNodeGIDs[0],
    &initialNodeX[0],
    &nodeBlockId[0]));

  // Create a Peridigm object
  peridigm = Teuchos::rcp<PeridigmNS::Peridigm>(new PeridigmNS::Peridigm(*mpiComm->getRawMpiComm(),
    peridigmParams,
    peridynamicDiscretization));

  // Create data structure for obtaining the global element id given the workset index and workset local element id.
  Albany::WsLIDList wsLIDList = stkDisc->getElemGIDws();
  for(Albany::WsLIDList::iterator it=wsLIDList.begin() ; it!=wsLIDList.end() ; ++it){
    int globalElementId = it->first;
    int worksetIndex = it->second.ws;
    int worksetLocalId = it->second.LID;
    std::vector<int>& wsGIDs = worksetLocalIdToGlobalId[worksetIndex];
    TEUCHOS_TEST_FOR_EXCEPT_MSG(worksetLocalId != wsGIDs.size(), "\n\n**** Error in PeridigmManager::initialize(), unexpected workset local id.\n\n");
    wsGIDs.push_back(globalElementId);
  }

  // Create a data structure for obtaining the Peridigm global ids given the global id of a Albany partial stress element
  for(unsigned int i=0 ; i<partialStressElements.size() ; ++i){
    int albanyGlobalElementId = bulkData->identifier(partialStressElements[i].albanyElement) - 1;
    vector<int>& peridigmGlobalIds = partialStressElements[i].peridigmGlobalIds;
    albanyPartialStressElementGlobalIdToPeridigmGlobalIds[albanyGlobalElementId] = peridigmGlobalIds;
  }

  // Create an overlap version of the Albany solution vector
  selector = stk::mesh::Selector( metaData->universal_part() ) & stk::mesh::Selector( metaData->locally_owned_part() );
  std::vector<stk::mesh::Entity> locallyOwnedElements;
  stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), locallyOwnedElements);
  std::set<int> overlapGlobalNodeIds;
  for(unsigned int iElem=0 ; iElem<locallyOwnedElements.size() ; ++iElem){
    int numNodes = bulkData->num_nodes(locallyOwnedElements[iElem]);
    const stk::mesh::Entity* nodes = bulkData->begin_nodes(locallyOwnedElements[iElem]);
    for(int iNode=0 ; iNode<numNodes ; iNode++){
      int globalNodeId = bulkData->identifier(nodes[iNode]) - 1;
      overlapGlobalNodeIds.insert(globalNodeId);
    }
  }
  Teuchos::ArrayRCP<int> nodeIds(3*overlapGlobalNodeIds.size());
  int index=0;
  for(std::set<int>::iterator it=overlapGlobalNodeIds.begin() ; it!=overlapGlobalNodeIds.end() ; it++){
    int globalId = *it;
    nodeIds[index++] = 3*globalId;
    nodeIds[index++] = 3*globalId + 1;
    nodeIds[index++] = 3*globalId + 2;
  }

  Teuchos::RCP<Tpetra_Map> tpetraMap = Teuchos::rcp(new Tpetra_Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
								   nodeIds(),
								   0,
								   teuchosComm));

  albanyOverlapSolutionVector = Teuchos::rcp(new Tpetra_Vector(tpetraMap));

  if(enableOptimizationBasedCoupling){
    obcOverlappingElementSearch();
  }
}

void LCM::PeridigmManager::obcOverlappingElementSearch()
{
  static bool searchComplete = false;
  if(searchComplete){
    std::cout << "DJL DEBUGGING PeridigmManager::obcOverlappingElementSearch() is being called more than once!" << std::endl;
  }
  searchComplete = true;

  obcDataPoints = Teuchos::rcp(new std::vector<OBCDataPoint>());

  stk::mesh::Field<double,stk::mesh::Cartesian3d>* coordinatesField = 
    metaData->get_field< stk::mesh::Field<double,stk::mesh::Cartesian3d>>(stk::topology::NODE_RANK, "coordinates");
  TEUCHOS_TEST_FOR_EXCEPT_MSG(coordinatesField == 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), unable to access coordinates field.\n\n");

  stk::mesh::Field<double,stk::mesh::Cartesian3d>* volumeField = 
    metaData->get_field< stk::mesh::Field<double,stk::mesh::Cartesian3d>>(stk::topology::ELEMENT_RANK, "volume");
  TEUCHOS_TEST_FOR_EXCEPT_MSG(volumeField == 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), unable to access volume field (volume field is expected because it is assumed that thre are sphere elements in the simulation).\n\n");

  // Create a selector to select everything in the universal part that is locally owned
  stk::mesh::Selector selector =
    stk::mesh::Selector( metaData->universal_part() ) & ( stk::mesh::Selector( metaData->locally_owned_part() ) | stk::mesh::Selector( metaData->globally_shared_part() ) );

  // Select element mesh entities that match the selector
  std::vector<stk::mesh::Entity> elements;
  stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), elements);

  // Collect data for the proximity search
  std::vector<double> proximitySearchCoords(3*elements.size(), 0.0);
  std::vector<double> proximitySearchRadii(elements.size(), 0.0);
  std::vector<int> globalIds(elements.size());
  std::vector<int> isSphere(elements.size());

  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    globalIds[iElem] = bulkData->identifier(elements[iElem]) - 1;
    int numNodes = bulkData->num_nodes(elements[iElem]);
    const stk::mesh::Entity* nodes = bulkData->begin_nodes(elements[iElem]);
    if(numNodes == 1){
      // If the element is a sphere element, include its node in the search and set the search radius to a small number
      double* coord = stk::mesh::field_data(*coordinatesField, nodes[0]);
      double* volume = stk::mesh::field_data(*volumeField, elements[iElem]);
      for(int dof=0 ; dof<3 ; ++dof){
	proximitySearchCoords[3*iElem+dof] = coord[dof];
	// DJL there is a bug in the proximity search, it doesn't work properly in parallel if the radius is set to 0.0.
	// So, set it to some small number, like a 1/1000 times the cube root of the volume.
	proximitySearchRadii[iElem] = 0.001 * std::cbrt(volume[0]);
	isSphere[iElem] = 1;
      }
    }

    else{
      // If the element is not a sphere element, use its barycenter in the proximity search and set
      // the search radius to be slightly larger than the largest element dimension
      double radiusMultiplier = 1.1;
      double largestElementDimensionSquared = 0.0;
      for(int i=0 ; i<numNodes ; ++i){
	double* pt1 = stk::mesh::field_data(*coordinatesField, nodes[i]);
	for(int dof=0 ; dof<3 ; ++dof){
	  proximitySearchCoords[3*iElem+dof] += pt1[dof];
	}
	for(int j=i+1 ; j<numNodes ; ++j){
	  double* pt2 = stk::mesh::field_data(*coordinatesField, nodes[j]);
	  double distanceSquared = (pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1]) + (pt1[2]-pt2[2])*(pt1[2]-pt2[2]);
	  if(distanceSquared > largestElementDimensionSquared){
	    largestElementDimensionSquared = distanceSquared;
	  }
	}
      }
      for(int dof=0 ; dof<3 ; dof++){
	proximitySearchCoords[3*iElem+dof] /= numNodes;
      }
      proximitySearchRadii[iElem] = radiusMultiplier * std::sqrt(largestElementDimensionSquared);
      isSphere[iElem] = 0;
    }
  }

  // Optain a Epetra_MpiComm object using a Teuchos_Comm
  const Teuchos::MpiComm<int>* mpiCommWrapper = dynamic_cast<const Teuchos::MpiComm<int>* >(teuchosComm.get());
  TEUCHOS_TEST_FOR_EXCEPT_MSG(mpiCommWrapper == 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), failed to dynamically cast comm object to Teuchos::MpiComm<int>.\n");
  MPI_Comm mpiComm = static_cast<MPI_Comm>(*mpiCommWrapper->getRawMpiComm());
  TEUCHOS_TEST_FOR_EXCEPT_MSG(mpiComm == 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), failed to dynamically cast comm object to MPI_Comm.\n");
  Epetra_MpiComm epetraComm(mpiComm);

  // Create Epetra_BlockMaps for the Peridigm proximity search
  Epetra_BlockMap epetraOneDimensionalMap(-1,
					  static_cast<int>( elements.size() ),
					  &globalIds[0],
					  1,
					  0,
					  epetraComm);
  Epetra_BlockMap epetraThreeDimensionalMap(-1,
					    static_cast<int>( elements.size() ),
					    &globalIds[0],
					    3,
					    0,
					    epetraComm);

  // Input for proximity search routine
  Teuchos::RCP<Epetra_Vector> epetraProximitySearchCoords = Teuchos::rcp(new Epetra_Vector(epetraThreeDimensionalMap));
  Teuchos::RCP<Epetra_Vector> epetraProximitySearchRadii = Teuchos::rcp(new Epetra_Vector(epetraOneDimensionalMap));
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    for(int dof=0 ; dof<3 ; dof++){
      (*epetraProximitySearchCoords)[3*iElem+dof] = proximitySearchCoords[3*iElem+dof];
    }
    (*epetraProximitySearchRadii)[iElem] = proximitySearchRadii[iElem];
  }

  // Output from proximity search routine
  Teuchos::RCP<Epetra_BlockMap> epetraOneDimensionalOverlapMap;
  int neighborListSize(0);
  int* neighborList(0);

  // Call the Peridigm proximity search routine
  PeridigmNS::ProximitySearch::GlobalProximitySearch(epetraProximitySearchCoords,
						     epetraProximitySearchRadii,
						     epetraOneDimensionalOverlapMap,
						     neighborListSize,
						     neighborList);

  // Create a three-dimensional version of the overlap map
  Epetra_BlockMap epetraThreeDimensionalOverlapMap(epetraOneDimensionalOverlapMap->NumGlobalElements(),
						   epetraOneDimensionalOverlapMap->NumMyElements(),
						   epetraOneDimensionalOverlapMap->MyGlobalElements(),
						   3,
						   0,
						   epetraComm);

  // To determine which peridynamic nodes are within a given on-processor element, we need to ghost
  // the coordinates of each potential node and determine if that node is associated with a sphere element
  Teuchos::RCP<Epetra_Vector> epetraIsSphere = Teuchos::rcp(new Epetra_Vector(epetraOneDimensionalMap));
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    (*epetraIsSphere)[iElem] = static_cast<double>(isSphere[iElem]);
  }

  // Bring all necessary off-processor data onto this processor
  Epetra_Vector epetraOverlapCoords(epetraThreeDimensionalOverlapMap);
  Epetra_Vector epetraOverlapIsSphere(*epetraOneDimensionalOverlapMap);
  Epetra_Import threeDimensionalImporter(epetraThreeDimensionalOverlapMap, epetraThreeDimensionalMap);
  Epetra_Import oneDimensionalImporter(*epetraOneDimensionalOverlapMap, epetraOneDimensionalMap);
  int coordsImportErrorCode = epetraOverlapCoords.Import(*epetraProximitySearchCoords, threeDimensionalImporter, Insert);
  int isSphereImportErrorCode = epetraOverlapIsSphere.Import(*epetraIsSphere, oneDimensionalImporter, Insert);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(coordsImportErrorCode != 0 || isSphereImportErrorCode != 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), import operation failed!\n\n");

  // Store the cell topology for each on-processor element
  std::map<int, CellTopologyData> albanyGlobalElementIdToCellTopolotyData;
  const stk::mesh::PartVector& stkParts = metaData->get_parts();
  for(stk::mesh::PartVector::const_iterator it = stkParts.begin(); it != stkParts.end(); ++it){
    stk::mesh::Part* const part = *it;
    if(!stk::mesh::is_auto_declared_part(*part) && part->primary_entity_rank() == stk::topology::ELEMENT_RANK){
      const CellTopologyData& cellTopologyData = *metaData->get_cell_topology(*part).getCellTopologyData();
      stk::mesh::Selector selector = stk::mesh::Selector( *part ) & stk::mesh::Selector( metaData->locally_owned_part() );
      std::vector<stk::mesh::Entity> elementsInPart;
      stk::mesh::get_selected_entities(selector, bulkData->buckets(stk::topology::ELEMENT_RANK), elementsInPart);
      for(unsigned int iElem=0 ; iElem<elementsInPart.size() ; ++iElem){
	int globalId = bulkData->identifier(elementsInPart[iElem]) - 1;
	albanyGlobalElementIdToCellTopolotyData[globalId] = cellTopologyData;
      }
    }
  }

  // All sphere elements that could possibly be within an on-processor solid element are now available
  // on processor.  Some other points have been ghosted as well (barycenters of solid elements); they
  // will be ignored as we check for peridynamic nodes within each on-processor solid element.

  // After identifying all the peridynamic nodes that lie within an on-processor solid element, a set
  // of Epetra_Vectors will be created for use with ghosting operations downstream.  Initially, data
  // for construction of the Epetra_Vectors will be stored in stl containers
  std::vector<int> peridynamicNodeGlobalIds;
  std::map<int, int> peridynamicNodeOverlappingElement;

  // Loop over all the on-processor nodes and check to see if any of the peridynamic nodes identified
  // by the proximity search are within the element
  int neighborhoodListIndex = 0;
  typedef PHX::KokkosViewFactory<RealType, PHX::Device> ViewFactory;

  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){

    // Get the elements nodes and cell topology
    int numNodesInElement = bulkData->num_nodes(elements[iElem]);
    const stk::mesh::Entity* nodesInElement = bulkData->begin_nodes(elements[iElem]);
    int globalElementId = bulkData->identifier(elements[iElem]) - 1;
    std::map<int, CellTopologyData>::iterator it = albanyGlobalElementIdToCellTopolotyData.find(globalElementId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(it == albanyGlobalElementIdToCellTopolotyData.end(), "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), failed to find cell topology.\n\n");
    const CellTopologyData& cellTopologyData = it->second;
    shards::CellTopology cellTopology(&cellTopologyData);

    // The neighbors are the nodes that might be within the element
    int numNeighbors = neighborList[neighborhoodListIndex++];

    // Skip peridynamic nodes
    if(numNodesInElement == 1){
      neighborhoodListIndex += numNeighbors;
    }
    else{

      for(int iNeighbor=0 ; iNeighbor<numNeighbors ; iNeighbor++){

	int neighborIndex = neighborList[neighborhoodListIndex++];
	int neighborIsSphere = epetraOverlapIsSphere[neighborIndex];

	// If the neighbor is a peridynamic sphere element, check to see if it's within the solid element
	if(neighborIsSphere){

	  std::vector<double> neighborCoords(3);
	  for(int dof=0 ; dof<3 ; dof++){
	    neighborCoords[dof] = epetraOverlapCoords[3*neighborIndex+dof];
	  }

	  // We're interested in a single point in a single element in a three-dimensional simulation
 	  int numCells = 1;
	  int numQuadPoints = 1;
	  int numDim = 3;

	  // Physical points, which are the physical (x, y, z) values of the peridynamic node (pay no attention to the "quadrature point" descriptor)
	  Intrepid::FieldContainer<RealType> physPoints;
	  physPoints.resize(numCells, numQuadPoints, numDim);

	  // Reference points, which are the natural coordinates of the quadrature points
	  Intrepid::FieldContainer<RealType> refPoints;
	  refPoints.resize(numCells, numQuadPoints, numDim);

	  // Cell workset, which is the set of nodes for the given element
	  Intrepid::FieldContainer<RealType> cellWorkset;
	  cellWorkset.resize(numCells, numNodesInElement, numDim);

 	  for(int dof=0 ; dof<3 ; dof++){
 	    physPoints(0, 0, dof) = neighborCoords[dof];
 	  }

 	  for(int i=0 ; i<numNodesInElement ; i++){
 	    double* coordinates = stk::mesh::field_data(*coordinatesField, nodesInElement[i]);
 	    for(int dof=0 ; dof<3 ; dof++){
 	      cellWorkset(0, i, dof) = coordinates[dof];
 	    }
 	  }

	  Intrepid::CellTools<RealType>::mapToReferenceFrame(refPoints, physPoints, cellWorkset, cellTopology, -1);

	  bool refPointsAreNan = !boost::math::isfinite(refPoints(0,0,0)) || !boost::math::isfinite(refPoints(0,0,1)) || !boost::math::isfinite(refPoints(0,0,2));
	  TEUCHOS_TEST_FOR_EXCEPT_MSG(refPointsAreNan, "\n**** Error in PeridigmManager::obcOverlappingElementSearch(), NaN in refPoints.\n");

	  std::vector<RealType> point(3);
	  for(int dof=0 ; dof<3 ; dof++){
	    point[dof] = refPoints(0, 0, dof);
	  }
	  int inElement = Intrepid::CellTools<RealType>::checkPointInclusion(&point[0], numDim, cellTopology);

	  if(inElement){
	    OBCDataPoint dataPoint;
	    for(int dof=0 ; dof<3 ; dof++){
	      dataPoint.initialCoords[dof] = neighborCoords[dof];
	      dataPoint.currentCoords[dof] = 0.0;
	      dataPoint.naturalCoords[dof] = point[dof];
	    }
	    dataPoint.peridigmGlobalId = epetraOverlapIsSphere.Map().GID(neighborIndex);
	    dataPoint.albanyElement = elements[iElem];
	    dataPoint.cellTopologyData = cellTopologyData;
	    obcDataPoints->push_back(dataPoint);
	  }
	}
      }
    }
  }

  // Store the volume of the sphere element
  const Epetra_Vector& peridigmVolume = *(peridigm->getVolume());
  const Epetra_BlockMap& peridigmVolumeMap = peridigmVolume.Map();
  for(unsigned int i=0 ; i<obcDataPoints->size() ; i++){
    int globalId = (*obcDataPoints)[i].peridigmGlobalId;
    int localId = peridigmVolumeMap.LID(globalId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(localId < 0, "\n\n**** Error in PeridigmManager::obcOverlappingElementSearch(), invalid local id!\n\n");
    (*obcDataPoints)[i].sphereElementVolume = peridigmVolume[localId];
  }

  // Create an Epetra_Vector for importing the displacements of the peridynamic nodes
  std::vector<int> tempGlobalIds(obcDataPoints->size());
  for(unsigned int i=0 ; i<obcDataPoints->size() ; i++){
    tempGlobalIds[i] = (*obcDataPoints)[i].peridigmGlobalId;
  }

  Epetra_BlockMap epetraTempMap(-1,
				static_cast<int>( tempGlobalIds.size() ),
				&tempGlobalIds[0],
				3,
				0,
				epetraComm);

  obcPeridynamicNodeCurrentCoords = Teuchos::rcp(new Epetra_Vector(epetraTempMap));

  // As a sanity check, determine the total number of overlapping peridynamic nodes
  vector<int> localVal(1), globalVal(1);
  localVal[0] = static_cast<int>(obcDataPoints->size());
  Teuchos::reduceAll(*teuchosComm, Teuchos::REDUCE_SUM, 1, &localVal[0], &globalVal[0]);
  int numberOfOverlappingPeridynamicNodes = globalVal[0];

  std::cout << "\n-- Overlapping Element Search --" << std::endl;
  std::cout << "  number of peridynamic nodes in overlap region: " << numberOfOverlappingPeridynamicNodes << std::endl;
}

double LCM::PeridigmManager::obcEvaluateFunctional(Epetra_Vector* obcFunctionalDerivWrtDisplacement)
{
  if(!enableOptimizationBasedCoupling){
    return 0.0;
  }




  // Set up access to the current displacements of the nodes in the solid elements
  Teuchos::ArrayRCP<const ST> albanyCurrentDisplacement = albanyOverlapSolutionVector->getData();
  const Teuchos::RCP<const Tpetra_Map> albanyMap = albanyOverlapSolutionVector->getMap();

  // Load the current displacements into the obcDataPoints data structures
  Epetra_Vector& peridigmCurrentPositions = *(peridigm->getY());
  Epetra_Import overlapCurrentCoordsImporter(obcPeridynamicNodeCurrentCoords->Map(), peridigmCurrentPositions.Map());
  int err = obcPeridynamicNodeCurrentCoords->Import(peridigmCurrentPositions, overlapCurrentCoordsImporter, Insert);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(err != 0, "\n\n**** Error in PeridigmManager::obcEvaluateFunctional(), import operation failed!\n\n");
  for(unsigned int iEvalPt=0 ; iEvalPt<obcDataPoints->size() ; iEvalPt++){
    int localId = obcPeridynamicNodeCurrentCoords->Map().LID((*obcDataPoints)[iEvalPt].peridigmGlobalId);
    for(int dof=0 ; dof<3 ; dof++){
      (*obcDataPoints)[iEvalPt].currentCoords[dof] = (*obcPeridynamicNodeCurrentCoords)[3*localId+dof];
    }
  }

  // Creating Overlapped obcFunctionalDerivWrtDisplacement
  Teuchos::RCP<Epetra_Vector> obcFunctionalDerivWrtDisplacementOverlap;
  Teuchos::RCP<Epetra_Export> overlapSolutionExporter;
  if(obcFunctionalDerivWrtDisplacement != NULL){
    obcFunctionalDerivWrtDisplacementOverlap = Teuchos::rcp<Epetra_Vector>(new Epetra_Vector(*stkDisc->getOverlapMap()));
    overlapSolutionExporter = Teuchos::rcp<Epetra_Export>(new Epetra_Export(*stkDisc->getOverlapMap(), obcFunctionalDerivWrtDisplacement->Map()));
  }

  // We're interested in a single point in a single element in a three-dimensional simulation
  int numCells = 1;
  int numPoints = 1;
  int numDim = 3;

  Intrepid::FieldContainer<RealType> physPoints;
  physPoints.resize(numCells, numPoints, numDim);

  Intrepid::FieldContainer<RealType> refPoints;
  refPoints.resize(numCells, numPoints, numDim);

  // Compute the difference in displacements at each peridynamic node
  Epetra_Vector displacementDiff(obcPeridynamicNodeCurrentCoords->Map());
  Epetra_Vector displacementDiffScaled(obcPeridynamicNodeCurrentCoords->Map());
  for(unsigned int iEvalPt=0 ; iEvalPt<obcDataPoints->size() ; iEvalPt++){

    for(int dof=0 ; dof<3 ; dof++){
      refPoints(0, 0, dof) = (*obcDataPoints)[iEvalPt].naturalCoords[dof];
    }

    int numNodes = bulkData->num_nodes((*obcDataPoints)[iEvalPt].albanyElement);
    const stk::mesh::Entity* nodes = bulkData->begin_nodes((*obcDataPoints)[iEvalPt].albanyElement);

    Intrepid::FieldContainer<RealType> cellWorkset;
    cellWorkset.resize(numCells, numNodes, numDim);
    for(int i=0 ; i<numNodes ; i++){
      int globalAlbanyNodeId = bulkData->identifier(nodes[i]) - 1;
      Tpetra_Map::local_ordinal_type albanyLocalId = albanyMap->getLocalElement(3*globalAlbanyNodeId);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(albanyLocalId == Teuchos::OrdinalTraits<LO>::invalid(), "\n\n**** Error in PeridigmManager::obcEvaluateFunctional(), invalid Albany local id.\n\n");
      cellWorkset(0, i, 0) = albanyCurrentDisplacement[albanyLocalId];
      cellWorkset(0, i, 1) = albanyCurrentDisplacement[albanyLocalId + 1];
      cellWorkset(0, i, 2) = albanyCurrentDisplacement[albanyLocalId + 2];
    }

    shards::CellTopology cellTopology(&(*obcDataPoints)[iEvalPt].cellTopologyData);

    Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, cellTopology);

    // Record the difference between the Albany displacement at the point (which was just computed using Intrepid) and
    // the Peridigm displacement at the point
    for(int dof=0 ; dof<3 ; dof++){
      displacementDiff[3*iEvalPt+dof] = physPoints(0,0,dof) - ((*obcDataPoints)[iEvalPt].currentCoords[dof] - (*obcDataPoints)[iEvalPt].initialCoords[dof]);
      // Multiply the displacement vector by the sphere element volume
      displacementDiffScaled[3*iEvalPt+dof] = displacementDiff[3*iEvalPt+dof]*(*obcDataPoints)[iEvalPt].sphereElementVolume;
    }

    if(obcFunctionalDerivWrtDisplacement != NULL) {
      Intrepid::FieldContainer<RealType> refPoint;
      refPoint.resize(numPoints, numDim);
      for(int dof=0 ; dof<3 ; dof++)
        refPoint(0, dof) = refPoints(0, 0, dof);

      Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType>>> refBasis = Albany::getIntrepidBasis((*obcDataPoints)[iEvalPt].cellTopologyData);
      Intrepid::FieldContainer<RealType> basisOnRefPoint(numNodes, 1);
      refBasis->getValues(basisOnRefPoint, refPoint, Intrepid::OPERATOR_VALUE);

      // Derivatives corresponding to nodal dof in Albany element
      double deriv[3];
      int globalNodeIds[3];
      for(int i=0 ; i<numNodes ; i++) {
        int globalAlbanyNodeId = bulkData->identifier(nodes[i]) - 1;

        for(int dim=0; dim<3; ++dim) {
          deriv[dim] = 2*displacementDiffScaled[3*iEvalPt+dim]*basisOnRefPoint(i,0);
          globalNodeIds[dim] = 3*globalAlbanyNodeId + dim;
        }
        obcFunctionalDerivWrtDisplacementOverlap->SumIntoGlobalValues(3, deriv, globalNodeIds);
      }

      // Derivatives corresponding to dof at peridigm node
      for(int dim=0; dim<3; ++dim) {
        deriv[dim] = -2*displacementDiffScaled[3*iEvalPt+dim];
        globalNodeIds[dim] = 3*((*obcDataPoints)[iEvalPt].peridigmGlobalId) + dim;
      }
      obcFunctionalDerivWrtDisplacementOverlap->SumIntoGlobalValues(3, deriv, globalNodeIds);
    }
  }

  // Assemble the derivative of the functional
  if(obcFunctionalDerivWrtDisplacement != NULL) {
    obcFunctionalDerivWrtDisplacement->Export(*obcFunctionalDerivWrtDisplacementOverlap, *overlapSolutionExporter, Add);
  }

  // Send displacement differences to Peridigm for output
  Teuchos::RCP< std::vector<PeridigmNS::Block>> peridigmBlocks = peridigm->getBlocks();
  for(unsigned int iBlock=0 ; iBlock<peridigmBlocks->size() ; iBlock++){
    std::string blockName = (*peridigmBlocks)[iBlock].getName();
    bool hasOBCFunctional = peridigm->hasBlockData(blockName, "OBC_Functional");
    if(hasOBCFunctional){
      Teuchos::RCP<Epetra_Vector> data = peridigm->getBlockData(blockName, "OBC_Functional"); 
      Epetra_Import importer(data->Map(), displacementDiff.Map());
      int importErrorCode = data->Import(displacementDiff, importer, Insert);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(importErrorCode != 0, "\n\n**** Error in PeridigmManager::obcEvaluateFunctional(), import operation failed!\n\n");
    }
  }

  // Evaluate the functional
  double functionalValue(0.0);
  displacementDiff.Dot(displacementDiffScaled, &functionalValue);

  return functionalValue;
}

void LCM::PeridigmManager::setCurrentTimeAndDisplacement(double time, const Teuchos::RCP<const Tpetra_Vector>& albanySolutionVector)
{
  if(hasPeridynamics){

    Tpetra_Import tpetraImport(albanySolutionVector->getMap(), albanyOverlapSolutionVector->getMap());
    albanyOverlapSolutionVector->doImport(*albanySolutionVector, tpetraImport, Tpetra::INSERT);

    currentTime = time;
    timeStep = currentTime - previousTime;
    // Odd undefined things can happen if the time step is zero (e.g., if force is evaluated at time zero)
    // Hack around this situation.
    if(timeStep <= 0.0)
      timeStep = 1.0;
    peridigm->setTimeStep(timeStep);

    Epetra_Vector& peridigmReferencePositions = *(peridigm->getX());
    Epetra_Vector& peridigmCurrentPositions = *(peridigm->getY());
    Epetra_Vector& peridigmDisplacements = *(peridigm->getU());
    Epetra_Vector& peridigmVelocities = *(peridigm->getV());

    // Peridynamic elements (sphere elements)
    Teuchos::ArrayRCP<const ST> albanyCurrentDisplacement = albanyOverlapSolutionVector->getData();
    const Teuchos::RCP<const Tpetra_Map> albanyMap = albanyOverlapSolutionVector->getMap();
    Tpetra_Map::local_ordinal_type albanyLocalId;
    const Epetra_BlockMap& peridigmMap = peridigmCurrentPositions.Map();
    int peridigmLocalId, globalId;

    for(unsigned int i=0 ; i<sphereElementGlobalNodeIds.size() ; ++i){
      globalId = sphereElementGlobalNodeIds[i];
      peridigmLocalId = peridigmMap.LID(globalId);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Peridigm local id.\n\n");
      albanyLocalId = albanyMap->getLocalElement(3*globalId);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(albanyLocalId == Teuchos::OrdinalTraits<LO>::invalid(), "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Albany local id.\n\n");
      peridigmDisplacements[3*peridigmLocalId]   = albanyCurrentDisplacement[albanyLocalId];
      peridigmDisplacements[3*peridigmLocalId+1] = albanyCurrentDisplacement[albanyLocalId+1];
      peridigmDisplacements[3*peridigmLocalId+2] = albanyCurrentDisplacement[albanyLocalId+2];
      peridigmCurrentPositions[3*peridigmLocalId]   = peridigmReferencePositions[3*peridigmLocalId] + peridigmDisplacements[3*peridigmLocalId];
      peridigmCurrentPositions[3*peridigmLocalId+1] = peridigmReferencePositions[3*peridigmLocalId+1] + peridigmDisplacements[3*peridigmLocalId+1];
      peridigmCurrentPositions[3*peridigmLocalId+2] = peridigmReferencePositions[3*peridigmLocalId+2] + peridigmDisplacements[3*peridigmLocalId+2];
      peridigmVelocities[3*peridigmLocalId]   = (peridigmCurrentPositions[3*peridigmLocalId]   - previousSolutionPositions[3*peridigmLocalId])/timeStep;
      peridigmVelocities[3*peridigmLocalId+1] = (peridigmCurrentPositions[3*peridigmLocalId+1] - previousSolutionPositions[3*peridigmLocalId+1])/timeStep;
      peridigmVelocities[3*peridigmLocalId+2] = (peridigmCurrentPositions[3*peridigmLocalId+2] - previousSolutionPositions[3*peridigmLocalId+2])/timeStep;
    }

    // Partial stress elements (solid elements with peridynamic material points at each integration point)

    for(std::vector<PartialStressElement>::iterator it=partialStressElements.begin() ; it!=partialStressElements.end() ; it++){

      // \todo This is brutal, need to store these data structures instead of re-creating them every time for every element.
      // Can probably store things by block and use worksets to compute things in one big call.

      shards::CellTopology cellTopology(&it->cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType>> cubature = cubFactory.create(cellTopology, cubatureDegree);
      const int numDim = cubature->getDimension();
      const int numQuadPoints = cubature->getNumPoints();
      const int numNodes = cellTopology.getNodeCount();
      const int numCells = 1;

      // Get the quadrature points and weights
      Intrepid::FieldContainer<RealType> quadratureRefPoints;
      Intrepid::FieldContainer<RealType> quadratureRefWeights;
      quadratureRefPoints.resize(numQuadPoints, numDim);
      quadratureRefWeights.resize(numQuadPoints);
      cubature->getCubature(quadratureRefPoints, quadratureRefWeights);

      typedef PHX::KokkosViewFactory<RealType, PHX::Device> ViewFactory;

      // Physical points, which are the physical (x, y, z) values of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim>> physPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> physPoints("Physical Points", physPointsLayout);
      physPoints.setFieldData(ViewFactory::buildView(physPoints.fieldTag()));

      // Reference points, which are the natural coordinates of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim>> refPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> refPoints("Reference Points", refPointsLayout);
      refPoints.setFieldData(ViewFactory::buildView(refPoints.fieldTag()));

      // Cell workset, which is the set of nodes for the given element
      Teuchos::RCP< PHX::MDALayout<Cell, Node, Dim>> cellWorksetLayout = Teuchos::rcp(new PHX::MDALayout<Cell, Node, Dim>(numCells, numNodes, numDim));
      PHX::MDField<RealType, Cell, Node, Dim> cellWorkset("Cell Workset", cellWorksetLayout);
      cellWorkset.setFieldData(ViewFactory::buildView(cellWorkset.fieldTag()));

      // Copy the reference points from the Intrepid::FieldContainer to a PHX::MDField
      for(int qp=0 ; qp<numQuadPoints ; ++qp){
	for(int dof=0 ; dof<3 ; ++dof){
	  refPoints(0, qp, dof) = quadratureRefPoints(qp, dof);
	}
      }

      int numNodesInElement = bulkData->num_nodes(it->albanyElement);
      const stk::mesh::Entity* node = bulkData->begin_nodes(it->albanyElement);
      Teuchos::ArrayRCP<const ST> albanyCurrentDisplacement = albanyOverlapSolutionVector->getData();

      for(int i=0 ; i<numNodesInElement ; i++){
	int globalAlbanyNodeId = bulkData->identifier(node[i]) - 1;
	Tpetra_Map::local_ordinal_type albanyLocalId = albanyMap->getLocalElement(3*globalAlbanyNodeId);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(albanyLocalId == Teuchos::OrdinalTraits<LO>::invalid(), "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Albany local id.\n\n");
	cellWorkset(0, i, 0) = it->albanyNodeInitialPositions[3*i]   + albanyCurrentDisplacement[albanyLocalId];
	cellWorkset(0, i, 1) = it->albanyNodeInitialPositions[3*i+1] + albanyCurrentDisplacement[albanyLocalId + 1];
	cellWorkset(0, i, 2) = it->albanyNodeInitialPositions[3*i+2] + albanyCurrentDisplacement[albanyLocalId + 2];
      }

      // Determine the global (x,y,z) coordinates of the quadrature points
      Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, cellTopology);

      for(unsigned int i=0 ; i<it->peridigmGlobalIds.size() ; ++i){
	peridigmLocalId = peridigmGlobalIdToPeridigmLocalId[it->peridigmGlobalIds[i]];
	TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Peridigm local id.\n\n");
	peridigmCurrentPositions[3*peridigmLocalId]   = physPoints(0, i, 0);
	peridigmCurrentPositions[3*peridigmLocalId+1] = physPoints(0, i, 1);
	peridigmCurrentPositions[3*peridigmLocalId+2] = physPoints(0, i, 2);
	peridigmDisplacements[3*peridigmLocalId]   = peridigmCurrentPositions[3*peridigmLocalId]   - peridigmReferencePositions[3*peridigmLocalId];
	peridigmDisplacements[3*peridigmLocalId+1] = peridigmCurrentPositions[3*peridigmLocalId+1] - peridigmReferencePositions[3*peridigmLocalId+1];
	peridigmDisplacements[3*peridigmLocalId+2] = peridigmCurrentPositions[3*peridigmLocalId+2] - peridigmReferencePositions[3*peridigmLocalId+2];
	peridigmVelocities[3*peridigmLocalId]   = (peridigmCurrentPositions[3*peridigmLocalId]   - previousSolutionPositions[3*peridigmLocalId])/timeStep;
	peridigmVelocities[3*peridigmLocalId+1] = (peridigmCurrentPositions[3*peridigmLocalId+1] - previousSolutionPositions[3*peridigmLocalId+1])/timeStep;
	peridigmVelocities[3*peridigmLocalId+2] = (peridigmCurrentPositions[3*peridigmLocalId+2] - previousSolutionPositions[3*peridigmLocalId+2])/timeStep;
      }
    }
  }
}

void LCM::PeridigmManager::updateState()
{
  if(hasPeridynamics){
    previousTime = currentTime;
    const Teuchos::RCP<const Epetra_Vector> peridigmY = peridigm->getY();
    for(unsigned int i=0 ; i<previousSolutionPositions.size() ; ++i)
      previousSolutionPositions[i] = (*peridigmY)[i];
    peridigm->updateState();
  }
}

void LCM::PeridigmManager::writePeridigmSubModel(RealType currentTime)
{
  if(hasPeridynamics)
    peridigm->writePeridigmSubModel(currentTime);
}

void LCM::PeridigmManager::evaluateInternalForce()
{
  if(hasPeridynamics)
    peridigm->computeInternalForce();
}

double LCM::PeridigmManager::getForce(int globalAlbanyNodeId, int dof)
{
  double force(0.0);

  if(hasPeridynamics){
    Epetra_Vector& peridigmForce = *(peridigm->getForce());
    int peridigmLocalId = peridigmForce.Map().LID(globalAlbanyNodeId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::getForce(), invalid global id.\n\n");
    force = peridigmForce[3*peridigmLocalId + dof];
  }
  return force;
}

double LCM::PeridigmManager::getDisplacementNeighborhoodFit(int globalAlbanyNodeId, double * coord, int dof)
{
  double fitDisp(0.0);

  if(hasPeridynamics){
    Epetra_Vector& peridigmU = *(peridigm->getU());
    Epetra_Vector& peridigmX = *(peridigm->getX());

    // determine the nieghbors to collect disp values from and fit with a polynomial:

    Teuchos::RCP<PeridigmNS::AlbanyDiscretization> castDisc = Teuchos::rcp_dynamic_cast<PeridigmNS::AlbanyDiscretization>(peridynamicDiscretization);
    Teuchos::RCP<PeridigmNS::NeighborhoodData> neighborhoodData = castDisc->getAlbanyPartialStressNeighborhoodData();

    const int numOwnedPoints = neighborhoodData->NumOwnedPoints();
    const int localAlbanyNodeId = castDisc->getAlbanyInterface1DMap()->LID(globalAlbanyNodeId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(localAlbanyNodeId>=numOwnedPoints || localAlbanyNodeId<0, "\n\n**** Error in PeridigmManager::getDisplacementNeighborhoodFit(), invalid local id.\n\n");

    Teuchos::ArrayRCP<double> neighborDispValues;
    Teuchos::ArrayRCP<double> neighborXValues;
    Teuchos::ArrayRCP<double> neighborYValues;
    Teuchos::ArrayRCP<double> neighborZValues;

    const int* neighborhoodList = neighborhoodData->NeighborhoodList();
    int neighborhoodListIndex = 0;
    for(int iID=0 ; iID<numOwnedPoints ; ++iID){
      // Sum in the contributions for the neighbors
      int numNeighbors = neighborhoodList[neighborhoodListIndex++];
      if(iID==localAlbanyNodeId){
        neighborDispValues = Teuchos::ArrayRCP<double>(numNeighbors,0.0);
        neighborXValues = Teuchos::ArrayRCP<double>(numNeighbors,0.0);
        neighborYValues = Teuchos::ArrayRCP<double>(numNeighbors,0.0);
        neighborZValues = Teuchos::ArrayRCP<double>(numNeighbors,0.0);
      }
      for(int iNID=0 ; iNID<numNeighbors ; ++iNID){
        int neighborID = neighborhoodList[neighborhoodListIndex++];
        if(iID==localAlbanyNodeId){
          neighborDispValues[iNID] = peridigmU[3*neighborID+dof];
          neighborXValues[iNID] = peridigmX[3*neighborID+0];
          neighborYValues[iNID] = peridigmX[3*neighborID+1];
          neighborZValues[iNID] = peridigmX[3*neighborID+2];
        }
      }
    } // iID

    // least squares linear fit in each dimension:

    const int N = 4;
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    int INFO = 0;
    double *WORK = new double[LWORK];
    double *GWORK = new double[10*N];
    int *IWORK = new int[LWORK];
    Teuchos::LAPACK<int,double> lapack;
    int num_neigh = neighborXValues.size();

    Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
    Teuchos::ArrayRCP<double> coeffs(N,0.0);
    Teuchos::SerialDenseMatrix<int,double> X_t(N,num_neigh, true);
    Teuchos::SerialDenseMatrix<int,double> X_t_X(N,N,true);

    // set up the X^T matrix
    for(int j=0;j<num_neigh;++j){
      X_t(0,j) = 1.0;
      X_t(1,j) = neighborXValues[j];
      X_t(2,j) = neighborYValues[j];
      X_t(3,j) = neighborZValues[j];
    }
    // set up X^T*X
    for(int k=0;k<N;++k){
      for(int m=0;m<N;++m){
        for(int j=0;j<num_neigh;++j){
          X_t_X(k,m) += X_t(k,j)*X_t(m,j);
        }
      }
    }
    //X_t_X.print(std::cout);

    // Invert X^T*X
    // TODO: remove for performance?
    // compute the 1-norm of H:
    std::vector<double> colTotals(X_t_X.numCols(),0.0);
    for(int i=0;i<X_t_X.numCols();++i){
      for(int j=0;j<X_t_X.numRows();++j){
        colTotals[i]+=std::abs(X_t_X(j,i));
      }
    }
    double anorm = 0.0;
    for(int i=0;i<X_t_X.numCols();++i){
      if(colTotals[i] > anorm) anorm = colTotals[i];
    }
    double rcond=0.0; // reciporical condition number
    try
    {
      lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
      lapack.GECON('1',X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),anorm,&rcond,GWORK,IWORK,&INFO);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(rcond < 1.0E-12, "\n\n**** Error, The pseudo-inverse of the least squares fit is (or is near) singular.\n\n");
    }
    catch(std::exception &e){
      std::cout << e.what();
      TEUCHOS_TEST_FOR_EXCEPT_MSG(false, "\n\n**** Error, Something went wrong in the condition number calculation.\n\n");
    }
    try
    {
      lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      std::cout << e.what();
      TEUCHOS_TEST_FOR_EXCEPT_MSG(false, "\n\n**** Error, Something went wrong in the inverse calculation of X^T*X .\n\n");
    }

    // compute X^T*u
    for(int i=0;i<N;++i){
      for(int j=0;j<num_neigh;++j){
        X_t_u_x[i] += X_t(i,j)*neighborDispValues[j];
      }
    }

    // compute the coeffs
    for(int i=0;i<N;++i){
      for(int j=0;j<N;++j){
        coeffs[i] += X_t_X(i,j)*X_t_u_x[j];
      }
    }

    fitDisp = coeffs[0] + coeffs[1]*coord[0] +  coeffs[2]*coord[1] +  coeffs[3]*coord[2];

    delete [] WORK;
    delete [] GWORK;
    delete [] IWORK;
    delete [] IPIV;

    // Do a weighted average of the disp values to compare

//    double totalX = 0.0;
//    double compareDisp = 0.0;
//    for(int i=0;i<neighborXValues.size();++i)
//      totalX += std::abs(neighborXValues[i]-coord[dof]);
//    TEUCHOS_TEST_FOR_EXCEPT_MSG(neighborXValues.size()!=neighborDispValues.size(), "\n\n**** Error in PeridigmManager, neighbor arrays are not equal size.\n\n");
//    for(int i=0;i<neighborDispValues.size();++i)
//      compareDisp += neighborDispValues[i]*std::abs(neighborXValues[i]-coord[dof])/totalX;
  }
  return fitDisp;
}


void LCM::PeridigmManager::getPartialStress(std::string blockName, int worksetIndex, int worksetLocalElementId, std::vector< std::vector<RealType>>& partialStressValues)
{
  if(hasPeridynamics){

    int globalElementId = worksetLocalIdToGlobalId[worksetIndex][worksetLocalElementId];
    std::vector<int>& peridigmGlobalIds = albanyPartialStressElementGlobalIdToPeridigmGlobalIds[globalElementId];
    Teuchos::RCP<const Epetra_Vector> data = peridigm->getBlockData(blockName, "Partial_Stress");
    Teuchos::RCP<const Epetra_Vector> displacement = peridigm->getBlockData(blockName, "Displacement");
    for(unsigned int i=0 ; i<peridigmGlobalIds.size() ; ++i){
      int peridigmLocalId = data->Map().LID(peridigmGlobalIds[i]);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::getPartialStress(), invalid global id.\n");
      for(int j=0 ; j<9 ; ++j){
	partialStressValues[i][j] = (*data)[9*peridigmLocalId + j];
      }
    }
  }
}

Teuchos::RCP<const Epetra_Vector> LCM::PeridigmManager::getBlockData(std::string blockName, std::string fieldName)
{

  Teuchos::RCP<const Epetra_Vector> data;

  if(hasPeridynamics)
    data = peridigm->getBlockData(blockName, fieldName);

  return data;
}

void LCM::PeridigmManager::setOutputFields(const Teuchos::ParameterList& params)
{
  for(Teuchos::ParameterList::ConstIterator it = params.begin(); it != params.end(); ++it) {
    std::string name = it->first;

    // Hard-code everything for the initial implementation
    // It would be best to just use the PeridigmNS::FieldManager to figure out if
    // a variable is a global, nodal, or element variable and if it is scalar or vector
    // But there is an order-of-operations issue; the PeridigmNS::FieldManager has not
    // been instantiated yet, so at this point it is empty

    OutputField field;
    field.albanyName = "Peridigm_" + name;
    field.peridigmName = name;

    if(name == "Dilatation" || name == "Weighted_Volume" || name == "Radius" || name == "Number_Of_Neighbors" || name == "Horizon" || name == "Volume"){
      field.relation = "element";
      field.initType = "scalar";
      field.length = 1;
    }
    else if(name == "OBC_Functional"){
      field.relation = "element";
      field.initType = "scalar";
      field.length = 3;
    }
    else if(name == "Model_Coordinates" || name == "Coordinates" || name == "Displacement" || name == "Velocity" || name == "Force"){
      field.relation = "node";
      field.initType = "scalar";
      field.length = 3;
    }
    else if(name == "Deformation_Gradient" || name == "Unrotated_Rate_Of_Deformation" || name == "Cauchy_Stress" || name == "Partial_Stress"){
      field.relation = "element";
      field.initType = "scalar";
      field.length = 9;
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "\n\n**** Error in PeridigmManager::setOutputVariableList(), unknown variable.  All variables must be hard-coded in PeridigmManager.cpp (sad but true).\n\n");
    }

    if( std::find(outputFields.begin(), outputFields.end(), field) == outputFields.end() )
      outputFields.push_back(field);
  }
}

void LCM::PeridigmManager::setDirichletFields(Teuchos::RCP<Albany::AbstractDiscretization> disc){
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  Teuchos::RCP<Albany::STKDiscretization> stkDisc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc);
  ScalarFieldType* dirichletField = stkDisc->getSTKMetaData().get_field <ScalarFieldType> (stk::topology::NODE_RANK, "dirichlet_field");
  ScalarFieldType* dirichletControlField = stkDisc->getSTKMetaData().get_field <ScalarFieldType> (stk::topology::NODE_RANK, "dirichlet_control_field");
  if (dirichletField != NULL) {
    {
      const std::string& meshPart = "nodelist_1"; //TODO: make this general
      Albany::NodeSetGIDsList::const_iterator it = disc->getNodeSetGIDs ().find (meshPart);
      if (it != disc->getNodeSetGIDs ().end ()) {
        const std::vector<GO>& nsNodesGIDs = it->second;
        for (int i = 0; i < nsNodesGIDs.size (); ++i) {
          GO gId = nsNodesGIDs[i];
          stk::mesh::Entity node = stkDisc->getSTKBulkData ().get_entity (stk::topology::NODE_RANK, gId + 1);
          double* coord = stk::mesh::field_data (*stkDisc->getSTKMeshStruct ()->getCoordinatesField (), node);
          double* dirichletData = stk::mesh::field_data (*dirichletField, node);

          //set dirichletData as any function of the coordinates;
          dirichletData[0] = 0.001*(coord[0]/1.625);
        }
      }
    }
    {
      const std::string& meshPart = "nodelist_2"; //TODO: make this general
      Albany::NodeSetGIDsList::const_iterator it= disc->getNodeSetGIDs().find(meshPart);
      if(it != disc->getNodeSetGIDs().end()) {
        const std::vector<GO>& nsNodesGIDs = it->second;
        for(int i=0; i<nsNodesGIDs.size(); ++i) {
          GO gId = nsNodesGIDs[i];
          stk::mesh::Entity node = stkDisc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, gId + 1);
          double* coord = stk::mesh::field_data(*stkDisc->getSTKMeshStruct()->getCoordinatesField(), node);
          double* dirichletData = stk::mesh::field_data(*dirichletField, node);
          //set dirichletData as any function of the coordinates;
          dirichletData[0]= 0.001*(coord[0]/1.625);
        }
      }
    }
  }

  if (dirichletControlField != NULL) {
    const std::string& meshPart = "nodelist_5"; //TODO: make this general
    Albany::NodeSetGIDsList::const_iterator it= disc->getNodeSetGIDs().find(meshPart);
    if(it != disc->getNodeSetGIDs().end()) {
      const std::vector<GO>& nsNodesGIDs = it->second;
      for(int i=0; i<nsNodesGIDs.size(); ++i) {
        GO gId = nsNodesGIDs[i];
        stk::mesh::Entity node = stkDisc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, gId + 1);
        double* coord = stk::mesh::field_data(*stkDisc->getSTKMeshStruct()->getCoordinatesField(), node);
        double* dirichletControlData = stk::mesh::field_data(*dirichletControlField, node);
        //set dirichletData as any function of the coordinates;
        dirichletControlData[0]= 0;
      }
    }
  }

}

std::vector<LCM::PeridigmManager::OutputField> LCM::PeridigmManager::getOutputFields()
{
  return outputFields;
}
