/*! \file PeridigmManager.cpp */

#include "PeridigmManager.hpp"
#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include "Phalanx_DataLayout.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "PHAL_Dimension.hpp"

LCM::PeridigmManager& LCM::PeridigmManager::self() {
  static PeridigmManager peridigmManager;
  return peridigmManager;
}

LCM::PeridigmManager::PeridigmManager() : hasPeridynamics(false), previousTime(0.0), currentTime(0.0), timeStep(0.0), cubatureDegree(-1)
{
  epetraComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
}

void LCM::PeridigmManager::initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                      Teuchos::RCP<Albany::AbstractDiscretization> disc)
{
#ifndef ALBANY_PERIDIGM

  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "\n\n**** Error:  Peridigm interface not available!  Recompile with -DUSE_PERIDIGM.\n\n");

#else

  peridigmParams = Teuchos::RCP<Teuchos::ParameterList>(new Teuchos::ParameterList(params->sublist("Problem").sublist("Peridigm Parameters", true)));
  Teuchos::ParameterList& problemParams = params->sublist("Problem");
  Teuchos::ParameterList& discretizationParams = params->sublist("Discretization");
  cubatureDegree = discretizationParams.get<int>("Cubature Degree", 2);

  // Read the material data base file, if any
  Teuchos::RCP<QCAD::MaterialDatabase> materialDataBase;
  if(problemParams.isType<std::string>("MaterialDB Filename")){
    std::string filename = problemParams.get<std::string>("MaterialDB Filename");
    materialDataBase = Teuchos::rcp(new QCAD::MaterialDatabase(filename, epetraComm));
  }

  Teuchos::RCP<Albany::STKDiscretization> stkDisc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(stkDisc.is_null(), "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for STK meshes.\n\n");
  const stk::mesh::fem::FEMMetaData& metaData = stkDisc->getSTKMetaData();
  const stk::mesh::BulkData& bulkData = stkDisc->getSTKBulkData();
  TEUCHOS_TEST_FOR_EXCEPT_MSG(metaData.spatial_dimension() != 3, "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for three-dimensional meshes.\n\n");

  // Store the cell topology for each element mesh part
  std::map<std::string,CellTopologyData> partCellTopologyData; 

  const stk::mesh::PartVector& stkParts = metaData.get_parts();
  stk::mesh::PartVector stkElementBlocks;
  for(stk::mesh::PartVector::const_iterator it = stkParts.begin(); it != stkParts.end(); ++it){
    stk::mesh::Part* const part = *it;
    if(part->name()[0] == '{')
      continue;
    if(part->primary_entity_rank() == metaData.element_rank()){
      stkElementBlocks.push_back(part);
      partCellTopologyData[part->name()] = *metaData.get_cell_topology(*part).getCellTopologyData();
    }
  }

  stk::mesh::Field<double, stk::mesh::Cartesian>* coordinatesField = 
    metaData.get_field< stk::mesh::Field<double, stk::mesh::Cartesian> >("coordinates");

  stk::mesh::Field<double, stk::mesh::Cartesian>* volumeField = 
    metaData.get_field< stk::mesh::Field<double, stk::mesh::Cartesian> >("volume");

  // Create a selector to select everything in the universal part that is either locally owned or globally shared
  stk::mesh::Selector selector = 
    stk::mesh::Selector( metaData.universal_part() ) & ( stk::mesh::Selector( metaData.locally_owned_part() ) | stk::mesh::Selector( metaData.globally_shared_part() ) );

  // Select element mesh entities that match the selector
  std::vector<stk::mesh::Entity*> elements;
  stk::mesh::get_selected_entities(selector, bulkData.buckets(metaData.element_rank()), elements);

  // Select node mesh entities that match the selector
  std::vector<stk::mesh::Entity*> nodes;
  stk::mesh::get_selected_entities(selector, bulkData.buckets(metaData.node_rank()), nodes);

  // List of blocks for peridynamics, partial stress, and standard FEM
  std::vector<std::string> peridynamicsBlocks, peridynamicPartialStressBlocks, classicalContinuumMechanicsBlocks;

  // List of global ids for the Peridigm maps
  vector<int> globalIds;
  int numPartialStressIds(0);

  // Bookkeeping so that partial stress nodes on the Peridigm side are guaranteed to have ids that don't exist in the Albany discretization
  int maxAlbanyElementId(0), maxAlbanyNodeId(0), minPeridigmPartialStressId(0), peridigmPartialStressId(0);

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
      stk::mesh::Selector( *stkElementBlocks[iBlock] ) & stk::mesh::Selector( metaData.locally_owned_part() );

    // Select the mesh entities that match the selector
    std::vector<stk::mesh::Entity*> elementsInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData.buckets(metaData.element_rank()), elementsInElementBlock);

    // Determine the material model assigned to this block
    std::string materialModelName;
    if(!materialDataBase.is_null())
      materialModelName = materialDataBase->getElementBlockSublist(blockName, "Material Model").get<std::string>("Model Name");

    // Sphere elements with the "Peridynamics" material model
    if(materialModelName == "Peridynamics"){
      peridynamicsBlocks.push_back(blockName);
      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	stk::mesh::PairIterRelation nodeRelations = elementsInElementBlock[iElement]->node_relations();
	int globalId = nodeRelations.begin()->entity()->identifier() - 1;
	globalIds.push_back(globalId);
      }
    }
    // Standard solid elements with the "Peridynamic Partial Stress" material model
    else if(materialModelName == "Peridynamic Partial Stress"){
      peridynamicPartialStressBlocks.push_back(blockName);
      CellTopologyData& cellTopologyData = partCellTopologyData[blockName];
      shards::CellTopology cellTopology(&cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(cellTopology, cubatureDegree);
      const int numQPts = cubature->getNumPoints();
      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	numPartialStressIds += numQPts;
      }
    }
    // Standard solid elements with a classical continum mechanics model
    else{
      classicalContinuumMechanicsBlocks.push_back(blockName);
    }

    // Track the max element and node id in the Albany discretization
    for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
      int elementId = elementsInElementBlock[iElement]->identifier() - 1;
      if(elementId > maxAlbanyElementId)
	maxAlbanyElementId = elementId;
      stk::mesh::PairIterRelation nodeRelations = elementsInElementBlock[iElement]->node_relations();
      for(stk::mesh::PairIterRelation::iterator it = nodeRelations.begin() ; it != nodeRelations.end() ; it++){
	int nodeId = it->entity()->identifier() - 1;
	if(nodeId > maxAlbanyNodeId)
	  maxAlbanyNodeId = nodeId;
      }
    }
  }

  minPeridigmPartialStressId = maxAlbanyElementId + 1;
  if(maxAlbanyNodeId > minPeridigmPartialStressId)
    minPeridigmPartialStressId = maxAlbanyNodeId + 1;

  vector<int> localVal(1), globalVal(1);
  localVal[0] = minPeridigmPartialStressId;
  epetraComm->MaxAll(&localVal[0], &globalVal[0], 1);
  minPeridigmPartialStressId = globalVal[0];

  for(int i=0 ; i<numPartialStressIds ; i++)
    globalIds.push_back(minPeridigmPartialStressId + i);

  peridigmPartialStressId = minPeridigmPartialStressId;

  // Write block information to stdout
  std::cout << "\n---- PeridigmManager ----";
  std::cout << "\n  peridynamics blocks:";
  for(unsigned int i=0 ; i<peridynamicsBlocks.size() ; ++i)
    std::cout << " " << peridynamicsBlocks[i];
  std::cout << "\n  peridynamic partial stres blocks:";
  for(unsigned int i=0 ; i<peridynamicPartialStressBlocks.size() ; ++i)
    std::cout << " " << peridynamicPartialStressBlocks[i];
  std::cout << "\n  classical continuum mechanics blocks:";
  for(unsigned int i=0 ; i<classicalContinuumMechanicsBlocks.size() ; ++i)
    std::cout << " " << classicalContinuumMechanicsBlocks[i];
  std::cout << "\n  max Albany element id: " << maxAlbanyElementId << std::endl;
  std::cout << "  max Albany node id: " << maxAlbanyNodeId << std::endl;
  std::cout << "  min Peridigm partial stress id: " << minPeridigmPartialStressId << std::endl;
  std::cout << "  number of Peridigm partial stress material points: " << numPartialStressIds << std::endl;
  std::cout << "\n---- PeridigmManager ----\n";

  // Bail if there are no sphere elements
  if(peridynamicsBlocks.size() == 0 && peridynamicPartialStressBlocks.size() == 0){
    hasPeridynamics = false;
    return;
  }
  else{
    hasPeridynamics = true;
  }

  // Create the owned maps
  Epetra_BlockMap oneDimensionalMap(-1, globalIds.size(), &globalIds[0], 1, 0, *epetraComm);
  Epetra_BlockMap threeDimensionalMap(-1, globalIds.size(), &globalIds[0], 3, 0, *epetraComm);

  // Create Epetra_Vectors for the initial positions, volumes, and block_ids
  Teuchos::RCP<Epetra_Vector> initialX = Teuchos::rcp(new Epetra_Vector(threeDimensionalMap));
  Teuchos::RCP<Epetra_Vector> cellVolume = Teuchos::rcp(new Epetra_Vector(oneDimensionalMap));
  Teuchos::RCP<Epetra_Vector> blockId = Teuchos::rcp(new Epetra_Vector(oneDimensionalMap));

  // loop over the element blocks and store the initial positions, volume, and block_id
  for(unsigned int iBlock=0 ; iBlock<stkElementBlocks.size() ; iBlock++){

    const std::string blockName = stkElementBlocks[iBlock]->name();
    int bId = blockNameToBlockId[blockName];

    // Create a selector for all locally-owned elements in the block
    stk::mesh::Selector selector = 
      stk::mesh::Selector( *stkElementBlocks[iBlock] ) & stk::mesh::Selector( metaData.locally_owned_part() );

    // Select the mesh entities that match the selector
    std::vector<stk::mesh::Entity*> elementsInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData.buckets(metaData.element_rank()), elementsInElementBlock);

    // Determine the material model assigned to this block
    std::string materialModelName;
    if(!materialDataBase.is_null())
      materialModelName = materialDataBase->getElementBlockSublist(blockName, "Material Model").get<std::string>("Model Name");

    if(materialModelName == "Peridynamics"){
      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	stk::mesh::PairIterRelation nodeRelations = elementsInElementBlock[iElement]->node_relations();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(nodeRelations.size() != 1, "\n\n**** Error in PeridigmManager::initialize(), \"Peridynamics\" material model may be assigned only to sphere elements.\n\n");
	stk::mesh::Entity* node = nodeRelations.begin()->entity();
	int globalId = node->identifier() - 1;
	int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(oneDimensionalMapLocalId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	int threeDimensionalMapLocalId = threeDimensionalMap.LID(globalId);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(threeDimensionalMapLocalId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	(*blockId)[oneDimensionalMapLocalId] = bId;
	double* exodusVolume = stk::mesh::field_data(*volumeField, *elementsInElementBlock[iElement]);
	(*cellVolume)[oneDimensionalMapLocalId] = exodusVolume[0];
	double* exodusCoordinates = stk::mesh::field_data(*coordinatesField, *node);
	(*initialX)[threeDimensionalMapLocalId*3]   = exodusCoordinates[0];
	(*initialX)[threeDimensionalMapLocalId*3+1] = exodusCoordinates[1];
	(*initialX)[threeDimensionalMapLocalId*3+2] = exodusCoordinates[2];
	sphereElementGlobalNodeIds.push_back(globalId);
      }
    }

    else if(materialModelName == "Peridynamic Partial Stress"){

      CellTopologyData& cellTopologyData = partCellTopologyData[blockName];
      shards::CellTopology cellTopology(&cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(cellTopology, cubatureDegree);
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

      // \todo I think these call all just be Intrepid::FieldContainer<RealType> and that what's below is overkill

      // Physical points, which are the physical (x, y, z) values of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim> > physPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> physPoints("Physical Points", physPointsLayout);
      Teuchos::ArrayRCP<RealType> physPointsMem = Teuchos::arcp<RealType>(physPointsLayout->size());
      physPoints.setFieldData(physPointsMem);

      // Reference points, which are the natural coordinates of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim> > refPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> refPoints("Reference Points", refPointsLayout);
      Teuchos::ArrayRCP<RealType> refPointsMem = Teuchos::arcp<RealType>(refPointsLayout->size());
      refPoints.setFieldData(refPointsMem);

      // Cell workset, which is the set of nodes for the given element
      Teuchos::RCP< PHX::MDALayout<Cell, Node, Dim> > cellWorksetLayout = Teuchos::rcp(new PHX::MDALayout<Cell, Node, Dim>(numCells, numNodes, numDim));
      PHX::MDField<RealType, Cell, Node, Dim> cellWorkset("Cell Workset", cellWorksetLayout);
      Teuchos::ArrayRCP<RealType> cellWorksetMem = Teuchos::arcp<RealType>(cellWorksetLayout->size());
      cellWorkset.setFieldData(cellWorksetMem);

      // Copy the reference points from the Intrepid::FieldContainer to a PHX::MDField
      for(int qp=0 ; qp<numQuadPoints ; ++qp){
	for(int dof=0 ; dof<3 ; ++dof){
	  refPoints(0, qp, dof) = quadratureRefPoints(qp, dof);
	}
      }

      for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
	stk::mesh::PairIterRelation nodeRelations = elementsInElementBlock[iElement]->node_relations();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(nodeRelations.size() != numNodes, "\n\n**** Error in PeridigmManager::initialize(), nodeRelations.size() != numNodes.\n\n");
	int index = 0;
	for(stk::mesh::PairIterRelation::iterator it = nodeRelations.begin() ; it != nodeRelations.end() ; it++){
	  stk::mesh::Entity* node = it->entity();
	  double* coordinates = stk::mesh::field_data(*coordinatesField, *node);
	  for(int dof=0 ; dof<3 ; ++dof)
	    cellWorkset(0, index, dof) = coordinates[dof];
	  index += 1;
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

	for(unsigned int i=0 ; i<nodeRelations.size() ; ++i){
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 0) );
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 1) );
	  partialStressElement.albanyNodeInitialPositions.push_back( cellWorkset(0, i, 2) );
	}

	for(unsigned int qp=0 ; qp<numQuadPoints ; ++qp){
	  int globalId = peridigmPartialStressId++;
	  int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
	  TEUCHOS_TEST_FOR_EXCEPT_MSG(oneDimensionalMapLocalId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	  int threeDimensionalMapLocalId = threeDimensionalMap.LID(globalId);
	  TEUCHOS_TEST_FOR_EXCEPT_MSG(threeDimensionalMapLocalId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	  (*blockId)[oneDimensionalMapLocalId] = bId;
	  (*cellVolume)[oneDimensionalMapLocalId] = weightedMeasures(0, qp);
	  (*initialX)[threeDimensionalMapLocalId*3]   = physPoints(0, qp, 0);
	  (*initialX)[threeDimensionalMapLocalId*3+1] = physPoints(0, qp, 1);
	  (*initialX)[threeDimensionalMapLocalId*3+2] = physPoints(0, qp, 2);

	  partialStressElement.peridigmGlobalIds.push_back(globalId);
	}

	partialStressElements.push_back(partialStressElement);
      }
    }
  }

  // Create a vector for storing the previous solution (from last converged load step)
  previousSolutionPositions = Teuchos::RCP<Epetra_Vector>(new Epetra_Vector(threeDimensionalMap));
  *previousSolutionPositions = *initialX;

  // Create a Peridigm discretization
  peridynamicDiscretization = Teuchos::rcp<PeridigmNS::Discretization>(new PeridigmNS::AlbanyDiscretization(epetraComm,
                                                                                                            peridigmParams,
                                                                                                            initialX,
                                                                                                            cellVolume,
                                                                                                            blockId));

  // Create a Peridigm object
  peridigm = Teuchos::rcp<PeridigmNS::Peridigm>(new PeridigmNS::Peridigm(epetraComm, peridigmParams, peridynamicDiscretization));

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
    int albanyGlobalElementId = partialStressElements[i].albanyElement->identifier() - 1;
    vector<int>& peridigmGlobalIds = partialStressElements[i].peridigmGlobalIds;
    albanyPartialStressElementGlobalIdToPeridigmGlobalIds[albanyGlobalElementId] = peridigmGlobalIds;
  }

#endif
}

void LCM::PeridigmManager::setCurrentTimeAndDisplacement(double time, const Epetra_Vector& albanySolutionVector)
{
#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics){

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
    const Epetra_Vector& albanyCurrentDisplacements = albanySolutionVector;
    const Epetra_BlockMap& peridigmMap = peridigmCurrentPositions.Map();
    const Epetra_BlockMap& albanyMap = albanySolutionVector.Map();
    int peridigmLocalId, albanyLocalId, globalId;
    for(unsigned int i=0 ; i<sphereElementGlobalNodeIds.size() ; ++i){
      globalId = sphereElementGlobalNodeIds[i];
      peridigmLocalId = peridigmMap.LID(globalId);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Peridigm local id.\n\n");
      albanyLocalId = albanyMap.LID(3*globalId);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(albanyLocalId == -1, "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Albany local id.\n\n");
      peridigmDisplacements[3*peridigmLocalId]   = albanyCurrentDisplacements[albanyLocalId];
      peridigmDisplacements[3*peridigmLocalId+1] = albanyCurrentDisplacements[albanyLocalId+1];
      peridigmDisplacements[3*peridigmLocalId+2] = albanyCurrentDisplacements[albanyLocalId+2];
      peridigmCurrentPositions[3*peridigmLocalId]   = peridigmReferencePositions[3*peridigmLocalId] + peridigmDisplacements[3*peridigmLocalId];
      peridigmCurrentPositions[3*peridigmLocalId+1] = peridigmReferencePositions[3*peridigmLocalId+1] + peridigmDisplacements[3*peridigmLocalId+1];
      peridigmCurrentPositions[3*peridigmLocalId+2] = peridigmReferencePositions[3*peridigmLocalId+2] + peridigmDisplacements[3*peridigmLocalId+2];
      peridigmVelocities[3*peridigmLocalId]   = (peridigmCurrentPositions[3*peridigmLocalId]   - (*previousSolutionPositions)[3*peridigmLocalId])/timeStep;
      peridigmVelocities[3*peridigmLocalId+1] = (peridigmCurrentPositions[3*peridigmLocalId+1] - (*previousSolutionPositions)[3*peridigmLocalId+1])/timeStep;
      peridigmVelocities[3*peridigmLocalId+2] = (peridigmCurrentPositions[3*peridigmLocalId+2] - (*previousSolutionPositions)[3*peridigmLocalId+2])/timeStep;
    }

    // Partial stress elements (solid elements with peridynamic material points at each integration point)

    for(std::vector<PartialStressElement>::iterator it=partialStressElements.begin() ; it!=partialStressElements.end() ; it++){

      // \todo This is brutal, need to store these data structures instead of re-creating them every time for every element.
      // Can probably store things by block and use worksets to compute things in one big call.

      // Create data structures for passing information to/from Intrepid.

      shards::CellTopology cellTopology(&it->cellTopologyData);
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(cellTopology, cubatureDegree);
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

      // Physical points, which are the physical (x, y, z) values of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim> > physPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> physPoints("Physical Points", physPointsLayout);
      Teuchos::ArrayRCP<RealType> physPointsMem = Teuchos::arcp<RealType>(physPointsLayout->size());
      physPoints.setFieldData(physPointsMem);

      // Reference points, which are the natural coordinates of the quadrature points
      Teuchos::RCP< PHX::MDALayout<Cell, QuadPoint, Dim> > refPointsLayout = Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim>(numCells, numQuadPoints, numDim));
      PHX::MDField<RealType, Cell, QuadPoint, Dim> refPoints("Reference Points", refPointsLayout);
      Teuchos::ArrayRCP<RealType> refPointsMem = Teuchos::arcp<RealType>(refPointsLayout->size());
      refPoints.setFieldData(refPointsMem);

      // Cell workset, which is the set of nodes for the given element
      Teuchos::RCP< PHX::MDALayout<Cell, Node, Dim> > cellWorksetLayout = Teuchos::rcp(new PHX::MDALayout<Cell, Node, Dim>(numCells, numNodes, numDim));
      PHX::MDField<RealType, Cell, Node, Dim> cellWorkset("Cell Workset", cellWorksetLayout);
      Teuchos::ArrayRCP<RealType> cellWorksetMem = Teuchos::arcp<RealType>(cellWorksetLayout->size());
      cellWorkset.setFieldData(cellWorksetMem);

      // Copy the reference points from the Intrepid::FieldContainer to a PHX::MDField
      for(int qp=0 ; qp<numQuadPoints ; ++qp){
	for(int dof=0 ; dof<3 ; ++dof){
	  refPoints(0, qp, dof) = quadratureRefPoints(qp, dof);
	}
      }

      stk::mesh::PairIterRelation nodeRelations = it->albanyElement->node_relations();
      int index = 0;
      for(stk::mesh::PairIterRelation::iterator nodeIt = nodeRelations.begin() ; nodeIt != nodeRelations.end() ; nodeIt++){
	int globalAlbanyNodeId = nodeIt->entity()->identifier() - 1;
	int albanySolutionVectorIndex = albanyMap.LID(3*globalAlbanyNodeId);
	cellWorkset(0, index, 0) = it->albanyNodeInitialPositions[3*index]   + albanySolutionVector[albanySolutionVectorIndex];
	cellWorkset(0, index, 1) = it->albanyNodeInitialPositions[3*index+1] + albanySolutionVector[albanySolutionVectorIndex+1];
	cellWorkset(0, index, 2) = it->albanyNodeInitialPositions[3*index+2] + albanySolutionVector[albanySolutionVectorIndex+2];
	index += 1;
      }

      // Determine the global (x,y,z) coordinates of the quadrature points
      Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, cellTopology);

      for(unsigned int i=0 ; i<it->peridigmGlobalIds.size() ; ++i){
	peridigmLocalId = peridigmMap.LID(it->peridigmGlobalIds[i]);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::setCurrentTimeAndDisplacement(), invalid Peridigm local id.\n\n");
	peridigmCurrentPositions[3*peridigmLocalId]   = physPoints(0, i, 0);
	peridigmCurrentPositions[3*peridigmLocalId+1] = physPoints(0, i, 1);
	peridigmCurrentPositions[3*peridigmLocalId+2] = physPoints(0, i, 2);
	peridigmDisplacements[3*peridigmLocalId]   = peridigmCurrentPositions[3*peridigmLocalId]   - peridigmReferencePositions[3*peridigmLocalId];
	peridigmDisplacements[3*peridigmLocalId+1] = peridigmCurrentPositions[3*peridigmLocalId+1] - peridigmReferencePositions[3*peridigmLocalId+1];
	peridigmDisplacements[3*peridigmLocalId+2] = peridigmCurrentPositions[3*peridigmLocalId+2] - peridigmReferencePositions[3*peridigmLocalId+2];
	peridigmVelocities[3*peridigmLocalId]   = (peridigmCurrentPositions[3*peridigmLocalId]   - (*previousSolutionPositions)[3*peridigmLocalId])/timeStep;
	peridigmVelocities[3*peridigmLocalId+1] = (peridigmCurrentPositions[3*peridigmLocalId+1] - (*previousSolutionPositions)[3*peridigmLocalId+1])/timeStep;
	peridigmVelocities[3*peridigmLocalId+2] = (peridigmCurrentPositions[3*peridigmLocalId+2] - (*previousSolutionPositions)[3*peridigmLocalId+2])/timeStep;
      }
    }
  }

#endif
}

void LCM::PeridigmManager::updateState()
{
#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics){
    previousTime = currentTime;
    *previousSolutionPositions = *(peridigm->getY());
    peridigm->updateState();
  }

#endif
}

void LCM::PeridigmManager::writePeridigmSubModel(RealType currentTime)
{
#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics)
    peridigm->writePeridigmSubModel(currentTime);

#endif
}

void LCM::PeridigmManager::evaluateInternalForce()
{
#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics)
    peridigm->computeInternalForce();

#endif
}

double LCM::PeridigmManager::getForce(int globalId, int dof)
{
  double force(0.0);

#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics){
    Epetra_Vector& peridigmForce = *(peridigm->getForce());
    int peridigmLocalId = peridigmForce.Map().LID(globalId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::getForce(), invalid global id.\n\n");
    force = peridigmForce[3*peridigmLocalId + dof];

    // DEBUGGING ?? NEED MAP FROM ALBANY ELEMENT ID TO PERIDIGM NODE ID (NOT THE SAME UNLESS THE SIMULATION IS 100% PERIDYNAMICS)
  }

#endif

  return force;
}

void LCM::PeridigmManager::getPartialStress(std::string blockName, int worksetIndex, int worksetLocalElementId, std::vector< std::vector<RealType> >& partialStressValues)
{
#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics){

    int globalElementId = worksetLocalIdToGlobalId[worksetIndex][worksetLocalElementId];
    std::vector<int>& peridigmGlobalIds = albanyPartialStressElementGlobalIdToPeridigmGlobalIds[globalElementId];
    Teuchos::RCP<const Epetra_Vector> data = peridigm->getBlockData(blockName, "Partial_Stress");
    for(unsigned int i=0 ; i<peridigmGlobalIds.size() ; ++i){
      int peridigmLocalId = data->Map().LID(peridigmGlobalIds[i]);
      TEUCHOS_TEST_FOR_EXCEPT_MSG(peridigmLocalId == -1, "\n\n**** Error in PeridigmManager::getPartialStress(), invalid global id.\n");
      for(int j=0 ; j<9 ; ++j){
	partialStressValues[i][j] = (*data)[9*peridigmLocalId + j];
      }
    }
  }

#endif
}

Teuchos::RCP<const Epetra_Vector> LCM::PeridigmManager::getBlockData(std::string blockName, std::string fieldName)
{

  Teuchos::RCP<const Epetra_Vector> data;

#ifdef ALBANY_PERIDIGM

  if(hasPeridynamics)
    data = peridigm->getBlockData(blockName, fieldName);

#endif

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

std::vector<LCM::PeridigmManager::OutputField> LCM::PeridigmManager::getOutputFields()
{
  return outputFields;
}
