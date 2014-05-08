/*! \file PeridigmManager.cpp */

#include "PeridigmManager.hpp"
#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include "Phalanx_DataLayout.hpp"

LCM::PeridigmManager& LCM::PeridigmManager::self() {
  static PeridigmManager peridigmManager;
  return peridigmManager;
}

LCM::PeridigmManager::PeridigmManager() : hasPeridynamics(false), previousTime(0.0), currentTime(0.0), timeStep(0.0)
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

  Teuchos::RCP<Albany::STKDiscretization> stkDisc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(stkDisc.is_null(), "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for STK meshes.\n\n");
  const stk::mesh::fem::FEMMetaData& metaData = stkDisc->getSTKMetaData();
  const stk::mesh::BulkData& bulkData = stkDisc->getSTKBulkData();
  TEUCHOS_TEST_FOR_EXCEPT_MSG(metaData.spatial_dimension() != 3, "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface is valid only for three-dimensional meshes.\n\n");

  const stk::mesh::PartVector& stkParts = metaData.get_parts();
  stk::mesh::PartVector stkElementBlocks;
  for(stk::mesh::PartVector::const_iterator it = stkParts.begin(); it != stkParts.end(); ++it){
    stk::mesh::Part* const part = *it;
    if(part->name()[0] == '{')
      continue;
    if(part->primary_entity_rank() == metaData.element_rank())
      stkElementBlocks.push_back(part);
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

  // Create a list of owned global element ids for sphere elements
  vector<int> globalIds;
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    stk::mesh::PairIterRelation nodeRelations = elements[iElem]->node_relations();
    // Process only sphere elements
    if(nodeRelations.size() == 1){
      int globalId = nodeRelations.begin()->entity()->identifier() - 1;
      globalIds.push_back(globalId);
    }
  }

  // Bail if there are no sphere elements
  if(globalIds.size() == 0){
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

  // loop over the elements and store the volume and initial coordinates
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    stk::mesh::PairIterRelation nodeRelations = elements[iElem]->node_relations();
    // Process only sphere elements
    if(nodeRelations.size() == 1){
      stk::mesh::Entity* node = nodeRelations.begin()->entity();
      int globalId = node->identifier() - 1;
      int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
      int threeDimensionalMapLocalId = threeDimensionalMap.LID(globalId);
      double* exodusVolume = stk::mesh::field_data(*volumeField, *elements[iElem]);
      (*cellVolume)[oneDimensionalMapLocalId] = exodusVolume[0];
      double* exodusCoordinates = stk::mesh::field_data(*coordinatesField, *node);
      (*initialX)[threeDimensionalMapLocalId*3]   = exodusCoordinates[0];
      (*initialX)[threeDimensionalMapLocalId*3+1] = exodusCoordinates[1];
      (*initialX)[threeDimensionalMapLocalId*3+2] = exodusCoordinates[2];
    }
  }

  // Create a vector for storing the previous solution (from last converged load step)
  previousSolutionPositions = Teuchos::RCP<Epetra_Vector>(new Epetra_Vector(threeDimensionalMap));

  // loop over the element blocks and record the block id for each sphere element
  for(unsigned int iBlock=0 ; iBlock<stkElementBlocks.size() ; iBlock++){

    // determine the block id under the assumption that the block names follow the format "block_1", "block_2", etc.
    // older versions of stk did not have the ability to return the block id directly, I think newer versions of stk can do this however
    const std::string blockName = stkElementBlocks[iBlock]->name();
    size_t loc = blockName.find_last_of('_');
    TEUCHOS_TEST_FOR_EXCEPT_MSG(loc == string::npos, "\n**** Parse error in PeridigmManager::initialize(), invalid block name: " + blockName + "\n");
    stringstream blockIDSS(blockName.substr(loc+1, blockName.size()));
    int bId;
    blockIDSS >> bId;

    // Create a selector for all locally-owned elements in the block
    stk::mesh::Selector selector = 
      stk::mesh::Selector( *stkElementBlocks[iBlock] ) & stk::mesh::Selector( metaData.locally_owned_part() );

    // Select the mesh entities that match the selector
    std::vector<stk::mesh::Entity*> elementsInElementBlock;
    stk::mesh::get_selected_entities(selector, bulkData.buckets(metaData.element_rank()), elementsInElementBlock);

    // Loop over the elements in this block
    for(unsigned int iElement=0 ; iElement<elementsInElementBlock.size() ; iElement++){
      stk::mesh::PairIterRelation nodeRelations = elementsInElementBlock[iElement]->node_relations();
      if(nodeRelations.size() == 1){
	int globalId = nodeRelations.begin()->entity()->identifier() - 1;
	int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
	TEUCHOS_TEST_FOR_EXCEPT_MSG(oneDimensionalMapLocalId == -1, "\n\n**** Error in PeridigmManager::initialize(), invalid global id.\n\n");
	(*blockId)[oneDimensionalMapLocalId] = bId;
      }
    }
  }

  // Create a Peridigm discretization
  peridynamicDiscretization = Teuchos::rcp<PeridigmNS::Discretization>(new PeridigmNS::AlbanyDiscretization(epetraComm,
                                                                                                            peridigmParams,
                                                                                                            initialX,
                                                                                                            cellVolume,
                                                                                                            blockId));

  // Create a Peridigm object
  peridigm = Teuchos::rcp<PeridigmNS::Peridigm>(new PeridigmNS::Peridigm(epetraComm, peridigmParams, peridynamicDiscretization));

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
    const Epetra_Vector& albanyCurrentDisplacements = albanySolutionVector;
    const Epetra_BlockMap& peridigmMap = peridigmCurrentPositions.Map();
    const Epetra_BlockMap& albanyMap = albanySolutionVector.Map();
    int peridigmLocalId, albanyLocalId, globalId;
    for(peridigmLocalId = 0 ; peridigmLocalId < peridigmMap.NumMyElements() ; peridigmLocalId++){
      globalId = peridigmMap.GID(peridigmLocalId);
      albanyLocalId = albanyMap.LID(3*globalId);
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
  }

#endif

  return force;
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

    if(name == "Dilatation" || name == "Weighted_Volume" || name == "Radius"){
      field.relation = "element";
      field.lengthName = "scalar";
      field.length = 1;
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
