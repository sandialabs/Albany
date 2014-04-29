/*! \file PeridigmManager.cpp */

#include "PeridigmManager.hpp"
#include "Albany_Utils.hpp"
//#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldData.hpp>

LCM::PeridigmManager& LCM::PeridigmManager::self() {
  static PeridigmManager peridigmManager;
  return peridigmManager;
}

LCM::PeridigmManager::PeridigmManager() : peridigmIsInitialized(false)
{
  epetraComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
}

bool LCM::PeridigmManager::isInitialized()
{
  return peridigmIsInitialized;
}

void LCM::PeridigmManager::addGlobalElementIds(const std::vector<int>& ids)
{
  myGlobalElements.insert(myGlobalElements.end(), ids.begin(), ids.end());
}

void LCM::PeridigmManager::initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
				      Teuchos::RCP<Albany::AbstractDiscretization> disc)
{
#ifndef ALBANY_PERIDIGM

  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "\n\n**** Error:  Peridigm interface not available!  Recompile with -DUSE_PERIDIGM.\n\n");

#else

  peridigmParams = Teuchos::rcp<Teuchos::ParameterList>(new Teuchos::ParameterList(params->sublist("Problem").sublist("Peridigm Parameters", true)));

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
  TEUCHOS_TEST_FOR_EXCEPT_MSG(volumeField == 0, "\n\n**** Error in PeridigmManager::initialize():  Peridigm interface requires STK sphere mesh with volume field.\n\n");

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
  vector<int> globalElementIds;
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    stk::mesh::PairIterRelation nodeRelations = elements[iElem]->node_relations();
    // Process only sphere elements
    if(nodeRelations.size() == 1){
      int elementId = elements[iElem]->identifier() - 1;
      globalElementIds.push_back(elementId);
    }
  }

  // Create the owned maps
  Epetra_BlockMap oneDimensionalMap(-1, globalElementIds.size(), &globalElementIds[0], 1, 0, *epetraComm);
  Epetra_BlockMap threeDimensionalMap(-1, globalElementIds.size(), &globalElementIds[0], 3, 0, *epetraComm);

  // Create Epetra_Vectors for the initial positions, volumes, and block_ids
  Teuchos::RCP<Epetra_Vector> initialX = Teuchos::rcp(new Epetra_Vector(threeDimensionalMap));
  Teuchos::RCP<Epetra_Vector> cellVolume = Teuchos::rcp(new Epetra_Vector(oneDimensionalMap));
  Teuchos::RCP<Epetra_Vector> blockId = Teuchos::rcp(new Epetra_Vector(oneDimensionalMap));

  // loop over the elements and store the volume initial coordinates
  for(unsigned int iElem=0 ; iElem<elements.size() ; ++iElem){
    stk::mesh::PairIterRelation nodeRelations = elements[iElem]->node_relations();
    // Process only sphere elements
    if(nodeRelations.size() == 1){
      int globalId = elements[iElem]->identifier() - 1;
      int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
      int threeDimensionalMapLocalId = threeDimensionalMap.LID(globalId);
      double* exodusVolume = stk::mesh::field_data(*volumeField, *elements[iElem]);
      (*cellVolume)[oneDimensionalMapLocalId] = exodusVolume[0];
      stk::mesh::Entity* node = nodeRelations.begin()->entity();
      double* exodusCoordinates = stk::mesh::field_data(*coordinatesField, *node);
      (*initialX)[threeDimensionalMapLocalId]   = exodusCoordinates[0];
      (*initialX)[threeDimensionalMapLocalId+1] = exodusCoordinates[1];
      (*initialX)[threeDimensionalMapLocalId+2] = exodusCoordinates[2];
    }
  }

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
      int globalId = elementsInElementBlock[iElement]->identifier() - 1;
      int oneDimensionalMapLocalId = oneDimensionalMap.LID(globalId);
      if(oneDimensionalMapLocalId != -1)
	(*blockId)[oneDimensionalMapLocalId] = bId;
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

  peridigmIsInitialized = true;

#endif
}

void LCM::PeridigmManager::setDisplacements(const Epetra_Vector& x)
{
#ifdef ALBANY_PERIDIGM

  Epetra_Vector& peridigmCurrentPositions = *(peridigm->getY());
  const Epetra_Vector& albanyCurrentPositions = x;
  const Epetra_BlockMap& peridigmMap = peridigmCurrentPositions.Map();
  const Epetra_BlockMap& albanyMap = x.Map();
  int peridigmLocalId, albanyLocalId, globalId;
  for(peridigmLocalId = 0 ; peridigmLocalId < peridigmMap.NumMyElements() ; peridigmLocalId++){
    globalId = peridigmMap.GID(peridigmLocalId);
    albanyLocalId = albanyMap.LID(3*globalId);
    peridigmCurrentPositions[3*peridigmLocalId]   = albanyCurrentPositions[albanyLocalId];
    peridigmCurrentPositions[3*peridigmLocalId+1] = albanyCurrentPositions[albanyLocalId+1];
    peridigmCurrentPositions[3*peridigmLocalId+2] = albanyCurrentPositions[albanyLocalId+2];
  }

//   Teuchos::RCP<Epetra_Vector> peridigmInitialPosition = this->peridigm->getX();
//   Teuchos::RCP<Epetra_Vector> peridigmCurrentPosition = this->peridigm->getY();
//   Teuchos::RCP<Epetra_Vector> peridigmDisplacement = this->peridigm->getU();
//   Teuchos::RCP<Epetra_Vector> peridigmVelocity = this->peridigm->getV();
//   Teuchos::RCP<Epetra_Vector> peridigmForce = this->peridigm->getForce();

  std::cout << "\nDEBUG confirm element sizes peridigmCurrentPosition.Map().ElementSize() " << peridigmCurrentPositions.Map().ElementSize() << ", albanyCurrentPosition.Map().ElementSize() " << albanyCurrentPositions.Map().ElementSize() << "\n" << endl;

#endif
}

void LCM::PeridigmManager::evaluateInternalForce()
{
#ifdef ALBANY_PERIDIGM

  this->peridigm->updateState();

#endif
}

double LCM::PeridigmManager::getForce(int globalId, int dof)
{
  double force(0.0);

#ifdef ALBANY_PERIDIGM

  Epetra_Vector& peridigmForce = *(peridigm->getForce());
  int peridigmLocalId = peridigmForce.Map().LID(globalId);
  force = peridigmForce[3*peridigmLocalId + dof];

#endif

  return force;
}
