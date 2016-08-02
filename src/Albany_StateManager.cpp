//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out!
//No Epetra if ALBANY_EPETRA_EXE turned off.

#include "Albany_StateManager.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_TestForException.hpp"

Albany::StateManager::StateManager() :
  stateVarsAreAllocated (false),
  stateInfo             (Teuchos::rcp(new StateInfoStruct))
{
  // Nothing to be done here
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const std::string& ebName,
                                            const std::string &init_type,
                                            const double init_val,
                                            const bool registerOldState)
{
  return registerStateVariable(name, dl, dummy, ebName, init_type, init_val, registerOldState, name);
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const std::string& ebName,
                                            const std::string &init_type,
                                            const double init_val,
                                            const bool registerOldState,
                                            const std::string& fieldName)
{
  const bool bOutputToExodus = true;
  registerStateVariable(stateName, dl, ebName, init_type, init_val, registerOldState, bOutputToExodus, fieldName);

  // Create param list for SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save or Load State "
                + stateName + " to/from field " + fieldName));
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", fieldName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("State Field Layout", dl);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout", dummy);
  return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const std::string& ebName,
                                            const std::string &init_type,
                                            const double init_val,
                                            const bool registerOldState,
                                            const bool outputToExodus)
{
  registerStateVariable(stateName, dl, ebName, init_type, init_val, registerOldState, outputToExodus, stateName);

  // Create param list for SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save or Load State "
                + stateName + " to/from field " + stateName));
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", stateName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("State Field Layout", dl);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout", dummy);
  return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerNodalVectorStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const std::string& ebName,
                                            const std::string &init_type,
                                            const double init_val,
                                            const bool registerOldState,
                                            const bool outputToExodus)
{
  registerNodalVectorStateVariable(stateName, dl, ebName, init_type, init_val, registerOldState, outputToExodus, stateName);

  // Create param list for SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save or Load State "
                + stateName + " to/from field " + stateName));
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", stateName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("State Field Layout", dl);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout", dummy);
  return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const std::string& ebName,
                                            const bool outputToExodus,
                                            StateStruct::MeshFieldEntity const* fieldEntity,
                                            const std::string& meshPartName)
{
  registerStateVariable(stateName, dl, ebName, "", 0., false, outputToExodus, "",fieldEntity, meshPartName);

  // Create param list for SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save or Load State "
                + stateName + " to/from field " + stateName));
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", stateName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("State Field Layout", dl);
  return p;
}

void
Albany::StateManager::registerStateVariable(const std::string &stateName,
              const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const std::string &init_type){

  // Grab the ebName
  std::string ebName;
  Albany::StateInfoStruct::const_iterator st = stateInfo->begin();
  ebName = (*st)->nameMap[stateName];

  // Call the below function
  registerStateVariable(stateName, dl, ebName, init_type, 0.0, false, true, "");

}

void
Albany::StateManager::registerStateVariable(const std::string &stateName,
              const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const std::string& ebName,
                                            const std::string &init_type,
                                            const double init_val,
                                            const bool registerOldState,
              const bool outputToExodus,
              const std::string &responseIDtoRequire,
              StateStruct::MeshFieldEntity const* fieldEntity,
              const std::string& meshPartName)
{
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);
  using Albany::StateStruct;

  if( statesToStore[ebName].find(stateName) != statesToStore[ebName].end() ) {
    //Duplicate registration.  This will occur when a problem's
    // constructEvaluators function (templated) registers state variables.

    //Perform a check here that dl and statesToStore[stateName] are the same:
    //TEUCHOS_TEST_FOR_EXCEPT(dl != statesToStore[stateName]);  //I don't know how to do this correctly (erik)
//    TEUCHOS_TEST_FOR_EXCEPT(!(*dl == *statesToStore[stateName]));
    return;  // Don't re-register the same state name
  }

  statesToStore[ebName][stateName] = dl;

  // Load into StateInfo
  StateStruct::MeshFieldEntity mfe_type;
  if(fieldEntity) mfe_type=*fieldEntity;
  else if(dl->rank() == 1 && dl->size() == 1)
     mfe_type = StateStruct::WorksetValue; // One value for the whole workset (i.e., time)
  else if (dl->rank() == 1 && dl->name(0) == "Cell")
     mfe_type = StateStruct::ElemData;
  else if(dl->rank() >= 1 && dl->name(0) == "Node") // Nodal data
     mfe_type = StateStruct::NodalData;
  else if(dl->rank() >= 1 && dl->name(0) == "Cell"){ // Element QP or node data
     if(dl->rank() > 1 && dl->name(1) == "Node") // Element node data
        mfe_type = StateStruct::ElemNode; // One value for the whole workset (i.e., time)
     else if(dl->rank() > 1 && dl->name(1) == "QuadPoint") // Element node data
        mfe_type = StateStruct::QuadPoint; // One value for the whole workset (i.e., time)
     else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
       "StateManager: Element Entity type - " << dl->name(1) << " - not supported" << std::endl);
  }
  else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
     "StateManager: Unknown Entity type - " << dl->name(0) << " - not supported" << std::endl);

  (*stateInfo).push_back(Teuchos::rcp(new StateStruct(stateName, mfe_type)));
  StateStruct& stateRef = *stateInfo->back();
  stateRef.setInitType(init_type);
  stateRef.setInitValue(init_val);
  stateRef.setMeshPart(meshPartName);

  dl->dimensions(stateRef.dim);

  if (stateRef.entity == StateStruct::NodalData)
  {
    // Register the state with the nodalDataVector also.
    Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getNodalDataBase();

    if (dl->rank() == 1) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 2) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]);
    else if (dl->rank() == 3) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]*stateRef.dim[2]);
  }
  else if (stateRef.entity == StateStruct::NodalDataToElemNode ||
           stateRef.entity == Albany::StateStruct::NodalDistParameter)
  {
    // Register the state with the nodalDataVector also.
    Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getNodalDataBase();

    // These entities store data as <Cell,Node,...>, so the dl has one more rank
    if (dl->rank() == 2) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 3) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]);
    else if (dl->rank() == 4) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]*stateRef.dim[2]);
  }

  stateRef.output = outputToExodus;
  stateRef.responseIDtoRequire = responseIDtoRequire;

  // If space is needed for old state
  if (registerOldState) {
    stateRef.saveOldState = true;

    std::string stateName_old = stateName + "_old";
    (*stateInfo).push_back(Teuchos::rcp(new Albany::StateStruct(stateName_old, mfe_type)));
    Albany::StateStruct& pstateRef = *stateInfo->back();
    pstateRef.initType  = init_type;
    pstateRef.initValue = init_val;
    pstateRef.pParentStateStruct = &stateRef;

    pstateRef.output = false;
    dl->dimensions(pstateRef.dim);
  }

  // insert
  stateRef.nameMap[stateName] = ebName;

}


void
Albany::StateManager::registerNodalVectorStateVariable (const std::string &stateName,
                                                        const Teuchos::RCP<PHX::DataLayout> &dl,
                                                        const std::string& ebName,
                                                        const std::string &init_type,
                                                        const double init_val,
                                                        const bool registerOldState,
                                                        const bool outputToExodus,
                                                        const std::string &responseIDtoRequire)

{
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);
  using Albany::StateStruct;

  if( statesToStore[ebName].find(stateName) != statesToStore[ebName].end() ) {
    //Duplicate registration.  This will occur when a problem's
    // constructEvaluators function (templated) registers state variables.

    //Perform a check here that dl and statesToStore[stateName] are the same:
    //TEUCHOS_TEST_FOR_EXCEPT(dl != statesToStore[stateName]);  //I don't know how to do this correctly (erik)
//    TEUCHOS_TEST_FOR_EXCEPT(!(*dl == *statesToStore[stateName]));
    return;  // Don't re-register the same state name
  }

  statesToStore[ebName][stateName] = dl;

  // Load into StateInfo
  StateStruct::MeshFieldEntity mfe_type;
  if(dl->rank() == 1 && dl->size() == 1)
     mfe_type = StateStruct::WorksetValue; // One value for the whole workset (i.e., time)
  else if(dl->rank() >= 1 && dl->name(0) == "Node") // Nodal data
     mfe_type = StateStruct::NodalData;
  else if(dl->rank() >= 1 && dl->name(0) == "Cell"){ // Element QP or node data
     if(dl->rank() > 1 && dl->name(1) == "Node") // Element node data
        mfe_type = StateStruct::ElemNode; // One value for the whole workset (i.e., time)
     else if(dl->rank() > 1 && dl->name(1) == "QuadPoint") // Element node data
        mfe_type = StateStruct::QuadPoint; // One value for the whole workset (i.e., time)
     else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
       "StateManager: Element Entity type - " << dl->name(1) << " - not supported" << std::endl);
  }
  else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
     "StateManager: Unknown Entity type - " << dl->name(0) << " - not supported" << std::endl);

  (*stateInfo).push_back(Teuchos::rcp(new StateStruct(stateName, mfe_type)));
  StateStruct& stateRef = *stateInfo->back();
  stateRef.setInitType(init_type);
  stateRef.setInitValue(init_val);

  dl->dimensions(stateRef.dim);

  Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getNodalDataBase();

  if (stateRef.entity == StateStruct::NodalData)
  {
    if (dl->rank() == 1) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 2) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]);
    else if (dl->rank() == 3) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]*stateRef.dim[2]);
  }
  else if (stateRef.entity == StateStruct::NodalDataToElemNode ||
           stateRef.entity == Albany::StateStruct::NodalDistParameter)
  {
    // These entities store data as <Cell,Node,...>, so the dl has one more rank
    if (dl->rank() == 2) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 3) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]);
    else if (dl->rank() == 4) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]*stateRef.dim[2]);
  }

  stateRef.output = outputToExodus;
  stateRef.responseIDtoRequire = responseIDtoRequire;

  // If space is needed for old state
  if (registerOldState) {
    stateRef.saveOldState = true;

    std::string stateName_old = stateName + "_old";
    (*stateInfo).push_back(Teuchos::rcp(new Albany::StateStruct(stateName_old, mfe_type)));
    Albany::StateStruct& pstateRef = *stateInfo->back();
    pstateRef.initType  = init_type;
    pstateRef.initValue = init_val;
    pstateRef.pParentStateStruct = &stateRef;

    pstateRef.output = false;
    dl->dimensions(pstateRef.dim);


    Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getNodalDataBase();

    if (stateRef.entity == StateStruct::NodalData)
    {
      // These entities store data as <Node,...>, so the dl rank and fieldDim are the same
      if (dl->rank() == 1) // node scalar
        nodalDataBase->registerVectorState (stateName_old, 1);
      else if (dl->rank() == 2) // node vector
        nodalDataBase->registerVectorState (stateName_old, pstateRef.dim[1]);
      else if (dl->rank() == 3) // node tensor
        nodalDataBase->registerVectorState (stateName_old, pstateRef.dim[1]*pstateRef.dim[2]);
    }
    else if (stateRef.entity == StateStruct::NodalDataToElemNode ||
             stateRef.entity == Albany::StateStruct::NodalDistParameter)
    {
      // These entities store data as <Cell,Node,...>, so the dl has rank = fieldDim + 1
      if (dl->rank() == 2) // node scalar
        nodalDataBase->registerVectorState (stateName_old, 1);
      else if (dl->rank() == 3) // node vector
        nodalDataBase->registerVectorState (stateName_old, pstateRef.dim[1]);
      else if (dl->rank() == 4) // node tensor
        nodalDataBase->registerVectorState (stateName_old, pstateRef.dim[1]*pstateRef.dim[2]);
    }
  }

  // insert
  stateRef.nameMap[stateName] = ebName;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerSideSetStateVariable(const std::string& sideSetName,
                                                   const std::string& stateName,
                                                   const std::string& fieldName,
                                                   const Teuchos::RCP<PHX::DataLayout> &dl,
                                                   const std::string& ebName,
                                                   const bool outputToExodus,
                                                   StateStruct::MeshFieldEntity const* fieldEntity)
{
  return registerSideSetStateVariable (sideSetName,stateName,fieldName,dl,ebName,
                                       "",0.0,false,outputToExodus,"",fieldEntity,"");
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerSideSetStateVariable(const std::string& sideSetName,
                                                   const std::string& stateName,
                                                   const std::string& fieldName,
                                                   const Teuchos::RCP<PHX::DataLayout> &dl,
                                                   const std::string& ebName,
                                                   const std::string &init_type,
                                                   const double init_val,
                                                   const bool registerOldState,
                                                   const bool outputToExodus,
                                                   const std::string &responseIDtoRequire,
                                                   StateStruct::MeshFieldEntity const* fieldEntity,
                                                   const std::string& meshPartName)
{
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);
  using Albany::StateStruct;

  // Create param list for SaveSideSetStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save Side Set State "
                + stateName + " to/from Side Set Field " + fieldName));
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", fieldName);
  p->set<const std::string>("Side Set Name", sideSetName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("Field Layout", dl);

  if( sideSetStatesToStore[sideSetName][ebName].find(stateName) != sideSetStatesToStore[sideSetName][ebName].end() ) {
    //Duplicate registration.  This will occur when a problem's
    // constructEvaluators function (templated) registers state variables.

    //Perform a check here that dl and statesToStore[stateName] are the same:
    //TEUCHOS_TEST_FOR_EXCEPT(dl != statesToStore[stateName]);  //I don't know how to do this correctly (erik)
//    TEUCHOS_TEST_FOR_EXCEPT(!(*dl == *statesToStore[stateName]));
    return p;  // Don't re-register the same state name
  }

  sideSetStatesToStore[sideSetName][ebName][stateName] = dl;

  if ( sideSetStateInfo.find(sideSetName)==sideSetStateInfo.end() )
  {
    // It's the first time we register states on this side set, so we initiate the pointer
    //TODO, when compiler allows, replace following with this for performance: sideSetStateInfo.emplace(sideSetName,Teuchos::rcp(new StateInfoStruct()));
    sideSetStateInfo.insert(std::make_pair(sideSetName,Teuchos::rcp(new StateInfoStruct())));
  }

  const Teuchos::RCP<StateInfoStruct>& sis_ptr = sideSetStateInfo.at(sideSetName);

  // Load into StateInfo
  StateStruct::MeshFieldEntity mfe_type;
  if(fieldEntity)
  {
    mfe_type = *fieldEntity;
  }
  else if(dl->rank() >= 1 && dl->name(0) == "Node") // Nodal data
  {
    mfe_type = StateStruct::NodalData; // One value per node
  }
  else if (dl->rank() == 2 && dl->name(1) == "Side") // Element data
  {
      mfe_type = StateStruct::ElemData; // One value per element
  }
  else if(dl->rank() > 2)
  {
    if (dl->name(2) == "Dim")
      mfe_type = StateStruct::ElemData; // One vector/tensor per element
    else if (dl->name(2) == "Node") // Element node data
      mfe_type = StateStruct::ElemNode; // One value per side node
    else if(dl->name(2) == "QuadPoint") // Quad point data
      mfe_type = StateStruct::QuadPoint; // One value per side quad point
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "StateManager: Unknown Entity type.\n");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "StateManager: Unknown Entity type.\n");
  }

  sis_ptr->push_back(Teuchos::rcp(new StateStruct(stateName, mfe_type)));
  StateStruct& stateRef = *sis_ptr->back();
  stateRef.setInitType(init_type);
  stateRef.setInitValue(init_val);
  stateRef.setMeshPart(meshPartName);

  if (stateRef.entity == StateStruct::NodalData)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(dl->name(0) == "Node", std::logic_error, "Error! NodalData states should have dl <Node,...>.\n");

    dl->dimensions(stateRef.dim);

    // Register the state with the nodalDataVector also.
    Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getSideSetNodalDataBase(sideSetName);

    if (dl->rank() == 1) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 2) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]);
    else if (dl->rank() == 3) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[1]*stateRef.dim[2]);
  }
  else if (stateRef.entity == StateStruct::NodalDataToElemNode ||
           stateRef.entity == Albany::StateStruct::NodalDistParameter)
  {
    // Strip one dimension out cause it's Side
    TEUCHOS_TEST_FOR_EXCEPTION (dl->rank()<2,  std::logic_error, "Error! The given layout does not appear to be that of a side set field.\n");

    stateRef.dim.resize(dl->rank()-1);
    stateRef.dim[0] = dl->dimension(0);
    for (int i(1); i<stateRef.dim.size(); ++i)
      stateRef.dim[i] = dl->dimension(i+1);

    // Register the state with the nodalDataVector also.
    Teuchos::RCP<Adapt::NodalDataBase> nodalDataBase = getSideSetNodalDataBase(sideSetName);

    // These entities store data as <Cell,Side,Node,...>, so the dl has one more rank
    if (dl->rank() == 3) // node scalar
      nodalDataBase->registerVectorState (stateName, 1);
    else if (dl->rank() == 4) // node vector
      nodalDataBase->registerVectorState (stateName, stateRef.dim[2]);
    else if (dl->rank() == 5) // node tensor
      nodalDataBase->registerVectorState (stateName, stateRef.dim[2]*stateRef.dim[3]);
  }
  else
  {
    // Strip one dimension out cause it's Side
    TEUCHOS_TEST_FOR_EXCEPTION (dl->rank()<2,  std::logic_error, "Error! The given layout does not appear to be that of a side set field.\n");

    stateRef.dim.resize(dl->rank()-1);
    stateRef.dim[0] = dl->dimension(0);
    for (int i(1); i<stateRef.dim.size(); ++i)
      stateRef.dim[i] = dl->dimension(i+1);
  }
  stateRef.output = outputToExodus;
  stateRef.responseIDtoRequire = responseIDtoRequire;
  stateRef.layered = (dl->name(dl->rank()-1) == "LayerDim");
  TEUCHOS_TEST_FOR_EXCEPTION (stateRef.layered && (dl->dimension(dl->rank()-1)<=0), std::logic_error,
                              "Error! Invalid number of layers for layered state " << stateName << ".\n");

  // If space is needed for old state
  if (registerOldState) {
    stateRef.saveOldState = true;

    std::string stateName_old = stateName + "_old";
    sis_ptr->push_back(Teuchos::rcp(new Albany::StateStruct(stateName_old, mfe_type)));
    Albany::StateStruct& pstateRef = *sis_ptr->back();
    pstateRef.initType  = init_type;
    pstateRef.initValue = init_val;
    pstateRef.pParentStateStruct = &stateRef;

    pstateRef.output = false;
    dl->dimensions(pstateRef.dim);
  }

  // insert
  stateRef.nameMap[stateName] = ebName;
}

Teuchos::RCP<Albany::StateInfoStruct>
Albany::StateManager::getStateInfoStruct() const
{
  return stateInfo;
}

const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >&
Albany::StateManager::getSideSetStateInfoStruct() const
{
  return sideSetStateInfo;
}

void
Albany::StateManager::setStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& disc_)
{
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);
  stateVarsAreAllocated = true;

  disc = disc_;

  doSetStateArrays(disc,stateInfo);

  // First, we check the explicitly required side discretizations exist...
  const auto& ss_discs = disc->getSideSetDiscretizations();
  for (auto const& it : sideSetStateInfo)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (ss_discs.find(it.first)==ss_discs.end(), std::logic_error,
                                "Error! Side Set " << it.first << "has sideSet states registered but no discretizations.\n");
  }

  // Then we make sure that for every side discretization there is a StateInfoStruct (possibly empty)
  for (auto const& it : disc->getSideSetDiscretizations())
  {
    Teuchos::RCP<StateInfoStruct>& sis = sideSetStateInfo[it.first];
    if (sis==Teuchos::null)
    {
      // Initialize to an empty StateInfoStruct
      sis = Teuchos::rcp(new StateInfoStruct());
      sis->createNodalDataBase();
    }
    doSetStateArrays (it.second,sis); // If sis was null, this should basically do nothing
  }
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::StateManager::
getDiscretization() const
{
  return disc;
}

void
Albany::StateManager::
importStateData(Albany::StateArrays& states_from)
{
  TEUCHOS_TEST_FOR_EXCEPT(!stateVarsAreAllocated);

  // Get states from STK mesh
  Albany::StateArrays& sa = getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  Albany::StateArrayVec& nsa = sa.nodeStateArrays;
  Albany::StateArrayVec& elemStatesToCopyFrom = states_from.elemStateArrays;
  Albany::StateArrayVec& nodeStatesToCopyFrom = states_from.nodeStateArrays;
  int numElemWorksets = esa.size();
  int numNodeWorksets = nsa.size();

  TEUCHOS_TEST_FOR_EXCEPT((unsigned int)numElemWorksets != elemStatesToCopyFrom.size());
  TEUCHOS_TEST_FOR_EXCEPT((unsigned int)numNodeWorksets != nodeStatesToCopyFrom.size());

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << std::endl;

  for (unsigned int i=0; i<stateInfo->size(); i++) {
    const std::string stateName = (*stateInfo)[i]->name;

    switch((*stateInfo)[i]->entity){

     case Albany::StateStruct::WorksetValue :
     case Albany::StateStruct::ElemData :
     case Albany::StateStruct::QuadPoint :
     case Albany::StateStruct::ElemNode :

      //check if state exists in statesToCopyFrom (check first workset only)
      if( elemStatesToCopyFrom[0].find(stateName) == elemStatesToCopyFrom[0].end() ) {
        //*out << "StateManager: state " << stateName << " not present, so not filled" << std::endl;
        continue;
      }

      *out << "StateManager: filling state:  " << stateName << std::endl;
      for (int ws = 0; ws < numElemWorksets; ws++)
      {
        Albany::StateStruct::FieldDims dims;
        esa[ws][stateName].dimensions(dims);
        int size = dims.size();

        switch (size) {
        case 1:
    for (int cell = 0; cell < dims[0]; ++cell)
      esa[ws][stateName](cell) = elemStatesToCopyFrom[ws][stateName](cell);
    break;
        case 2:
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
        esa[ws][stateName](cell, qp) = elemStatesToCopyFrom[ws][stateName](cell, qp);
    break;
        case 3:
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
        for (int i = 0; i < dims[2]; ++i)
          esa[ws][stateName](cell, qp, i) = elemStatesToCopyFrom[ws][stateName](cell, qp, i);
    break;
        case 4:
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
        for (int i = 0; i < dims[2]; ++i)
          for (int j = 0; j < dims[3]; ++j)
      esa[ws][stateName](cell, qp, i, j) = elemStatesToCopyFrom[ws][stateName](cell, qp, i, j);
    break;
        default:
    TEUCHOS_TEST_FOR_EXCEPTION(size<2||size>4, std::logic_error,
             "Something is wrong during zero state variable fill: " << size);
        }
     }

     break;

     case Albany::StateStruct::NodalData :

      //check if state exists in statesToCopyFrom (check first workset only)
      if( nodeStatesToCopyFrom[0].find(stateName) == nodeStatesToCopyFrom[0].end() ) {
        //*out << "StateManager: state " << stateName << " not present, so not filled" << std::endl;
        continue;
      }

      *out << "StateManager: filling state:  " << stateName << std::endl;
      for (int ws = 0; ws < numNodeWorksets; ws++){

        Albany::StateStruct::FieldDims dims;
        nsa[ws][stateName].dimensions(dims);
        int size = dims.size();

        switch (size) {
        case 1: // node scalar
    for (int node = 0; node < dims[0]; ++node)
      nsa[ws][stateName](node) = nodeStatesToCopyFrom[ws][stateName](node);
    break;
        case 2: // node vector
    for (int node = 0; node < dims[0]; ++node)
      for (int dim = 0; dim < dims[1]; ++dim)
        nsa[ws][stateName](node, dim) = nodeStatesToCopyFrom[ws][stateName](node, dim);
    break;
        case 3: // node tensor
    for (int node = 0; node < dims[0]; ++node)
      for (int dim = 0; dim < dims[1]; ++dim)
        for (int i = 0; i < dims[2]; ++i)
          nsa[ws][stateName](node, dim, i) = nodeStatesToCopyFrom[ws][stateName](node, dim, i);
    break;
        default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
             "Something is wrong during node zero state variable fill: " << size);
        }
      }
     break;
        default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
             "Unknown state variable entity encountered " << (*stateInfo)[i]->entity);
    }
  }

  *out << std::endl;
}

Albany::StateArray&
Albany::StateManager::getStateArray(SAType type, const int ws) const
{
  TEUCHOS_TEST_FOR_EXCEPT(!stateVarsAreAllocated);

  switch(type){

  case ELEM:
    return getStateArrays().elemStateArrays[ws];
    break;
  case NODE:
    return getStateArrays().nodeStateArrays[ws];
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error: Cannot match state array type in getStateArray()" << std::endl);
  }
}

Albany::StateArrays&
Albany::StateManager::getStateArrays() const
{
  TEUCHOS_TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  return disc->getStateArrays();
}

void
Albany::StateManager::updateStates()
{
  // Swap boolean that defines old and new (in terms of state1 and 2) in accessors
  TEUCHOS_TEST_FOR_EXCEPT(!stateVarsAreAllocated);

  // Get states from STK mesh
  Albany::StateArrays& sa = disc->getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  Albany::StateArrayVec& nsa = sa.nodeStateArrays;
  int numElemWorksets = esa.size();
  int numNodeWorksets = nsa.size();

  // For each workset, loop over registered states

  for (unsigned int i=0; i<stateInfo->size(); i++) {
    if ((*stateInfo)[i]->saveOldState) {
      const std::string stateName = (*stateInfo)[i]->name;
      const std::string stateName_old = stateName + "_old";

      switch((*stateInfo)[i]->entity){

      case Albany::StateStruct::NodalDataToElemNode :
        for (int ws = 0; ws < numNodeWorksets; ws++)
           for (int j = 0; j < nsa[ws][stateName].size(); j++)
             nsa[ws][stateName_old][j] = nsa[ws][stateName][j];

      case Albany::StateStruct::WorksetValue :
      case Albany::StateStruct::ElemData :
      case Albany::StateStruct::QuadPoint :
      case Albany::StateStruct::ElemNode :

        for (int ws = 0; ws < numElemWorksets; ws++)
          for (int j = 0; j < esa[ws][stateName].size(); j++)
            esa[ws][stateName_old][j] = esa[ws][stateName][j];

        break;

      case Albany::StateStruct::NodalData :

        for (int ws = 0; ws < numNodeWorksets; ws++)
          for (int j = 0; j < nsa[ws][stateName].size(); j++)
            nsa[ws][stateName_old][j] = nsa[ws][stateName][j];

        break;

       default :
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
             "Error: Cannot match state entity : " << (*stateInfo)[i]->entity << " in state manager. " << std::endl);
        break;
      }
    }
  }
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<Albany::EigendataStruct>
Albany::StateManager::getEigenData()
{
  return eigenData;
}

void
Albany::StateManager::setEigenData(const Teuchos::RCP<Albany::EigendataStruct>& eigdata)
{
  eigenData = eigdata;
}


Teuchos::RCP<Epetra_MultiVector>
Albany::StateManager::getAuxData()
{
  return auxData;
}

Teuchos::RCP<Tpetra_MultiVector>
Albany::StateManager::getAuxDataT()
{
  return auxDataT;
}


void
Albany::StateManager::setAuxData(const Teuchos::RCP<Epetra_MultiVector>& aux_data)
{
  auxData = aux_data;
}

#endif

void
Albany::StateManager::setAuxDataT(const Teuchos::RCP<Tpetra_MultiVector>& aux_data)
{
  auxDataT = aux_data;
}

void
Albany::StateManager::setEigenDataT(const Teuchos::RCP<Albany::EigendataStructT>& eigdata)
{
  eigenDataT = eigdata;
}

std::vector<std::string>
Albany::StateManager::getResidResponseIDsToRequire(std::string & elementBlockName)
{
  std::string id, name, ebName;
  std::vector<std::string> idsToRequire;

  int i = 0;
  for (Albany::StateInfoStruct::const_iterator st = stateInfo->begin(); st!= stateInfo->end(); st++) {
    name = (*st)->name;
    id = (*st)->responseIDtoRequire;
    ebName = (*st)->nameMap[name];
    if ( id.length() > 0 && ebName == elementBlockName ) {
      idsToRequire.push_back(id);
#ifdef ALBANY_VERBOSE
      cout << "RRR1  " << name << " requiring " << id << " (" << i << ")" << endl;
#endif
    }
    else {
#ifdef ALBANY_VERBOSE
      cout << "RRR1  " << name << " empty (" << i << ")" << endl;
#endif
    }
    i++;
  }
  return idsToRequire;
}

// ============================================= PRIVATE METHODS =============================================== //

void
Albany::StateManager::doSetStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
                                       const Teuchos::RCP<StateInfoStruct>& stateInfoPtr)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Get states from STK mesh
  Albany::StateArrays& sa = disc->getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  Albany::StateArrayVec& nsa = sa.nodeStateArrays;

  int numElemWorksets = esa.size();
  int numNodeWorksets = nsa.size();

  // For each workset, loop over registered states

  for (unsigned int i=0; i<stateInfoPtr->size(); i++) {
    const std::string& stateName = (*stateInfoPtr)[i]->name;
    const std::string& init_type = (*stateInfoPtr)[i]->initType;
    const double init_val        = (*stateInfoPtr)[i]->initValue;
    bool have_restart            = (*stateInfoPtr)[i]->restartDataAvailable;
    Albany::StateStruct *pParentStruct = (*stateInfoPtr)[i]->pParentStateStruct;

    // JTO: specifying zero recovers previous behavior
    // if (stateName == "zero")
    // {
    //   init_val = 0.0;
    //   init_type = "scalar";
    // }

    *out << "StateManager: initializing state:  " << stateName << "\n";
    switch((*stateInfoPtr)[i]->entity){

     case Albany::StateStruct::WorksetValue :
     case Albany::StateStruct::ElemData :
     case Albany::StateStruct::QuadPoint :
     case Albany::StateStruct::ElemNode :

      if(have_restart){
          *out << " from restart file." << std::endl;
          // If we are restarting, arrays should already be initialized from exodus file
          continue;
      }
      else if(pParentStruct && pParentStruct->restartDataAvailable){
          *out << " from restarted parent state." << std::endl;
          // If we are restarting, my parent is initialized from exodus file
          // Copy over parent's state

          for (int ws = 0; ws < numElemWorksets; ws++)

            esa[ws][stateName] = esa[ws][pParentStruct->name];

          continue;
      }
      else if (init_type == "scalar")
        *out << " with initialization type 'scalar' and value: " << init_val << std::endl;
      else if (init_type == "identity")
        *out << " with initialization type 'identity'" << std::endl;

      for (int ws = 0; ws < numElemWorksets; ws++){

        Albany::StateStruct::FieldDims dims;
        esa[ws][stateName].dimensions(dims);
        int size = dims.size();

        if (init_type == "scalar"){

          switch (size) {

            case 1:
              for (int cell = 0; cell < dims[0]; ++cell)
                esa[ws][stateName](cell) = init_val;
              break;

            case 2:
              for (int cell = 0; cell < dims[0]; ++cell)
                for (int qp = 0; qp < dims[1]; ++qp)
                  esa[ws][stateName](cell, qp) = init_val;
              break;

            case 3:
              for (int cell = 0; cell < dims[0]; ++cell)
                for (int qp = 0; qp < dims[1]; ++qp)
                  for (int i = 0; i < dims[2]; ++i)
                    esa[ws][stateName](cell, qp, i) = init_val;
              break;

            case 4:
              for (int cell = 0; cell < dims[0]; ++cell)
                for (int qp = 0; qp < dims[1]; ++qp)
                  for (int i = 0; i < dims[2]; ++i)
                   for (int j = 0; j < dims[3]; ++j)
                     esa[ws][stateName](cell, qp, i, j) = init_val;
              break;

            case 5:
              for (int cell = 0; cell < dims[0]; ++cell)
                for (int qp = 0; qp < dims[1]; ++qp)
                  for (int i = 0; i < dims[2]; ++i)
                   for (int j = 0; j < dims[3]; ++j)
                     for (int k = 0; k < dims[4]; ++k)
                       esa[ws][stateName](cell, qp, i, j, k) = init_val;
              break;

            default:
              TEUCHOS_TEST_FOR_EXCEPTION(size<2||size>5, std::logic_error,
                       "Something is wrong during scalar state variable initialization: " << size);
          }

        }
        else if (init_type == "identity"){

          // we assume operating on the last two indices is correct
          TEUCHOS_TEST_FOR_EXCEPTION(size != 4, std::logic_error,
             "Something is wrong during tensor state variable initialization: " << size);
          TEUCHOS_TEST_FOR_EXCEPT( ! (dims[2] == dims[3]) );

          for (int cell = 0; cell < dims[0]; ++cell)
            for (int qp = 0; qp < dims[1]; ++qp)
              for (int i = 0; i < dims[2]; ++i)
                for (int j = 0; j < dims[3]; ++j)
                  if (i==j) esa[ws][stateName](cell, qp, i, i) = 1.0;
                  else      esa[ws][stateName](cell, qp, i, j) = 0.0;
        }
      }
     break;

     case Albany::StateStruct::NodalData :

      if(have_restart){
          *out << " from restart file." << std::endl;
          // If we are restarting, arrays should already be initialized from exodus file
          continue;
      }
      else if(pParentStruct && pParentStruct->restartDataAvailable){
          *out << " from restarted parent state." << std::endl;
          // If we are restarting, my parent is initialized from exodus file
          // Copy over parent's state

          for (int ws = 0; ws < numNodeWorksets; ws++)

            nsa[ws][stateName] = nsa[ws][pParentStruct->name];

          continue;
      }
      else if (init_type == "scalar")
        *out << " with initialization type 'scalar' and value: " << init_val << std::endl;
      else if (init_type == "identity")
        *out << " with initialization type 'identity'" << std::endl;

      for (int ws = 0; ws < numNodeWorksets; ws++){

        Albany::StateStruct::FieldDims dims;
        nsa[ws][stateName].dimensions(dims);
        int size = dims.size();

        if (init_type == "scalar")

          switch (size) {

            case 1: // node scalar
              for (int node = 0; node < dims[0]; ++node)
                nsa[ws][stateName](node) = init_val;
              break;

            case 2: // node vector
              for (int node = 0; node < dims[0]; ++node)
                for (int dim = 0; dim < dims[1]; ++dim)
                  nsa[ws][stateName](node, dim) = init_val;
              break;

            case 3: // node tensor
              for (int node = 0; node < dims[0]; ++node)
                for (int dim = 0; dim < dims[1]; ++dim)
                  for (int i = 0; i < dims[2]; ++i)
                    nsa[ws][stateName](node, dim, i) = init_val;
              break;

            default:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "Something is wrong during node scalar state variable initialization: " << size);
        }
        else if (init_type == "identity"){

          // we assume operating on the last two indices is correct
          TEUCHOS_TEST_FOR_EXCEPTION(size != 3, std::logic_error,
             "Something is wrong during node tensor state variable initialization: " << size);
          TEUCHOS_TEST_FOR_EXCEPT( ! (dims[1] == dims[2]) );

          for (int node = 0; node < dims[0]; ++node)
            for (int i = 0; i < dims[1]; ++i)
              for (int j = 0; j < dims[2]; ++j)
                if (i==j) nsa[ws][stateName](node, i, i) = 1.0;
                else      nsa[ws][stateName](node, i, j) = 0.0;
        }
      }
     break;

     case Albany::StateStruct::NodalDataToElemNode :
     case Albany::StateStruct::NodalDistParameter :

      if(have_restart){
          *out << " from restart file." << std::endl;
          // If we are restarting, arrays should already be initialized from exodus file
          continue;
      }
      else if(pParentStruct && pParentStruct->restartDataAvailable) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error: At the moment it is not possible to restart a NodalDataToElemNode field or a NodalDistParameter field from parent structure" << std::endl);
      }
      else if ((init_type == "scalar")||(init_type == "identity")) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error: At the moment it is not possible to initialize a NodalDataToElemNode field or a NodalDistParameter field. It should be initialized when building the mesh" << std::endl);
      }
    }
  }
  *out << std::endl;
}
