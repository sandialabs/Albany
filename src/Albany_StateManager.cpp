//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_StateManager.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "PHAL_Dimension.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Albany
{

StateManager::StateManager()
 : stateVarsAreAllocated(false)
 , stateInfo(Teuchos::rcp(new StateInfoStruct))
{
  // Nothing to be done here
}

Teuchos::RCP<Teuchos::ParameterList>
StateManager::registerStateVariable(
    const std::string&                   stateName,
    const Teuchos::RCP<PHX::DataLayout>& dl,
    const std::string&                   ebName,
    const bool                           outputToExodus,
    StateStruct::MeshFieldEntity const*  fieldEntity,
    const std::string&                   meshPartName)
{
  std::string init_type = "none";
  double  init_val  = 0;
  std::string responseIDtoRequire = "";
  return registerStateVariable(
            stateName,
            dl,
            ebName,
            init_type,
            init_val,
            outputToExodus,
            responseIDtoRequire,
            fieldEntity,
            meshPartName);
}

Teuchos::RCP<Teuchos::ParameterList>
StateManager::registerStateVariable(
    const std::string&                   stateName,
    const Teuchos::RCP<PHX::DataLayout>& dl,
    const std::string&                   ebName,
    const std::string&                   init_type,
    const double                         init_val,
    const bool                           outputToExodus,
    const std::string&                   responseIDtoRequire,
    StateStruct::MeshFieldEntity const*  fieldEntity,
    const std::string&                   meshPartName)
{
  ALBANY_ASSERT(stateName != "", "State Name cannot be the empty string");
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);

  // Create param list for SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p =
      Teuchos::rcp(new Teuchos::ParameterList(
          "Save or Load State " + stateName + " to/from field " + stateName));
  p->set("State Name", stateName);
  p->set("Field Name", stateName);
  p->set("State Field Layout", dl);

  // Store layout (check for consistency if already present)
  auto& stored_dl = statesToStore[ebName][stateName];
  if (not stored_dl.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION (*stored_dl!=*dl, std::logic_error,
        "Error! Attempt to register side state with two different layouts.\n"
        " - eb name: " + ebName + "\n"
        " - state name: " + stateName + "\n"
        " - old layout: " + stored_dl->identifier() + "\n"
        " - new layout: " + dl->identifier() + "\n");
        
    return p;  // Don't re-register the same state name
  }
  stored_dl = dl;

  // Load into StateInfo
  StateStruct::MeshFieldEntity mfe_type;
  if (fieldEntity)
    mfe_type = *fieldEntity;
  else if (dl->rank() == 1 && dl->size() == 1)
    mfe_type = StateStruct::WorksetValue;  // One value for the whole workset (e.g., time)
  else if (dl->rank() == 1 && dl->name(0) == PHX::print<Cell>())
    mfe_type = StateStruct::ElemData;
  else if (dl->rank() >= 1 && dl->name(0) == PHX::print<Node>())  // Nodal data
    mfe_type = StateStruct::NodalData;
  else if (dl->rank() >= 1 && dl->name(0) == PHX::print<Cell>()) {  // Element QP or node data
    if (dl->rank() > 1 && dl->name(1) == PHX::print<Node>())        // Element node data
      mfe_type = StateStruct::ElemNode;
    else if (dl->rank() > 1 && dl->name(1) == PHX::print<QuadPoint>())  // Element node data
      mfe_type = StateStruct::QuadPoint;
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "StateManager: Element Entity type - " << dl->name(1) << " - not supported" << std::endl);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "StateManager: Unknown Entity type - " << dl->name(0) << " - not supported" << std::endl);

  stateInfo->push_back(Teuchos::rcp(new StateStruct(stateName, mfe_type)));
  StateStruct& stateRef = *stateInfo->back();
  stateRef.setInitType(init_type);
  stateRef.setInitValue(init_val);
  stateRef.setMeshPart(meshPartName);
  stateRef.setEBName(ebName);

  dl->dimensions(stateRef.dim);

  stateRef.output              = outputToExodus;
  stateRef.responseIDtoRequire = responseIDtoRequire;
  stateRef.layered             = (dl->name(dl->rank() - 1) == PHX::print<LayerDim>());

  std::cout << "register state " << stateName << ", dims: " << util::join(stateRef.dim,",") << "\n";

  // If space is needed for old state

  // insert
  stateRef.nameMap[stateName] = ebName;

  return p;
}

Teuchos::RCP<Teuchos::ParameterList>
StateManager::registerSideSetStateVariable(
    const std::string&                   sideSetName,
    const std::string&                   stateName,
    const std::string&                   fieldName,
    const Teuchos::RCP<PHX::DataLayout>& dl,
    const std::string&                   ebName,
    const bool                           outputToExodus,
    StateStruct::MeshFieldEntity const*  fieldEntity,
    const std::string&                   meshPartName)

{
  std::string init_type = "none";
  double  init_val  = 0;
  std::string responseIDtoRequire = "";
  return registerSideSetStateVariable(
        sideSetName,
        stateName,
        fieldName,
        dl,
        ebName,
        init_type,
        init_val,
        outputToExodus,
        responseIDtoRequire,
        fieldEntity,
        meshPartName);
}

Teuchos::RCP<Teuchos::ParameterList>
StateManager::registerSideSetStateVariable(
    const std::string&                   sideSetName,
    const std::string&                   stateName,
    const std::string&                   fieldName,
    const Teuchos::RCP<PHX::DataLayout>& dl,
    const std::string&                   ebName,
    const std::string&                   init_type,
    const double                         init_val,
    const bool                           outputToExodus,
    const std::string&                   responseIDtoRequire,
    StateStruct::MeshFieldEntity const*  fieldEntity,
    const std::string&                   meshPartName)
{
  TEUCHOS_TEST_FOR_EXCEPT(stateVarsAreAllocated);

  // Create param list for SaveSideSetStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList> p =
      Teuchos::rcp(new Teuchos::ParameterList(
          "Save Side Set State " + stateName + " to/from Side Set Field " +
          fieldName));
  p->set("State Name", stateName);
  p->set("Field Name", fieldName);
  p->set("Side Set Name", sideSetName);
  p->set("Field Layout", dl);

  // Store layout (check for consistency if already present)
  auto& stored_dl = sideSetStatesToStore[sideSetName][ebName][stateName];
  if (not stored_dl.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION (*stored_dl!=*dl, std::logic_error,
        "Error! Attempt to register side state with two different layouts.\n"
        " - side set name: " + sideSetName + "\n"
        " - eb name: " + ebName + "\n"
        " - state name: " + stateName + "\n"
        " - old layout: " + stored_dl->identifier() + "\n"
        " - new layout: " + dl->identifier() + "\n");
        
    return p;  // Don't re-register the same state name
  }
  stored_dl = dl;

  auto& sis_ptr = sideSetStateInfo[sideSetName];
  if (sis_ptr.is_null()) {
    // It's the first time we register states on this side set, so we initiate the pointer
    sis_ptr =  Teuchos::rcp(new StateInfoStruct());
  }

  // Load into StateInfo
  StateStruct::MeshFieldEntity mfe_type;
  if (fieldEntity) {
    mfe_type = *fieldEntity;
  } else if (dl->rank() >= 1 && dl->name(0) == PHX::print<Node>()) {
    // Nodal data (one value per node)
    mfe_type = StateStruct::NodalData;
  } else if (dl->rank() == 1 && dl->name(0) == PHX::print<Side>()) {
    // Element data (one value per element)
    mfe_type = StateStruct::ElemData;
  } else if (dl->rank() > 1) {
    if (dl->name(1) == PHX::print<Dim>())
      mfe_type = StateStruct::ElemData;   // One vector/tensor per element
    else if (dl->name(1) == PHX::print<Node>())       // Element node data
      mfe_type = StateStruct::ElemNode;   // One value per side node
    else if (dl->name(1) == PHX::print<QuadPoint>())  // Quad point data
      mfe_type = StateStruct::QuadPoint;  // One value per side quad point
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error, "StateManager: Unknown Entity type.\n"
          " - layout: " + dl->identifier() + "\n");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error, "StateManager: Unknown Entity type.\n"
          " - layout: " + dl->identifier() + "\n");
  }

  sis_ptr->push_back(Teuchos::rcp(new StateStruct(stateName, mfe_type)));
  StateStruct& stateRef = *sis_ptr->back();
  stateRef.setInitType(init_type);
  stateRef.setInitValue(init_val);
  stateRef.setMeshPart(meshPartName);

  dl->dimensions(stateRef.dim);
  if (stateRef.entity == StateStruct::NodalData) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        dl->name(0) == PHX::print<Node>(),std::logic_error,
        "Error! NodalData states should have dl <Node,...>.\n");
  }
  stateRef.output              = outputToExodus;
  stateRef.responseIDtoRequire = responseIDtoRequire;
  stateRef.layered             = (dl->name(dl->rank() - 1) == PHX::print<LayerDim>());
  TEUCHOS_TEST_FOR_EXCEPTION(
      stateRef.layered && (dl->extent(dl->rank() - 1) <= 0), std::logic_error,
      "Error! Invalid number of layers for layered state " << stateName << ".\n");

  // insert
  stateRef.nameMap[stateName] = ebName;

  return p;
}

Teuchos::RCP<StateInfoStruct>
StateManager::getStateInfoStruct() const
{
  return stateInfo;
}

const std::map<std::string, Teuchos::RCP<StateInfoStruct>>&
StateManager::getSideSetStateInfoStruct() const
{
  return sideSetStateInfo;
}

void
StateManager::initStateArrays(
    const Teuchos::RCP<AbstractDiscretization>& disc)
{
  if (stateVarsAreAllocated) return;

  stateVarsAreAllocated = true;

  doSetStateArrays(disc, stateInfo);

  // First, we check the explicitly required side discretizations exist...
  const auto& ss_discs = disc->getSideSetDiscretizations();
  for (auto const& it : sideSetStateInfo) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        ss_discs.find(it.first) == ss_discs.end(), std::logic_error,
        "Error! Side Set " << it.first << " has sideSet states registered but no discretizations.\n");
  }

  // Then we make sure that for every side discretization there is a
  // StateInfoStruct (possibly empty)
  for (auto const& it : disc->getSideSetDiscretizations()) {
    Teuchos::RCP<StateInfoStruct>& sis = sideSetStateInfo[it.first];
    if (sis == Teuchos::null) {
      // Initialize to an empty StateInfoStruct
      sis = Teuchos::rcp(new StateInfoStruct());
      // sis->createNodalFieldContainer();
    }
    doSetStateArrays(it.second, sis);
  }
}

std::vector<std::string>
StateManager::getResidResponseIDsToRequire(
    std::string& elementBlockName)
{
  std::string              id, name, ebName;
  std::vector<std::string> idsToRequire;

  int i = 0;
  for (StateInfoStruct::const_iterator st = stateInfo->begin();
       st != stateInfo->end();
       st++) {
    name   = (*st)->name;
    id     = (*st)->responseIDtoRequire;
    ebName = (*st)->nameMap[name];
    if (id.length() > 0 && ebName == elementBlockName) {
      idsToRequire.push_back(id);
#ifdef ALBANY_VERBOSE
      std::cout << "RRR1  " << name << " requiring " << id << " (" << i << ")"
		<< std::endl;
#endif
    } else {
#ifdef ALBANY_VERBOSE
      std::cout << "RRR1  " << name << " empty (" << i << ")" << std::endl;
#endif
    }
    i++;
  }
  return idsToRequire;
}

void
StateManager::doSetStateArrays(
    const Teuchos::RCP<AbstractDiscretization>& disc,
    const Teuchos::RCP<StateInfoStruct>&        stateInfoPtr)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(
      Teuchos::VerboseObjectBase::getDefaultOStream());

  // Get states from mesh accessor
  auto mfa = disc->getMeshStruct()->get_field_accessor();
  StateArrayVec& esa = mfa->getElemStates();
  StateArrayVec& nsa = mfa->getNodeStates();
  StateArray&    gsa = mfa->getGlobalStates();

  int numElemWorksets = esa.size();
  int numNodeWorksets = nsa.size();

  // For each workset, loop over registered states

  bool transferNodalStates = false;
  for (const auto& st : *stateInfoPtr) {
    const std::string& stateName    = st->name;
    const std::string& init_type    = st->initType;
    const std::string& ebName       = st->ebName;
    const double       init_val     = st->initValue;
    bool               have_restart = st->restartDataAvailable;

    *out << "[StateManager] Initializing state '" << stateName << "':";
    if (have_restart) {
      // If we are restarting, arrays should already be initialized from exodus file
      *out << " from restart file." << std::endl;
      continue;
    } else if (init_type=="none") {
      // Perhaps this is something we compute during the assembly, and does not need
      // an initial value
      *out << " no initialization needed." << std::endl;
      continue;
    } else if (init_type=="scalar") {
      *out << " with initialization type 'scalar' and value: " << init_val << "\n";
    } else if (init_type=="identity") {
      *out << " with initialization type 'identity'" << std::endl;
    } else {
      throw std::runtime_error("[StateManager] Unknown state init type '" + init_type + "'\n");
    }
    StateStruct* pParentStruct = st->pParentStateStruct;

    if (st->entity==StateStruct::NodalDataToElemNode) {
      transferNodalStates = true;
    }

    switch (st->entity) {
      case StateStruct::WorksetValue:
        if (init_type=="scalar") {
          auto gsa_h = gsa[stateName].host();
          for (size_t i=0; i<gsa_h.size(); ++i) {
            gsa_h.data()[i] = init_val;
          }
        }
        break;

      case StateStruct::ElemData:
      case StateStruct::QuadPoint:
      case StateStruct::ElemNode:

        if (pParentStruct && pParentStruct->restartDataAvailable) {
          // If we are restarting, my parent is initialized from exodus file
          // Copy over parent's state

          for (int ws = 0; ws < numElemWorksets; ws++)

            esa[ws][stateName] = esa[ws][pParentStruct->name];

          continue;
        }

        for (int ws = 0; ws < numElemWorksets; ws++) {
          /* because we loop over all worksets above, we need to check
             that the wsEBName is the same as the state variable ebName,
             and if it is not, we continue, otherwise we overwrite previously
             initialized data */
          std::string wsEBName = (disc->getWsEBNames())[ws];
          if (wsEBName != ebName) continue;

          StateStruct::FieldDims dims;
          esa[ws][stateName].dimensions(dims);
          int size = dims.size();

          auto& esa_h = esa[ws][stateName].host();

          if (init_type == "scalar") {
            switch (size) {
              case 1:
                for (size_t cell = 0; cell < dims[0]; ++cell)
                  esa_h(cell) = init_val;
                break;

              case 2:
                for (size_t cell = 0; cell < dims[0]; ++cell)
                  for (size_t qp = 0; qp < dims[1]; ++qp)
                    esa_h(cell, qp) = init_val;
                break;

              case 3:
                for (size_t cell = 0; cell < dims[0]; ++cell)
                  for (size_t qp = 0; qp < dims[1]; ++qp)
                    for (size_t i = 0; i < dims[2]; ++i)
                      esa_h(cell, qp, i) = init_val;
                break;

              case 4:
                for (size_t cell = 0; cell < dims[0]; ++cell)
                  for (size_t qp = 0; qp < dims[1]; ++qp)
                    for (size_t i = 0; i < dims[2]; ++i)
                      for (size_t j = 0; j < dims[3]; ++j)
                        esa_h(cell, qp, i, j) = init_val;
                break;

              case 5:
                for (size_t cell = 0; cell < dims[0]; ++cell)
                  for (size_t qp = 0; qp < dims[1]; ++qp)
                    for (size_t i = 0; i < dims[2]; ++i)
                      for (size_t j = 0; j < dims[3]; ++j)
                        for (size_t k = 0; k < dims[4]; ++k)
                          esa_h(cell, qp, i, j, k) = init_val;
                break;

              default:
                TEUCHOS_TEST_FOR_EXCEPTION(
                    size < 2 || size > 5,
                    std::logic_error,
                    "Something is wrong during scalar state variable "
                    "initialization: "
                        << size);
            }

          } else if (init_type == "identity") {
            // we assume operating on the last two indices is correct
            TEUCHOS_TEST_FOR_EXCEPTION(
                size != 4,
                std::logic_error,
                "Something is wrong during tensor state variable "
                "initialization: "
                    << size);
            TEUCHOS_TEST_FOR_EXCEPT(!(dims[2] == dims[3]));

            for (size_t cell = 0; cell < dims[0]; ++cell)
              for (size_t qp = 0; qp < dims[1]; ++qp)
                for (size_t i = 0; i < dims[2]; ++i)
                  for (size_t j = 0; j < dims[3]; ++j)
                    if (i == j)
                      esa_h(cell, qp, i, i) = 1.0;
                    else
                      esa_h(cell, qp, i, j) = 0.0;
          }
          esa[ws][stateName].sync_to_dev();
        }
        break;

      case StateStruct::NodalData:

        if (pParentStruct && pParentStruct->restartDataAvailable) {
          // If we are restarting, my parent is initialized from exodus file
          // Copy over parent's state

          for (int ws = 0; ws < numNodeWorksets; ws++)

            nsa[ws][stateName] = nsa[ws][pParentStruct->name];

          continue;
        }

        for (int ws = 0; ws < numNodeWorksets; ws++) {
          StateStruct::FieldDims dims;
          nsa[ws][stateName].dimensions(dims);
          int size = dims.size();

          auto& nsa_h = nsa[ws][stateName].host();

          if (init_type == "scalar") switch (size) {
              case 1:  // node scalar
                for (size_t node = 0; node < dims[0]; ++node)
                  nsa_h(node) = init_val;
                break;

              case 2:  // node vector
                for (size_t node = 0; node < dims[0]; ++node)
                  for (size_t dim = 0; dim < dims[1]; ++dim)
                    nsa_h(node, dim) = init_val;
                break;

              case 3:  // node tensor
                for (size_t node = 0; node < dims[0]; ++node)
                  for (size_t dim = 0; dim < dims[1]; ++dim)
                    for (size_t i = 0; i < dims[2]; ++i)
                      nsa_h(node, dim, i) = init_val;
                break;

              default:
                TEUCHOS_TEST_FOR_EXCEPTION(
                    true,
                    std::logic_error,
                    "Something is wrong during node scalar state variable "
                    "initialization: "
                        << size);
            }
          else if (init_type == "identity") {
            // we assume operating on the last two indices is correct
            TEUCHOS_TEST_FOR_EXCEPTION(
                size != 3,
                std::logic_error,
                "Something is wrong during node tensor state variable "
                "initialization: "
                    << size);
            TEUCHOS_TEST_FOR_EXCEPT(!(dims[1] == dims[2]));
            for (size_t node = 0; node < dims[0]; ++node)
              for (size_t i = 0; i < dims[1]; ++i)
                for (size_t j = 0; j < dims[2]; ++j)
                  if (i == j)
                    nsa_h(node, i, i) = 1.0;
                  else
                    nsa_h(node, i, j) = 0.0;
          }
          nsa[ws][stateName].sync_to_dev();
        }
        break;

      case StateStruct::NodalDataToElemNode:
      case StateStruct::NodalDistParameter:

        if (pParentStruct && pParentStruct->restartDataAvailable) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "Error: At the moment it is not possible to restart a "
              "NodalDataToElemNode field or a NodalDistParameter field from "
              "parent structure"
                  << std::endl);
        } else if ((init_type == "scalar") || (init_type == "identity")) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "Error: At the moment it is not possible to initialize a "
              "NodalDataToElemNode field or a NodalDistParameter field. It "
              "should be initialized when building the mesh"
                  << std::endl);
        }
    }
  }
  *out << std::endl;

  // If we init-ed states with NodalDataToElemNode entity, we must transfer the
  // new data to the elem-based states
  if (transferNodalStates)
    disc->getMeshStruct()->get_field_accessor()->transferNodeStatesToElemStates();
}

} // namespace Albany
