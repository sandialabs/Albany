//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATE_MANAGER_HPP
#define ALBANY_STATE_MANAGER_HPP

#include <map>
#include <string>
#include <vector>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Adapt_NodalDataBase.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_TpetraTypes.hpp"

namespace Albany {

/// Class to manage saved state data.
/* \brief The usage is to register state variables that will be saved
 * during problem construction, where they are described by a string
 * and a DataLayout. One time, the allocate method is called, which
 * creates the memory for a vector of worksets of these states, which
 * are stored as MDFields.
 */

class StateManager
{
public:
  StateManager();

  ~StateManager() = default;

  // Register a state (may be called multiple times)
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const std::string&                   init_type           = "scalar",
      const double                         init_val            = 0.0,
      const bool                           outputToExodus      = true,
      const std::string&                   responseIDtoRequire = "",
      StateStruct::MeshFieldEntity const*  fieldEntity         = 0,
      const std::string&                   meshPartName        = "");

  // Shortcut: allow to specify fieldEntity without bothering with initialization options
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const bool                           outputToExodus,
      StateStruct::MeshFieldEntity const*  fieldEntity,
      const std::string&                   meshPartName = "");

  // Register state living on side set (may be called multiple times)
  Teuchos::RCP<Teuchos::ParameterList>
  registerSideSetStateVariable(
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
      const std::string&                   meshPartName = "");

  // Shortcut: allow to specify fieldEntity without bothering with initialization options
  Teuchos::RCP<Teuchos::ParameterList>
  registerSideSetStateVariable(
      const std::string&                   sideSetName,
      const std::string&                   stateName,
      const std::string&                   fieldName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const bool                           outputToExodus,
      StateStruct::MeshFieldEntity const*  fieldEntity  = NULL,
      const std::string&                   meshPartName = "");

  /// Method to get the ResponseIDs for states which have been registered and
  /// (should)
  ///  have a SaveStateField evaluator associated with them that evaluates the
  ///  responseID
  std::vector<std::string>
  getResidResponseIDsToRequire(std::string& elementBlockName);

  /// Method to get a StateInfoStruct of info needed by STK to output States as
  /// Fields
  Teuchos::RCP<Albany::StateInfoStruct>
  getStateInfoStruct() const;

  /// Equivalent of previous method for the sideSets states
  const std::map<std::string, Teuchos::RCP<StateInfoStruct>>&
  getSideSetStateInfoStruct() const;

  /// Method to set discretization object
  void
  initStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& disc);

  Albany::StateArrays&
  getSideSetStateArrays(const std::string& sideSet);

  Teuchos::RCP<Adapt::NodalDataBase>
  getNodalDataBase()
  {
    return stateInfo->createNodalDataBase();
  }

  Teuchos::RCP<Adapt::NodalDataBase>
  getSideSetNodalDataBase(const std::string& sideSet)
  {
    return sideSetStateInfo.at(sideSet)->createNodalDataBase();
  }

  bool
  areStateVarsAllocated() const
  {
    return stateVarsAreAllocated;
  }

 private:
  /// Private to prohibit copying
  StateManager(const StateManager&);

  /// Private to prohibit copying
  StateManager&
  operator=(const StateManager&);

  /// Sets states arrays from a given StateInfoStruct into a given
  /// discretization
  void
  doSetStateArrays(
      const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
      const Teuchos::RCP<StateInfoStruct>&                stateInfoPtr);

  /// boolean to enforce that allocate gets called once, and after registration
  /// and before gets
  bool stateVarsAreAllocated;

  template<typename T>
  using strmap_t = std::map<std::string,T>;
  using RegisteredStates = strmap_t<Teuchos::RCP<PHX::DataLayout>>; // name->layout

  // Keep track of registered states elem block and layout,
  // to ensure we don't re-register with different ones later
  strmap_t<RegisteredStates> statesToStore; // ebName->RegisteredStates
  strmap_t<strmap_t<RegisteredStates>> sideSetStatesToStore; // sideSetName->ebName->RegisteredStates

  /// NEW WAY
  Teuchos::RCP<StateInfoStruct> stateInfo;
  std::map<std::string, Teuchos::RCP<StateInfoStruct>>
      sideSetStateInfo;  // A map sideSetName->stateInfoBd
};

}  // Namespace Albany

#endif  // ALBANY_STATE_MANAGER_HPP
