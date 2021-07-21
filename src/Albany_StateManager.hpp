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
#include "Albany_EigendataInfoStructT.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_TpetraTypes.hpp"

#if defined(ALBANY_EPETRA)
#include "Albany_EigendataInfoStruct.hpp"
#include "Epetra_Vector.h"
#endif

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
  enum SAType
  {
    ELEM,
    NODE
  };

  StateManager();

  ~StateManager(){};

  typedef std::map<std::string, Teuchos::RCP<PHX::DataLayout>> RegisteredStates;

  /// Method to call multiple times (before allocate) to register which states
  /// will be saved.
  void
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const std::string&                   init_type           = "scalar",
      const double                         init_val            = 0.0,
      const bool                           registerOldState    = false,
      const bool                           outputToExodus      = true,
      const std::string&                   responseIDtoRequire = "",
      StateStruct::MeshFieldEntity const*  fieldEntity         = 0,
      const std::string&                   meshPartName        = "");

  void
  registerNodalVectorStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const std::string&                   init_type           = "scalar",
      const double                         init_val            = 0.0,
      const bool                           registerOldState    = false,
      const bool                           outputToExodus      = true,
      const std::string&                   responseIDtoRequire = "");

  /// Method to call multiple times (before allocate) to register which states
  /// will be saved.
  /// Returns param vector with all info to build a SaveStateField or
  /// LoadStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   name,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      const std::string&                   ebName,
      const std::string&                   init_type        = "scalar",
      const double                         init_val         = 0.0,
      const bool                           registerOldState = false);

  // Field entity is known. Useful for NodalDataToElemNode field. Input dl is of
  // ElemNode type
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const bool                           outputToExodus,
      StateStruct::MeshFieldEntity const*  fieldEntity,
      const std::string&                   meshPartName = "");

  /// If field name to save/load is different from state name
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      const std::string&                   ebName,
      const std::string&                   init_type,
      const double                         init_val,
      const bool                           registerOldState,
      const std::string&                   fieldName);

  /// If you want to give more control over whether or not to output to Exodus
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      const std::string&                   ebName,
      const std::string&                   init_type,
      const double                         init_val,
      const bool                           registerOldState,
      const bool                           outputToExodus);

  Teuchos::RCP<Teuchos::ParameterList>
  registerNodalVectorStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const Teuchos::RCP<PHX::DataLayout>& dummy,
      const std::string&                   ebName,
      const std::string&                   init_type,
      const double                         init_val,
      const bool                           registerOldState,
      const bool                           outputToExodus);

  /// Very basic
  void
  registerStateVariable(
      const std::string&                   stateName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   init_type);

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

  Teuchos::RCP<Teuchos::ParameterList>
  registerSideSetStateVariable(
      const std::string&                   sideSetName,
      const std::string&                   stateName,
      const std::string&                   fieldName,
      const Teuchos::RCP<PHX::DataLayout>& dl,
      const std::string&                   ebName,
      const std::string&                   init_type,
      const double                         init_val,
      const bool                           registerOldState,
      const bool                           outputToExodus,
      const std::string&                   responseIDtoRequire,
      StateStruct::MeshFieldEntity const*  fieldEntity,
      const std::string&                   meshPartName = "");

  /// Method to re-initialize state variables, which can be called multiple
  /// times after allocating
  void
  importStateData(Albany::StateArrays& statesToCopyFrom);

  /// Method to get the Names of the state variables
  const std::map<std::string, RegisteredStates>&
  getRegisteredStates() const
  {
    return statesToStore;
  }

  /// Method to get the Names of the state variables
  const std::map<std::string, std::map<std::string, RegisteredStates>>&
  getRegisteredSideSetStates() const
  {
    return sideSetStatesToStore;
  }

  /// Method to get the ResponseIDs for states which have been registered and
  /// (should)
  ///  have a SaveStateField evaluator associated with them that evaluates the
  ///  responseID
  std::vector<std::string>
  getResidResponseIDsToRequire(std::string& elementBlockName);

  /// Method to make the current newState the oldState, and vice versa
  void
  updateStates();

  /// Method to get a StateInfoStruct of info needed by STK to output States as
  /// Fields
  Teuchos::RCP<Albany::StateInfoStruct>
  getStateInfoStruct() const;

  /// Equivalent of previous method for the sideSets states
  const std::map<std::string, Teuchos::RCP<StateInfoStruct>>&
  getSideSetStateInfoStruct() const;

  /// Method to set discretization object
  void
  setupStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& discObj);

  /// Method to get discretization object
  Teuchos::RCP<Albany::AbstractDiscretization>
  getDiscretization() const;

  /// Method to get state information for a specific workset
  Albany::StateArray&
  getStateArray(SAType type, int ws) const;

  /// Method to get state information for all worksets
  Albany::StateArrays&
  getStateArrays() const;

  // Set the state array for all worksets.
  void
  setStateArrays(Albany::StateArrays& sa);

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

  void
  printStates(std::string const& where = "") const;

#if defined(ALBANY_EPETRA)
  /// Methods to get/set the EigendataStruct which holds eigenvalue /
  /// eigenvector data
  Teuchos::RCP<Albany::EigendataStruct>
  getEigenData();
  void
  setEigenData(const Teuchos::RCP<Albany::EigendataStruct>& eigdata);

  /// Methods to get/set Auxilliary data vectors
  Teuchos::RCP<Epetra_MultiVector>
  getAuxData();
  void
  setAuxData(const Teuchos::RCP<Epetra_MultiVector>& aux_data);
#endif
  Teuchos::RCP<Tpetra_MultiVector>
  getAuxDataT();

  void
  setEigenDataT(const Teuchos::RCP<Albany::EigendataStructT>& eigdata);
  void
  setAuxDataT(const Teuchos::RCP<Tpetra_MultiVector>& aux_data);
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
  /// and befor gets
  bool stateVarsAreAllocated;

  /// Container to hold the states that have been registered, by element block,
  /// to be allocated later
  std::map<std::string, RegisteredStates> statesToStore;
  std::map<std::string, std::map<std::string, RegisteredStates>>
      sideSetStatesToStore;

  /// Discretization object which allows StateManager to perform input/output
  /// with exodus and Epetra vectors
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  /// NEW WAY
  Teuchos::RCP<StateInfoStruct> stateInfo;
  std::map<std::string, Teuchos::RCP<StateInfoStruct>>
      sideSetStateInfo;  // A map sideSetName->stateInfoBd

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<EigendataStruct>    eigenData;
  Teuchos::RCP<Epetra_MultiVector> auxData;
#endif
  Teuchos::RCP<EigendataStructT>   eigenDataT;
  Teuchos::RCP<Tpetra_MultiVector> auxDataT;
};

}  // Namespace Albany

#endif  // ALBANY_STATE_MANAGER_HPP
