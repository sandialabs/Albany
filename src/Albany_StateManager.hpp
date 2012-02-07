/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_STATEMANAGER
#define ALBANY_STATEMANAGER

#include <string>
#include <map>
#include <vector>
#include "Epetra_Vector.h"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Albany_DataTypes.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"

namespace Albany {

//! Class to manage saved state data. 
/* \brief The usage is to register state variables that will be saved
 * during problem construction, where they are described by a string
 * and a DataLayout. One time, the allocate method is called, which 
 * creates the memory for a vector of worksets of these states, which
 * are stored as MDFields.
*/

class StateManager {
public:
  StateManager ();

  ~StateManager () { };

  typedef std::map<std::string, Teuchos::RCP<PHX::DataLayout> >  RegisteredStates;

  //! Method to call multiple times (before allocate) to register which states will be saved.
  void registerStateVariable(const std::string &stateName, 
			     const Teuchos::RCP<PHX::DataLayout> &dl,
			     const std::string &init_type="scalar",
			     const double init_val=0.0,
			     const bool registerOldState=false,
			     const bool outputToExodus=true,
			     const std::string &responseIDtoRequire="");

  //! Method to call multiple times (before allocate) to register which states will be saved.
  //! Returns param vector with all info to build a SaveStateField or LoadStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl, 
                        const Teuchos::RCP<PHX::DataLayout> &dummy,
                        const std::string &init_type="scalar",
                        const double init_val=0.0,
                        const bool registerOldState=false);

  //! If field name to save/load is different from state name
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl, 
			const Teuchos::RCP<PHX::DataLayout> &dummy,
			const std::string &init_type,
                        const double init_val,
                        const bool registerOldState,
			const std::string &fieldName);




  //! Method to re-initialize state variables, which can be called multiple times after allocating
  void importStateData(Albany::StateArrays& statesToCopyFrom);

  //! Method to get the Names of the state variables
  RegisteredStates& getRegisteredStates(){return statesToStore;};

  //! Method to get the ResponseIDs for states which have been registered and (should)
  //!  have a SaveStateField evaluator associated with them that evaluates the responseID
  std::vector<std::string> getResidResponseIDsToRequire();

  //! Method to make the current newState the oldState, and vice versa
  void updateStates();

  //! Method to get a StateInfoStruct of info needed by STK to output States as Fields
  Teuchos::RCP<Albany::StateInfoStruct> getStateInfoStruct();

  //! Method to set discretization object
  void setStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& discObj);

  //! Method to get discretization object
  Teuchos::RCP<Albany::AbstractDiscretization> getDiscretization();

  //! Method to get state information for a specific workset
  Albany::StateArray& getStateArray(int ws) const;
  //! Method to get state information for all worksets
  Albany::StateArrays& getStateArrays() const;
  
  //! Methods to get/set the EigendataStruct which holds eigenvalue / eigenvector data
  Teuchos::RCP<Albany::EigendataStruct> getEigenData();
  void setEigenData(const Teuchos::RCP<Albany::EigendataStruct>& eigdata);

private:
  //! Private to prohibit copying
  StateManager(const StateManager&);

  //! Private to prohibit copying
  StateManager& operator=(const StateManager&);

private:

  //! boolean to enforce that allocate gets called once, and after registration and befor gets
  bool stateVarsAreAllocated;

  //! Container to hold the states that have been registered, to be allocated later
  RegisteredStates statesToStore;

  //! Discretization object which allows StateManager to perform input/output with exodus and Epetra vectors
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  //! NEW WAY
  Teuchos::RCP<StateInfoStruct> stateInfo;
  Teuchos::RCP<EigendataStruct> eigenData;

  // Experiment in dealing with Time
  double time;
  double timeOld;
};

}
#endif
