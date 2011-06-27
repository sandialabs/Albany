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

namespace Albany {

//! Class to manage saved state data. 
/* \brief The usage is to register state variables that will be saved
 * during problem construction, where they are described by a string
 * and a DataLayout. One time, the allocate method is called, which 
 * creates the memory for a vector of worksets of these states, which
 * are stored as MDFields.
*/
typedef std::map<std::string, Teuchos::RCP<Intrepid::FieldContainer<RealType> > >  StateVariables;

class StateManager {
public:
  StateManager ();

  ~StateManager () { };

  typedef std::map<std::string, Teuchos::RCP<PHX::DataLayout> >  RegisteredStates;

  //! Method to call multiple timed (before allocate) to register which states will be saved.
  //! Now returns param vector with all info to build a SaveStateField or LoadStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl, 
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const int saveOrLoadStateFieldID,
                                            const std::string &init_type="zero",
                                            const bool registerOldState=false);

  //! If field name to save/load is different from state name
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl, 
			const Teuchos::RCP<PHX::DataLayout> &dummy,
			const int saveOrLoadStateFieldID,
			const std::string &init_type,
                        const bool registerOldState,
			const std::string &fieldName);


  //! Function to allocate storage, called once after registering and before get calls
  void allocateStateVariables(const int numWorksets=1);

  //! Method to initialize state variables, called once after allocating and before get calls
  void initializeStateVariables(const int numWorksets=1);

  //! Method to re-initialize state variables, which can be called multiplie times after allocating
  void reinitializeStateVariables(Teuchos::RCP<std::vector<StateVariables> >& stateVarsToCopyFrom, const int numWorksets=1);

  //! Method to get the saved "old" state as a const
  Teuchos::RCP<const StateVariables> getOldStateVariables(const int workset=0) const;

  //! Method to get the "new" state so that it can be overwritten
  Teuchos::RCP<StateVariables> getNewStateVariables(const int workset=0);

  //! Method to get thesaved "old" state as a vector over worksets
  Teuchos::RCP<std::vector<StateVariables> > getAllOldStateVariables();

  //! Method to get the "new" state as a vector over worksets
  Teuchos::RCP<std::vector<StateVariables> > getAllNewStateVariables();

  //! Method to get the "new" state so that it can be overwritten
  const std::vector<std::vector<double> > getElementAveragedStates();

  //! Method to set the "new" and "old" state using an Epetra vector
  void saveVectorAsState(const std::string& stateName, const Epetra_Vector& vec);

  //! Method to get the Names of the state variables
  RegisteredStates& getRegisteredStates(){return statesToStore;};

  //! Method to make the current newState the oldState, and vice versa
  void updateStates();

  //! Method to test whether a state is present (and allocated)
  bool containsState(const std::string& stateName) { return state1[0].find(stateName) != state1[0].end(); }

  //! Method to set discretization object
  void setDiscretization(const Teuchos::RCP<Albany::AbstractDiscretization>& discObj) { disc = discObj; }

  //! Method to get a StateInfoStruct of info needed by STK to output States as Fields
  Teuchos::RCP<Albany::StateInfoStruct> getStateInfoStruct();

  //! Method to set discretization object
  void setStateArrays(const Teuchos::RCP<Albany::AbstractDiscretization>& discObj);
  //! Method to set discretization object
  Albany::StateArray& getStateArray(int ws) const;

  //! Method to integrate a scalar-valued state over an element block 
  //  (zero-length ebName integrates over entire mesh)
  RealType integrateStateVariable(const std::string& stateName, const std::string& ebName,
				  const std::string& BFName, const std::string& wBFName);


private:
  //! Private to prohibit copying
  StateManager(const StateManager&);

  //! Private to prohibit copying
  StateManager& operator=(const StateManager&);

private:

  typedef std::map<std::string, std::string >  InitializationType;

  //! boolean that takes care of swapping new and old state
  bool state1_is_old_state;

  //! boolean to enforce that allocate gets called once, and after registration and befor gets
  bool stateVarsAreAllocated;

  //! Fully allocated memory for states, as a vector over worksets
  //! One of these is oldState, one is newState, depending on value of bool state1_is_old_state
  std::vector<StateVariables> state1;
  std::vector<StateVariables> state2;

  //! Container to hold the states that have been registered, to be allocated later
  RegisteredStates statesToStore;

  //! Container to hold the initilization type for each state
  InitializationType stateInit;

  //! Discretization object which allows StateManager to perform input/output with exodus and Epetra vectors
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  //! NEW WAY
  Teuchos::RCP<StateInfoStruct> stateInfo;
};

}
#endif
