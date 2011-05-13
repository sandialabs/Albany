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
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Albany_DataTypes.hpp"
#include "Teuchos_ParameterList.hpp"

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
  //! Now returns param vector with all info to build a SaveStateField evaluator
  Teuchos::RCP<Teuchos::ParameterList>
  registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl, 
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const int saveStateFieldID,
                                            const std::string &init_type="zero");

  //! Method to call multiple timed (before allocate) to register which states will be saved.
  void registerStateVariable(const std::string &name, const Teuchos::RCP< PHX::DataLayout > &t, 
			     const std::string &init_type="zero");

  //! Function to allocate storage, called once after registering and before get calls
  void allocateStateVariables(const int numWorksets=1);

  //! Method to initialize state variables, called once after allocating and before get calls
  void initializeStateVariables(const int numWorksets=1);

  //! Method to get the saved "old" state as a const
  Teuchos::RCP<const StateVariables> getOldStateVariables(const int workset=0) const;

  //! Method to get the "new" state so that it can be overwritten
  Teuchos::RCP<StateVariables> getNewStateVariables(const int workset=0);

  //! Method to get the "new" state so that it can be overwritten
  const std::vector<std::vector<double> > getElementAveragedStates();

  //! Method to get the Names of the state variables
  RegisteredStates& getRegisteredStates(){return statesToStore;};

  //! Method to make the current newState the oldState, and vice versa
  void updateStates();

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
};

}
#endif
