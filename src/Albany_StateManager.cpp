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
#include "Albany_StateManager.hpp"
#include "Teuchos_VerboseObject.hpp"

Albany::StateManager::StateManager () :
  state1_is_old_state(true),
  stateVarsAreAllocated(false)
{}

void 
Albany::StateManager::registerStateVariable(
  const std::string &name,
  const Teuchos::RCP< PHX::DataLayout > &dl) 
{
  TEST_FOR_EXCEPT(stateVarsAreAllocated);

  statesToStore[name] = dl;
}

void 
Albany::StateManager::allocateStateVariables(const int numWorksets)
{
  TEST_FOR_EXCEPT(stateVarsAreAllocated);
  stateVarsAreAllocated=true;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());


  //Outer container is a vector over worksets
  state1.resize(numWorksets);
  state2.resize(numWorksets);

    // For each workset, loop over registered states
  RegisteredStates::iterator st = statesToStore.begin();

  *out << std::endl;
  while (st != statesToStore.end()) {

    std::vector<PHX::DataLayout::size_type> dims;
    st->second->dimensions(dims);

    *out << "StateManager: allocating space for state:  " << st->first << std::endl;

    for (int ws=0; ws<numWorksets; ws++) {
      state1[ws][st->first] = Teuchos::rcp(new Intrepid::FieldContainer<RealType>(dims));
      state2[ws][st->first] = Teuchos::rcp(new Intrepid::FieldContainer<RealType>(dims));
    }
    st++;
  }
  *out << std::endl;
}

Teuchos::RCP<const Albany::StateVariables> 
Albany::StateManager::getOldStateVariables(const int ws) const
{
  if (statesToStore.empty()) return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state) return Teuchos::rcp(&state1[ws],false);
  else                     return Teuchos::rcp(&state2[ws],false);
}

Teuchos::RCP<Albany::StateVariables> 
Albany::StateManager::getNewStateVariables(const int ws)
{
  if (statesToStore.empty()) return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state) return Teuchos::rcp(&state2[ws],false);
  else                     return Teuchos::rcp(&state1[ws],false);
}

void 
Albany::StateManager::updateStates()
{
  if (statesToStore.empty()) return;

  // Swap boolean that defines old and new (in terms of state1 and 2) in accessors
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  state1_is_old_state = !state1_is_old_state;
}
