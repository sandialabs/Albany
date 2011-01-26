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
  const Teuchos::RCP< PHX::DataLayout > &dl,
  const std::string &init_type) 
{
  TEST_FOR_EXCEPT(stateVarsAreAllocated);

  statesToStore[name] = dl;
  stateInit[name] = init_type;
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

void 
Albany::StateManager::initializeStateVariables(const int numWorksets)
{
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  stateVarsAreAllocated=true;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // For each workset, loop over registered states
  InitializationType::iterator st = stateInit.begin();

  *out << std::endl;
  while (st != stateInit.end()) {

    std::string init_type = st->second;

    *out << "StateManager: initializing state:  " << st->first << std::endl;

    for (int ws=0; ws<numWorksets; ws++) {
      if ( init_type == "zero" )
      {
	std::cout << "initialization type 'zero'" << std::endl;
	state1[ws][st->first]->initialize(0.0);
	state2[ws][st->first]->initialize(0.0);
      }
      else if ( init_type == "identity" )
      {
	std::cout << "initialization type 'identity'" << std::endl;
	// we assume operating on the last two indices is correct
	std::vector<PHX::DataLayout::size_type> dims;
	state1[ws][st->first]->dimensions(dims);

	int size = dims.size();
	TEST_FOR_EXCEPTION(size != 4, std::logic_error, 
			   "Something is wrong during identity state variable initialization");
	int cells = dims[0];
	int qps   = dims[1];
	int dim   = dims[2];
	int dim2  = dims[3];

	TEST_FOR_EXCEPT( ! (dim == dim2) );
	
	for (int cell = 0; cell < cells; ++cell)
	{
	  for (int qp = 0; qp < qps; ++qp)
	  {
	    for (int i = 0; i < dim; ++i)
	    {
	      (*state1[ws][st->first])(cell,qp,i,i) = 1.0;
	      (*state2[ws][st->first])(cell,qp,i,i) = 1.0;
	    }
	  }
	}
      }
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

const std::vector<std::vector<double> >
Albany::StateManager::getElementAveragedStates() {
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  std::vector<std::vector<double> > states;

  std::vector<StateVariables>& stateVariables = state1;
  if (state1_is_old_state)  stateVariables = state2;

  int numStates = stateVariables[0].size();
  if (numStates==0) return states;

  int numWorksets = stateVariables.size();

  int containerSize = stateVariables[0].begin()->second->dimension(0);
  int numQP  = stateVariables[0].begin()->second->dimension(1);
  int numDim  = stateVariables[0].begin()->second->dimension(2);
  int numDim2 = stateVariables[0].begin()->second->dimension(3);
  int numScalarStates = numDim * numDim2; // 2D stress tensor

  states.resize(numWorksets*containerSize);
  for (int i=0; i<numWorksets*containerSize;i++)  states[i].resize(numScalarStates);

  for (int i=0; i< numWorksets; i++) {
    const Intrepid::FieldContainer<RealType>& fc = *(stateVariables[i].begin()->second);
    for (int j=0; j< containerSize; j++) {
      for (int k=0; k< numQP; k++) {
        for (int l=0; l< numDim; l++) {
          for (int m=0; m< numDim2; m++) {
             states[i*containerSize + j][m+l*numDim] += fc(j,k,l,m)/numQP;
          }
        }
      }
    }
  }
  return states;
}
