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
#include "Teuchos_TestForException.hpp"

Albany::StateManager::StateManager() :
  state1_is_old_state(true), stateVarsAreAllocated(false)
{
  //Teuchos::RCP<PHX::DataLayout> dummy = Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const int saveOrLoadStateFieldID,
                                            const std::string &init_type)
{
  return registerStateVariable(name, dl, dummy, saveOrLoadStateFieldID, init_type, name);
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::StateManager::registerStateVariable(const std::string &stateName, const Teuchos::RCP<PHX::DataLayout> &dl,
                                            const Teuchos::RCP<PHX::DataLayout> &dummy,
                                            const int saveOrLoadStateFieldID,
                                            const std::string &init_type, const std::string& fieldName)
{
  TEST_FOR_EXCEPT(stateVarsAreAllocated);

  statesToStore[stateName] = dl;
  stateInit[stateName] = init_type;

  // Create param list for SaveStateField evaluator 
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Save or Load State " 
							  + stateName + " to/from field " + fieldName));
  p->set<const int>("Type", saveOrLoadStateFieldID);
  p->set<const std::string>("State Name", stateName);
  p->set<const std::string>("Field Name", fieldName);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("State Field Layout", dl);
  p->set<const Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout", dummy);
  return p;
}

void
Albany::StateManager::registerStateVariable(const std::string &name, const Teuchos::RCP<PHX::DataLayout> &dl,
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
  stateVarsAreAllocated = true;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //Outer container is a vector over worksets
  state1.resize(numWorksets);
  state2.resize(numWorksets);

  // For each workset, loop over registered states
  RegisteredStates::iterator st = statesToStore.begin();

  *out << std::endl;
  while (st != statesToStore.end())
  {

    std::vector<PHX::DataLayout::size_type> dims;
    st->second->dimensions(dims);

    *out << "StateManager: allocating space for state:  " << st->first << std::endl;

    for (int ws = 0; ws < numWorksets; ws++)
    {
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
  stateVarsAreAllocated = true;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // For each workset, loop over registered states
  InitializationType::iterator st = stateInit.begin();

  *out << std::endl;
  while (st != stateInit.end())
  {

    std::string init_type = st->second;

    *out << "StateManager: initializing state:  " << st->first << std::endl;

    for (int ws = 0; ws < numWorksets; ws++)
    {
      if (init_type == "zero")
      {
        if (ws == 0)
          *out << "initialization type 'zero'" << std::endl;
        state1[ws][st->first]->initialize(0.0);
        state2[ws][st->first]->initialize(0.0);
      }
      else if (init_type == "identity")
      {
        if (ws == 0)
          *out << "initialization type 'identity'" << std::endl;
        // we assume operating on the last two indices is correct
        std::vector<PHX::DataLayout::size_type> dims;
        state1[ws][st->first]->dimensions(dims);

        int size = dims.size();
        TEST_FOR_EXCEPTION(size != 4, std::logic_error,
            "Something is wrong during identity state variable initialization");
        int cells = dims[0];
        int qps = dims[1];
        int dim = dims[2];
        int dim2 = dims[3];

        TEST_FOR_EXCEPT( ! (dim == dim2) );

        for (int cell = 0; cell < cells; ++cell)
        {
          for (int qp = 0; qp < qps; ++qp)
          {
            for (int i = 0; i < dim; ++i)
            {
              (*state1[ws][st->first])(cell, qp, i, i) = 1.0;
              (*state2[ws][st->first])(cell, qp, i, i) = 1.0;
            }
          }
        }
      }
    }
    st++;
  }
  *out << std::endl;
}

void
Albany::StateManager::reinitializeStateVariables(Teuchos::RCP<std::vector<StateVariables> >& 
						 stateVarsToCopyFrom, const int numWorksets)
{
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  stateVarsAreAllocated = true;

  if( stateVarsToCopyFrom == Teuchos::null ) return;

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  *out << std::endl;
  for (int ws = 0; ws < numWorksets; ws++)
  {
    std::vector<StateVariables>& vec = *stateVarsToCopyFrom;
    StateVariables& stateVarsForWorkset = vec[ws];
    StateVariables::iterator st;

    for( st = stateVarsForWorkset.begin(); st != stateVarsForWorkset.end(); st++)
    {
      std::string stateName = st->first;

      if( state1[ws].find(stateName) != state1[ws].end() ) {
	if(ws == 0) *out << "StateManager: reinitializing state:  " << st->first << std::endl;

        // we assume operating on the last two indices is correct
        std::vector<PHX::DataLayout::size_type> dims;
        state1[ws][st->first]->dimensions(dims);

        int size = dims.size();
        TEST_FOR_EXCEPTION(size != 2, std::logic_error,
	   "Something is wrong during identity state variable reinitialization: size = " << size);
        int cells = dims[0];
        int qps = dims[1];
        //int dim = dims[2];
        //int dim2 = dims[3];

        //TEST_FOR_EXCEPT( ! (dim == dim2) );

        for (int cell = 0; cell < cells; ++cell)
        {
          for (int qp = 0; qp < qps; ++qp)
          {
            //for (int i = 0; i < dim; ++i)
            //{
            //  (*state1[ws][st->first])(cell, qp, i, i) = (*(st->second))(cell, qp, i, i);
	    //  (*state2[ws][st->first])(cell, qp, i, i) = (*(st->second))(cell, qp, i, i);
	    //}
	    (*state1[ws][st->first])(cell, qp) = (*(st->second))(cell, qp);
	    (*state2[ws][st->first])(cell, qp) = (*(st->second))(cell, qp);
          }
        }

	//TODO - 3,4 dimensions
      }
      else {
	if(ws == 0) *out << "StateManager: state " << st->first << " not present, so not reinitialized" << std::endl;
      }
    }
  }
  *out << std::endl;
}

Teuchos::RCP<const Albany::StateVariables>
Albany::StateManager::getOldStateVariables(const int ws) const
{
  if (statesToStore.empty())
    return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state)
    return Teuchos::rcp(&state1[ws], false);
  else
    return Teuchos::rcp(&state2[ws], false);
}

Teuchos::RCP<Albany::StateVariables>
Albany::StateManager::getNewStateVariables(const int ws)
{
  if (statesToStore.empty())
    return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state)
    return Teuchos::rcp(&state2[ws], false);
  else
    return Teuchos::rcp(&state1[ws], false);
}


//ANDY - make const?
Teuchos::RCP<std::vector<Albany::StateVariables> >
Albany::StateManager::getAllOldStateVariables()
{
  if (statesToStore.empty())
    return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state)
    return Teuchos::rcp(&state1, false);
  else
    return Teuchos::rcp(&state2, false);
}

Teuchos::RCP<std::vector<Albany::StateVariables> >
Albany::StateManager::getAllNewStateVariables()
{
  if (statesToStore.empty())
    return Teuchos::null;

  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  if (state1_is_old_state)
    return Teuchos::rcp(&state2, false);
  else
    return Teuchos::rcp(&state1, false);
}


void
Albany::StateManager::updateStates()
{
  if (statesToStore.empty())
    return;

  // Swap boolean that defines old and new (in terms of state1 and 2) in accessors
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  state1_is_old_state = !state1_is_old_state;
}

const std::vector<std::vector<double> >
Albany::StateManager::getElementAveragedStates()
{
  // we will be filling up states with QP averaged element quantities
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);
  std::vector<std::vector<double> > states;

  // here we are getting a pointer to the current state
  // (we don't need to output oldState
  std::vector<StateVariables>* stateVarPtr;
  if (state1_is_old_state)
    stateVarPtr = &state2;
  else
    stateVarPtr = &state1;

  // exit if we aren't storing any states
  int numStates = (*stateVarPtr)[0].size();
  if (numStates == 0)
    return states;

  // the number of worksets being used
  int numWorksets = (*stateVarPtr).size();

  // StateVariables is a map(string,RCP(FieldContainer))
  // iterate over each state and get the dimensions of the FC
  StateVariables::iterator it = (*stateVarPtr)[0].begin();
  std::vector<std::vector<int> > dims(numStates);
  int p = 0;
  while (it != (*stateVarPtr)[0].end())
  {
    it->second->dimensions(dims[p]);
    p++;
    it++;
  }

  // JTO: containersize and numQP won't/shouldn't depend on the type of state var
  int containerSize = dims[0][0];
  int numQP = dims[0][1];

  // since we will be writing out a flat vector of states, we need to figure
  // out the size of that vector, each state will be a number of scalars,
  // totaling numScalarStates
  int numScalarStates(0);
  int numDim, stateRank;
  for (int i = 0; i < numStates; ++i)
  {
    stateRank = dims[i].size();
    switch (stateRank)
    {
    case 2:
      numDim = -1;
      numScalarStates += 1;
      break;
    case 3:
      numDim = dims[i][2];
      numScalarStates += numDim;
      break;
    case 4:
      numDim = dims[i][2];
      numScalarStates += numDim * numDim;
      break;
    default:
      TEST_FOR_EXCEPTION(true, std::logic_error,
          "state manager postprocessing logic error")
      ;
    }
  }



  std::vector<int> worksetSizes(numWorksets);
  std::vector<int> worksetOffsets(numWorksets);

  if(disc != Teuchos::null) {
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > wsElNodeID;
    wsElNodeID = disc->getWsElNodeID();
    for(int ws=0; ws<numWorksets; ++ws) 
      worksetSizes[ws] = wsElNodeID[ws].size();
  }
  else { 
    //All worksets = container size (mostly for back compat?).
    //This means:
    // flattenedSize = numWorksets * containerSize and
    // worksetOffets[i] = i*containerSize
    for(int ws=0; ws<numWorksets; ++ws) 
      worksetSizes[ws] = containerSize; 
  }
    
  int flattenedSize = 0;
  for(int ws=0; ws<numWorksets; ++ws) {
    worksetOffsets[ws] = flattenedSize;
    flattenedSize += worksetSizes[ws];
  }

  // resize the vector<vector<double> >
  states.resize(flattenedSize);
  for (int i = 0; i < flattenedSize; i++)
    states[i].resize(numScalarStates);

  // Average states over QPs and store. Separate logic for scalar,vector,tensor
  for (int i = 0; i < numWorksets; i++)
  {
    StateVariables::iterator it = (*stateVarPtr)[i].begin();
    int index = 0;
    int numCells;

    while (it != (*stateVarPtr)[i].end())
    {
      const Intrepid::FieldContainer<RealType>& fc = *(it->second);
      stateRank = fc.rank();
      containerSize = fc.dimension(0);
      numQP = fc.dimension(1);
      numCells = worksetSizes[i];

      switch (stateRank)
      {
      case 2: //scalar
        for (int j = 0; j < numCells; ++j)
          for (int k = 0; k < numQP; ++k)
            states[worksetOffsets[i] + j][index + 0] += fc(j, k) / numQP;
        index++;
        break;
      case 3: //vector
        numDim = fc.dimension(2);
        for (int j = 0; j < numCells; ++j)
          for (int k = 0; k < numQP; ++k)
            for (int l = 0; l < numDim; ++l)
              states[worksetOffsets[i] + j][index + l] += fc(j, k, l) / numQP;
        index += numDim;
        break;
      case 4: //tensor
        numDim = fc.dimension(2);
        assert(fc.dimension(2) == fc.dimension(3));
        for (int j = 0; j < numCells; ++j)
          for (int k = 0; k < numQP; ++k)
            for (int l = 0; l < numDim; ++l)
              for (int m = 0; m < numDim; ++m)
                states[worksetOffsets[i] + j][index + m + l * numDim] += fc(j, k, l, m) / numQP;
        index += numDim * numDim;
        break;
      }
      it++;
    }
  }
  return states;
}




void
Albany::StateManager::saveVectorAsState(const std::string& stateName, const Epetra_Vector& vec)
{
  // we will be filling up states with QP averaged element quantities
  TEST_FOR_EXCEPT(!stateVarsAreAllocated);

  // make sure we have discretization object
  TEST_FOR_EXCEPT(disc == Teuchos::null); 

  // the number of worksets being used
  int numWorksets = state1.size(); //state2 should be the same size

  //Get field container for first workset of desired state
  Intrepid::FieldContainer<RealType>& fc = *(state1[0][stateName]);

  std::vector<int> dims; // size of field containter for first workset. We 
  fc.dimensions(dims);   //  assume numQP and numDim is the same for all worksets

  //int containerSize = dims[0];
  std::size_t numNodes = dims[1];
  int stateRank = dims.size();
  TEST_FOR_EXCEPT(!(stateRank == 2)); //save Epetra vector to a scalar field only

  //Set state1 and state2 with given Epetra vector data
  // NOTE: these states must have node-type data layouts; can we enforce this?
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > wsElNodeID;
  wsElNodeID = disc->getWsElNodeID();

  for (int i = 0; i < numWorksets; i++) {
    std::size_t numCells = wsElNodeID[i].size();
    for (std::size_t cell=0; cell < numCells; ++cell ) {
      const Teuchos::ArrayRCP<int>& nodeID = wsElNodeID[i][cell];
    
      for(std::size_t node =0; node < numNodes; ++node) {
	int offsetIntoVec = nodeID[node];
	(*(state1[i][stateName]))(cell,node) = vec[offsetIntoVec];
	(*(state2[i][stateName]))(cell,node) = vec[offsetIntoVec];
      }
    }
  }
}
