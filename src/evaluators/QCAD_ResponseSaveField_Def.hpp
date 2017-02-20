//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "Adapt_NodalDataVector.hpp"

class QCAD::ResponseSaveFieldManager : public Adapt::NodalDataBase::Manager {
public:
  ResponseSaveFieldManager () : nwrkr_(0), prectr_(0), postctr_(0) {}

  void registerWorker () { ++nwrkr_; }
  int nWorker () const { return nwrkr_; }

  void initCounters () { prectr_ = postctr_ = 0; }
  int incrPreCounter () { return ++prectr_; }
  int incrPostCounter () { return ++postctr_; }
  
private:
  int nwrkr_, prectr_, postctr_;
};

template<typename EvalT, typename Traits>
QCAD::ResponseSaveField<EvalT, Traits>::
ResponseSaveField(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  weights("Weights", dl->qp_scalar)
{
  //! Register with state manager
  pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");

  const std::string key = "ResponseSaveField" + PHX::typeAsString<EvalT>();
  Teuchos::RCP<Adapt::NodalDataBase>
    ndb = pStateMgr->getNodalDataBase();
  if (ndb->isManagerRegistered(key))
    mgr_ = Teuchos::rcp_dynamic_cast<ResponseSaveFieldManager>(
      ndb->getManager(key));
  else {
    mgr_ = Teuchos::rcp(new ResponseSaveFieldManager());
    ndb->registerManager(key, mgr_);
  }
  mgr_->registerWorker();

  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! User-specified parameters
  if(plist->isParameter("Vector Field Name")) {
    fieldName = plist->get<std::string>("Vector Field Name");
    isVectorField = true;
  }
  else {
    fieldName = plist->get<std::string>("Field Name");
    isVectorField = false;
  }
  stateName = plist->get<std::string>("State Name", fieldName);
  outputToExodus = plist->get<bool>("Output to Exodus", true);
  outputCellAverage = plist->get<bool>("Output Cell Average", true);
  memoryHolderOnly = plist->get<bool>("Memory Placeholder Only", false);
  vectorOp = plist->get<std::string>("Vector Operation", "magnitude");
  fieldIndices = plist->get<std::string>("Field Indices", "Cell,QuadPt");

  Teuchos::RCP<PHX::DataLayout> cell_dl = dl->cell_scalar;
  Teuchos::RCP<PHX::DataLayout> scalar_dl, vector_dl;

  if(fieldIndices == "Cell,QuadPt") {
    //! number of quad points per cell and dimension
    scalar_dl = dl->qp_scalar;
    vector_dl = dl->qp_vector;
    numQPs = vector_dl->dimension(1);
    numDims = vector_dl->dimension(2);
  }
  else if(fieldIndices == "Cell,Node") {
    //! number of nodes per cell and dimension
    scalar_dl = dl->node_scalar;
    vector_dl = dl->node_vector;
    numNodes = vector_dl->dimension(1);
    numDims = vector_dl->dimension(2);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			       "Invalid value of the 'Field Indices' parameter: " << fieldIndices 
			       << ". Allowed values are 'Cell,QuadPt' and 'Cell,Node'.");
  }
 
  //! add dependent fields
  Teuchos::RCP<PHX::DataLayout>& field_dl = isVectorField ? vector_dl : scalar_dl;
  PHX::MDField<ScalarT> f(fieldName, field_dl);  field = f;
  this->addDependentField(field);
  this->addDependentField(weights);

  if(fieldIndices == "Cell,QuadPt") { //register a cell,qp state => cell-valued quantity
    if( outputCellAverage ) {
      pStateMgr->registerStateVariable(stateName, cell_dl, "ALL", "scalar", 0.0, false, outputToExodus);
    }
    else {
      pStateMgr->registerStateVariable(stateName, scalar_dl, "ALL", "scalar", 0.0, false, outputToExodus);
    }
  }
  else if(fieldIndices == "Cell,Node") {
    TEUCHOS_TEST_FOR_EXCEPTION(isVectorField, std::logic_error, "Vector-valued Cell,Node fields are not supported yet");
    pStateMgr->registerNodalVectorStateVariable(stateName, dl->node_node_scalar, dl->dummy, "all", "scalar", 0.0, false, outputToExodus);
  }

  // Create field tag
  response_field_tag = 
    Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName + " Save Field Response",
				       dl->dummy));
  this->addEvaluatedField(*response_field_tag);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(weights,fm);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  const int ctr = mgr_->incrPreCounter();
  const bool am_first = ctr == 1;
  if ( ! am_first) return;

  if(fieldIndices == "Cell,Node") {
    Teuchos::RCP<Adapt::NodalDataVector> node_data =
      pStateMgr->getStateInfoStruct()->getNodalDataBase()
      ->getNodalDataVector();
    node_data->initializeVectors(0.0);
  }
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  using Albany::ADValue;

  const std::size_t iX=0; //index for x coordinate
  const std::size_t iY=1; //index for y coordinate
  const std::size_t iZ=2; //index for z coordinate

  //Don't do anything if this response is just used to allocate 
  // and hold a block of memory (the state)
  if(memoryHolderOnly) return;

  if(fieldIndices == "Cell,QuadPt") {
    // Get shards Array (from STK) for this state
    // Need to check if we can just copy full size -- can assume same ordering?
    Albany::MDArray sta = (*workset.stateArrayPtr)[stateName];
    std::vector<PHX::DataLayout::size_type> dims;
    sta.dimensions(dims);
    int size = dims.size();

    if(!isVectorField) {
      switch (size) {  //Note: size should always == 2 now: qp_scalar type or cell_scalar state registered
      case 2:     
	for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
	  if( outputCellAverage ) {
	    double integral = 0, vol = 0;
	    for (std::size_t qp = 0; qp < numQPs; ++qp) {
	      integral += ADValue(field(cell,qp)) * ADValue(weights(cell,qp));
	      vol += ADValue(weights(cell, qp));
	    }
	    sta(cell,(std::size_t)0) = integral / vol;
	  }
	  else {
	    for (std::size_t qp = 0; qp < numQPs; ++qp)
	      sta(cell, qp) = ADValue(field(cell,qp));
	  }
	}
	break;
	/*case 3:     
	  for (int cell = 0; cell < dims[0]; ++cell)
	  for (int qp = 0; qp < dims[1]; ++qp)
	  for (int i = 0; i < dims[2]; ++i)
	  sta(cell, qp, i) = field(cell,qp,i);
	  break;
	  case 4:     
	  for (int cell = 0; cell < dims[0]; ++cell)
	  for (int qp = 0; qp < dims[1]; ++qp)
	  for (int i = 0; i < dims[2]; ++i)
	  for (int j = 0; j < dims[3]; ++j)
	  sta(cell, qp, i, j) = field(cell,qp,i,j);
	  break;
	*/
      default:
	TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
				   "Unexpected dimensions in SaveField response Evaluator: " << size);
      }
    }
    else {
      ScalarT t;
      switch (size) {
      case 2:     
	for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
	  
	  ScalarT stateValue = 0.0;
	  double vol = 0.0;
	  if( outputCellAverage ) sta(cell,(std::size_t)0) = 0.0;
	  
	  for (std::size_t qp = 0; qp < numQPs; ++qp) {
	    t = 0.0;
	  
	    if(vectorOp == "magnitude") {
	      for (std::size_t i = 0; i < numDims; ++i)
		t += field(cell,qp,i)*field(cell,qp,i);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "xyMagnitude") {
	      if(numDims > iX) t += field(cell,qp,iX)*field(cell,qp,iX);
	      if(numDims > iY) t += field(cell,qp,iY)*field(cell,qp,iY);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "xzMagnitude") {
	      if(numDims > iX) t += field(cell,qp,iX)*field(cell,qp,iX);
	      if(numDims > iZ) t += field(cell,qp,iZ)*field(cell,qp,iZ);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "yzMagnitude") {
	      if(numDims > iY) t += field(cell,qp,iY)*field(cell,qp,iY);
	      if(numDims > iZ) t += field(cell,qp,iZ)*field(cell,qp,iZ);
	      stateValue = sqrt(t);
	    }

	    else if(vectorOp == "magnitude2") {
	      for (std::size_t i = 0; i < numDims; ++i)
		t += field(cell,qp,i)*field(cell,qp,i);
	      stateValue = t;
	    }
	    else if(vectorOp == "xyMagnitude2") {
	      if(numDims > iX) t += field(cell,qp,iX)*field(cell,qp,iX);
	      if(numDims > iY) t += field(cell,qp,iY)*field(cell,qp,iY);
	      stateValue = t;
	    }
	    else if(vectorOp == "xzMagnitude2") {
	      if(numDims > iX) t += field(cell,qp,iX)*field(cell,qp,iX);
	      if(numDims > iZ) t += field(cell,qp,iZ)*field(cell,qp,iZ);
	      stateValue = t;
	    }
	    else if(vectorOp == "yzMagnitude2") {
	      if(numDims > iY) t += field(cell,qp,iY)*field(cell,qp,iY);
	      if(numDims > iZ) t += field(cell,qp,iZ)*field(cell,qp,iZ);
	      stateValue = t;
	    }
	    
	    else if(vectorOp == "xCoord") {
	      if(numDims > iX) stateValue = field(cell,qp,iX);
	    }
	    else if(vectorOp == "yCoord") {
	      if(numDims > iY) stateValue = field(cell,qp,iY);
	    }
	    else if(vectorOp == "zCoord") {
	      if(numDims > iZ) stateValue = field(cell,qp,iZ);
	    }
	    else {
	      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
					 "Unknown vector operation: " << vectorOp);
	    }
	    
	    if( outputCellAverage ) {
	      sta(cell, (std::size_t)0) += ADValue(stateValue) * ADValue(weights(cell,qp));
	      vol += ADValue(weights(cell,qp));
	    }
	    else sta(cell, qp) = ADValue(stateValue);
	  }
	  
	  if( outputCellAverage ) sta(cell,(std::size_t)0) /= vol;
	}
	break;
      default:
	TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
				   "Unexpected dimensions in SaveField response Evaluator: " << size);
      }
    }
  } // end of Cell,QuadPt

  else if(fieldIndices == "Cell,Node") {

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataVector> node_data =
      pStateMgr->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();
    const Teuchos::RCP<Tpetra_MultiVector>& data = node_data->getLocalNodeVector();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Tpetra_Map> local_node_map = node_data->getLocalMap();

    int node_var_offset, node_var_ndofs; // offset into MultiVector of vector corresponding to stateName, (ndofs not used)
    node_data->getNDofsAndOffset(stateName, node_var_offset,  node_var_ndofs);

    if(!isVectorField) {
      int size = 2; //HACK - size always == 2 now since we assume Cell,Node
      switch (size) {  //Note: size should always == 2 now: node_scalar type or cell_scalar state registered
      case 2: 
	std::cout << "DEBUG: QCAD::ResponseSaveField is saving nodal " << fieldName << " nCells = " << workset.numCells << ", nNodes=" 
		  << numNodes << ", EBName = " << workset.EBName << std::endl;

	for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
	  for (std::size_t node = 0; node < numNodes; ++node) {
	    //sta(cell, node) = ADValue(field(cell,node));
	    const GO global_row = wsElNodeID[cell][node];
	    if ( ! local_node_map->isNodeGlobalElement(global_row)) continue;
	    data->sumIntoGlobalValue(global_row, node_var_offset, ADValue(field(cell,node)) );
	    //data->sumIntoGlobalValue(global_row, node_var_offset, 10.0 ); //DEBUG - to get weighting correct

	    //std::cout << "DEBUG: Saving nodal " <<  fieldName << "(" << global_row << ") from local (" 
	    //      << cell << "," << node << ") = " << field(cell,node) << std::endl;
	  }
	}
	break;
      default:
	TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
				   "Unexpected dimensions in SaveField response Evaluator: " << size);
      }
    }
    /*else {
      ScalarT t;
      switch (size) {
      case 2:     
	for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
	  
	  ScalarT stateValue = 0.0;
	  for (std::size_t node = 0; node < numNodes; ++node) {
	    t = 0.0;
	    
	    if(vectorOp == "magnitude") {
	      for (std::size_t i = 0; i < numDims; ++i)
		t += field(cell,node,i)*field(cell,node,i);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "xyMagnitude") {
	      if(numDims > iX) t += field(cell,node,iX)*field(cell,node,iX);
	      if(numDims > iY) t += field(cell,node,iY)*field(cell,node,iY);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "xzMagnitude") {
	      if(numDims > iX) t += field(cell,node,iX)*field(cell,node,iX);
	      if(numDims > iZ) t += field(cell,node,iZ)*field(cell,node,iZ);
	      stateValue = sqrt(t);
	    }
	    else if(vectorOp == "yzMagnitude") {
	      if(numDims > iY) t += field(cell,node,iY)*field(cell,node,iY);
	      if(numDims > iZ) t += field(cell,node,iZ)*field(cell,node,iZ);
	      stateValue = sqrt(t);
	    }
	    
	    else if(vectorOp == "magnitude2") {
	      for (std::size_t i = 0; i < numDims; ++i)
		t += field(cell,node,i)*field(cell,node,i);
	      stateValue = t;
	    }
	    else if(vectorOp == "xyMagnitude2") {
	      if(numDims > iX) t += field(cell,node,iX)*field(cell,node,iX);
	      if(numDims > iY) t += field(cell,node,iY)*field(cell,node,iY);
	      stateValue = t;
	    }
	    else if(vectorOp == "xzMagnitude2") {
	      if(numDims > iX) t += field(cell,node,iX)*field(cell,node,iX);
	      if(numDims > iZ) t += field(cell,node,iZ)*field(cell,node,iZ);
	      stateValue = t;
	    }
	    else if(vectorOp == "yzMagnitude2") {
	      if(numDims > iY) t += field(cell,node,iY)*field(cell,node,iY);
	      if(numDims > iZ) t += field(cell,node,iZ)*field(cell,node,iZ);
	      stateValue = t;
	    }
	    
	    else if(vectorOp == "xCoord") {
	      if(numDims > iX) stateValue = field(cell,node,iX);
	    }
	    else if(vectorOp == "yCoord") {
	      if(numDims > iY) stateValue = field(cell,node,iY);
	    }
	    else if(vectorOp == "zCoord") {
	      if(numDims > iZ) stateValue = field(cell,node,iZ);
	    }
	    else {
	      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
					 "Unknown vector operation: " << vectorOp);
	    }	    
	    
	    sta(cell, node) = ADValue(stateValue);
	  }
	  
	}
	break;
      default:
	TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
				   "Unexpected dimensions in SaveField response Evaluator: " << size);
      }
    } */
  } //end of Cell,Node
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  const int ctr = mgr_->incrPostCounter();
  const bool am_last = ctr == mgr_->nWorker();
  if ( ! am_last) return;
  mgr_->initCounters();

  if(fieldIndices == "Cell,Node") {

    // Get the node data block container.
    Teuchos::RCP<Adapt::NodalDataVector> node_data =
        pStateMgr->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();
  
    // Export the data from the local to overlapped decomposition.
    node_data->initializeExport();
    node_data->exportAddNodalDataVector();
  
    const Teuchos::RCP<Tpetra_MultiVector>& data = node_data->getOverlapNodeVector();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Tpetra_Map> overlap_node_map = node_data ->getOverlapMap();
  
    const int num_nodes = overlap_node_map->getNodeNumElements();
    const int blocksize = node_data->getVecSize();
  
    /*// Get weight info.
    int node_weight_offset;
    int node_weight_ndofs;
    node_data->getNDofsAndOffset(
        this->nodal_weights_name_,
        node_weight_offset,
        node_weight_ndofs);*/
    // XXX Divide the overlap field through by the weights.
    //Teuchos::ArrayRCP<const ST> weights = data->getData(node_weight_offset);
  
    /*int node_var_offset;
    int node_var_ndofs;
    node_data->getNDofsAndOffset(
  			       stateName,
  			       node_var_offset,
  			       node_var_ndofs);
  
    for (int k = 0; k < node_var_ndofs; ++k) {
      Teuchos::ArrayRCP<ST> v = data->getDataNonConst(node_var_offset + k);
      for (LO overlap_node = 0; overlap_node < num_nodes; ++overlap_node)
        v[overlap_node] *= 1.0; //weights[overlap_node];
	}*/
  
    // Store the overlapped vector data back in stk in the field "field_name".
    node_data->saveNodalDataState();
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseSaveField<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseSaveField Params"));;

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Field to save");
  validPL->set<std::string>("Vector Field Name", "", "Vector field to save");
  validPL->set<std::string>("Vector Operation", "magnitude", "How to convert vector to scalar value, e.g., magnitude, xyMagnitude, xCoord");
  validPL->set<std::string>("Field Indices", "Cell,QuadPt", "How the field is indexed.  Allowed values are Cell,QuadPt and Cell,Node");
  validPL->set<std::string>("State Name", "<Field Name>", "State name to save field as");
  validPL->set<bool>("Output to Exodus", true, "Whether state should be output in STK dump to exodus");
  validPL->set<bool>("Output Cell Average", true, "Whether cell average or all quadpoint data should be output to exodus");
  validPL->set<bool>("Memory Placeholder Only", false, "True if data should not actually be transferred to this state, i.e., the state is just used as a memory container and should not be overwritten when responses are computed");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

