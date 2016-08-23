//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
SaveCellStateField<EvalT, Traits>::
SaveCellStateField(const Teuchos::ParameterList& p)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save Cell State Field");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveCellStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveCellStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************
// **********************************************************************
template<typename Traits>
SaveCellStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveCellStateField(const Teuchos::ParameterList& p) :
  weights(p.get<std::string>("Weights Name"), p.get<Teuchos::RCP<PHX::DataLayout> >("Weights Layout"))
{

  i_index = 0; j_index = 0; k_index = 0;
  if(p.isType<int>("component i")) {i_index = p.get<int>("component i");}
  if(p.isType<int>("component j")) {j_index = p.get<int>("component j");}
  if(p.isType<int>("component k")) {k_index = p.get<int>("component k");}

  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");
  PHX::MDField<ScalarT> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Field Layout") );
  field = f;

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>
    (stateName, p.get< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout")));

  this->addDependentField(weights);
  this->addDependentField(field);
  this->addEvaluatedField(*savestate_operation);

  this->setName("Save Field " + fieldName +" to Cell State " + stateName + "Residual");
}

// **********************************************************************
template<typename Traits>
void SaveCellStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(weights,fm);
}
// **********************************************************************
template<typename Traits>
void SaveCellStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Get shards Array (from STK) for this state
  // Need to check if we can just copy full size -- can assume same ordering?
    Albany::StateArray::const_iterator it;
    it = workset.stateArrayPtr->find(stateName);

    TEUCHOS_TEST_FOR_EXCEPTION((it == workset.stateArrayPtr->end()), std::logic_error,
           std::endl << "Error: cannot locate " << stateName << " in PHAL_SaveCellStateField_Def" << std::endl);

    Albany::MDArray sta = it->second;

    std::vector<int> dims;
    field.dimensions(dims);
    int size = dims.size();
    int numCells = workset.numCells;
    int numQPs = dims[1];
    
   
    double el_weight;

    switch (size) {
    case 1:
      for (int cell = 0; cell < numCells; ++cell)
        sta(cell) = field(cell);
      break;
    case 2:
      for (int cell = 0; cell < numCells; ++cell){
        sta(cell, 0) = 0.0;
        el_weight = 0.0;
	for (int qp = 0; qp < numQPs; ++qp){
	  sta(cell, 0) += weights(cell,qp)*field(cell,qp);
          el_weight += weights(cell,qp);
        }
        sta(cell, 0) /= el_weight;
      }
      break;
    case 3:
      for (int cell = 0; cell < numCells; ++cell){
        sta(cell, 0) = 0.0;
        el_weight = 0.0;
	for (int qp = 0; qp < numQPs; ++qp){
	  sta(cell, 0) += weights(cell,qp)*field(cell,qp,i_index);
          el_weight += weights(cell,qp);
        }
        sta(cell, 0) /= el_weight;
      }
      break;
    case 4:
      for (int cell = 0; cell < numCells; ++cell){
        sta(cell, 0) = 0.0;
        el_weight = 0.0;
	for (int qp = 0; qp < numQPs; ++qp){
          sta(cell, 0) += weights(cell,qp)*field(cell,qp,i_index,j_index);
          el_weight += weights(cell,qp);
        }
        sta(cell, 0) /= el_weight;
      }
      break;
    case 5:
      for (int cell = 0; cell < numCells; ++cell){
        sta(cell, 0) = 0.0;
        el_weight = 0.0;
	for (int qp = 0; qp < numQPs; ++qp){
          sta(cell, 0) += weights(cell,qp)*field(cell,qp,i_index,j_index,k_index);
          el_weight += weights(cell,qp);
        }
        sta(cell, 0) /= el_weight;
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPT_MSG(size<1||size>5,
                          "Unexpected Array dimensions in SaveCellStateField: " << size);
    }
}

}
