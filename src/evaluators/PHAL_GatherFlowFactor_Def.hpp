//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherFlowFactor<EvalT, Traits>::
GatherFlowFactor(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  flowFactor  (p.get<std::string> ("Flow Factor Name"), dl->cell_scalar2 ),
  worksetSize(0)
{  
  this->addEvaluatedField(flowFactor);
  this->setName("Gather Flow Factor"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherFlowFactor<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(flowFactor,fm);

  typename std::vector< typename PHX::template MDField<ScalarT,Cell>::size_type > dims;
  flowFactor.dimensions(dims); //get dimensions

  worksetSize = dims[0];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherFlowFactor<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<double> wsFlowFactor = workset.wsFlowFactor;

  for (std::size_t cell=0; cell < numCells; ++cell)
        flowFactor(cell) = wsFlowFactor[cell];

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell)
        flowFactor(cell) = flowFactor(0);
}
}
