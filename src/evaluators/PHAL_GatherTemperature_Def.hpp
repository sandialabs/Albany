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
GatherTemperature<EvalT, Traits>::
GatherTemperature(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  temperature  (p.get<std::string> ("Temperature Name"), dl->cell_scalar ),
  numVertices(0), numDim(0), worksetSize(0)
{  
  this->addEvaluatedField(temperature);
  this->setName("Gather Temperature"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherTemperature<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(temperature,fm);

  typename std::vector< typename PHX::template MDField<ScalarT,Cell>::size_type > dims;
  temperature.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numDim = dims[1];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherTemperature<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<double> wsTemperature = workset.wsTemperature;

  for (std::size_t cell=0; cell < numCells; ++cell)
        temperature(cell) = wsTemperature[cell];

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell)
        temperature(cell) = temperature(0);
}
}
