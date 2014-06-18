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
GatherSHeight<EvalT, Traits>::
GatherSHeight(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  sHeight  (p.get<std::string> ("Surface Height Name"), dl->node_scalar ),
  numVertices(0), worksetSize(0)
{  
  this->addEvaluatedField(sHeight);
  this->setName("Gather Surface Height" );
}

template<typename EvalT, typename Traits>
GatherSHeight<EvalT, Traits>::
GatherSHeight(const Teuchos::ParameterList& p) :
	sHeight  (p.get<std::string> ("Surface Height Name"),p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") ),
  numVertices(0), worksetSize(0)
{
	this->addEvaluatedField(sHeight);
	this->setName("Gather Surface Height" );
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherSHeight<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sHeight,fm);

  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Vertex>::size_type > dims;
  sHeight.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSHeight<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > wsSHeight = workset.wsSHeight;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
        sHeight(cell,node) = wsSHeight[cell][node];
      }
    }


  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
        sHeight(cell,node) = sHeight(0,node);
    }
  }
}
}
