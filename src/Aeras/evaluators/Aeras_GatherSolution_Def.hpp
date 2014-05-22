//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace Aeras {

template<typename EvalT, typename Traits>
GatherCoordinateVector<EvalT, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec  (p.get<std::string> ("Coordinate Vector Name"), dl->node_3vector ),
  numNodes(0), numCoords(0), worksetSize(0)
{  
  this->addEvaluatedField(coordVec);
  this->setName("Aeras::GatherCoordinateVector"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherCoordinateVector<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numNodes = dims[1];
  numCoords = dims[2];

}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateVector<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t i=0; i < numCoords; ++i) { 
        coordVec(cell,node,i) = wsCoords[cell][node][i]; 
      }
    }
  }

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t i=0; i < numCoords; ++i) { 
        coordVec(cell,node,i) = coordVec(0,node,i); 
      }
    }
  }
}

}
