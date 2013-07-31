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
GatherCoordinateFromSolutionVector<EvalT, Traits>::
GatherCoordinateFromSolutionVector(const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector),
  solutionVec(p.get<std::string> ("Solution Names"), dl->node_vector),
  numVertices(0), numNodes(0), numDim(0), worksetSize(0) {
  this->addEvaluatedField(coordVec);
  this->addDependentField(solutionVec);
  this->setName("Gather Coordinate From Solution Vector" + PHX::TypeString<EvalT>::value);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateFromSolutionVector<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(solutionVec, fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT, Cell, Vertex, Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
  numDim = dims[2];

  typename std::vector< typename PHX::template MDField<MeshScalarT, Cell, Node, Dim>::size_type > n_dims;
  solutionVec.dimensions(n_dims); //get dimensions
  numNodes = n_dims[1];

  TEUCHOS_TEST_FOR_EXCEPTION(numNodes != numVertices,
                             std::logic_error,
                             "Error in GatherCoordinateFromSolutionVector: specification of coordinate vector vs. solution layout is incorrect."
                             << std::endl);


}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateFromSolutionVector<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset) {
  unsigned int numCells = workset.numCells;
  //  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for(std::size_t cell = 0; cell < worksetSize; ++cell) {
    for(std::size_t node = 0; node < numVertices; ++node) {
      for(std::size_t eq = 0; eq < numDim; ++eq) {
        coordVec(cell, node, eq) = solutionVec(cell, node, eq);
      }
    }
  }
}

}
