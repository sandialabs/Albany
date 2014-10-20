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
GatherSurfaceVelocity<EvalT, Traits>::
GatherSurfaceVelocity(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  surfaceVelocity  (p.get<std::string> ("Surface Velocity Name"), dl->node_vector ),
  numVertices(0), worksetSize(0), numVecDim(0)
{  
  this->addEvaluatedField(surfaceVelocity);
  this->setName("Gather Surface Velocity"+PHX::TypeString<EvalT>::value);
}

template<typename EvalT, typename Traits>
GatherSurfaceVelocity<EvalT, Traits>::
GatherSurfaceVelocity(const Teuchos::ParameterList& p) :
	surfaceVelocity  (p.get<std::string> ("Surface Velocity Name"),p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") ),
  numVertices(0), worksetSize(0), numVecDim(0)
{
	this->addEvaluatedField(surfaceVelocity);
	this->setName("Gather Surface Velocity"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherSurfaceVelocity<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(surfaceVelocity,fm);

  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node, VecDim>::size_type > dims;
  surfaceVelocity.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
  numVecDim = dims[2];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSurfaceVelocity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsSurfaceVelocity = workset.wsSurfaceVelocity;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
      for (std::size_t dim = 0; dim < numVecDim; ++dim) {
        surfaceVelocity(cell,node,dim) = wsSurfaceVelocity[cell][node][dim];
      }
    }
  }


  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
      for (std::size_t dim = 0; dim < numVecDim; ++dim) {
        surfaceVelocity(cell,node,dim) = surfaceVelocity(0,node,dim );
      }
    }
  }
}

}
