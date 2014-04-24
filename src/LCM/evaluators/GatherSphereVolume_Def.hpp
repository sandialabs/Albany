//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
GatherSphereVolume<EvalT, Traits>::
GatherSphereVolume(const Teuchos::ParameterList& p,
		   const Teuchos::RCP<Albany::Layouts>& dl) :
  sphereVolume  (p.get<std::string> ("Sphere Volume Name"), dl->node_scalar ),
  numVertices(0), worksetSize(0)
{  
  this->addEvaluatedField(sphereVolume);
  this->setName("Gather Sphere Volume"+PHX::TypeString<EvalT>::value);
}

template<typename EvalT, typename Traits>
GatherSphereVolume<EvalT, Traits>::
GatherSphereVolume(const Teuchos::ParameterList& p) :
  sphereVolume(p.get<std::string> ("Sphere Volume Name"),p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") ),
  numVertices(0), worksetSize(0)
{
  this->addEvaluatedField(sphereVolume);
  this->setName("Gather Sphere Volume"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherSphereVolume<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphereVolume,fm);

  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Vertex>::size_type > dims;
  sphereVolume.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSphereVolume<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<double> wsSphereVolume = workset.wsSphereVolume;

  TEUCHOS_TEST_FOR_EXCEPTION(wsSphereVolume.is_null(), std::logic_error, "\n****Error:  Sphere Volume field not found in GatherSphereVolume evaluator!\n");

  for (std::size_t cell=0; cell < numCells; ++cell) {
    sphereVolume(cell) = wsSphereVolume[cell];
  }

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
    sphereVolume(cell) = sphereVolume(0);
  }
}
}
