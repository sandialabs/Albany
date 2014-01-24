//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_GATHER_COORDINATEVECTOR_HPP
#define AERAS_GATHER_COORDINATEVECTOR_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace Aeras {
/** \brief Gathers Coordinates values from the Newton coordinates vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the coordinates
    names vector.

*/

template<typename EvalT, typename Traits> 
class GatherCoordinateVector : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  GatherCoordinateVector(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherCoordinateVector(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
 
  bool  periodic;
  std::size_t worksetSize;
  std::size_t numNodes;
  std::size_t numCoords;
};
}

#endif
