//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_GATHER_SOLUTION_HPP
#define AERAS_GATHER_SOLUTION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"

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
class ScatterSolution : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  ScatterSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  ScatterSolution(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim> > val;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim> > val_dot;
  std::size_t numNodes;

  int numFields;
  int numLevels;

  std::size_t worksetSize;
};
}

#endif
