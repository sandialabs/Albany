//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef CURRENTCOORDS_HPP
#define CURRENTCOORDS_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Compute the current coordinates

**/

template <typename EvalT, typename Traits>
class CurrentCoords : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  CurrentCoords(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> refCoords;
  PHX::MDField<const ScalarT, Cell, Vertex, Dim>     displacement;

  // Output:
  PHX::MDField<ScalarT, Cell, Vertex, Dim> currentCoords;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numDims;
};
}  // namespace LCM

#endif
