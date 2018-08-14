//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef DEFGRAD_HPP
#define DEFGRAD_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief Deformation Gradient

 This evaluator computes the deformation gradient

 */

template <typename EvalT, typename Traits>
class DefGrad : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  DefGrad(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> GradU;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>       weights;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
  PHX::MDField<ScalarT, Cell, QuadPoint>           J;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  //! flag to compute the weighted average of J
  bool weightedAverage;

  //! stabilization parameter for the weighted average
  ScalarT alpha;
};
}  // namespace LCM
#endif
