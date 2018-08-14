//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TLPOROPLASTICITYRESIDMOMENTUM_HPP
#define TLPOROPLASTICITYRESIDMOMENTUM_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief

    This evaluator calculate residual of the mass balance equation
    for the poromechanics problem.

*/

template <typename EvalT, typename Traits>
class TLPoroPlasticityResidMomentum
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  TLPoroPlasticityResidMomentum(const Teuchos::ParameterList& p);

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
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      TotalStress;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      defgrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                J;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;

  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>      uDotDot;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> wBF;

  // Output:
  PHX::MDField<ScalarT, Cell, Node, Dim> ExResidual;

  int  worksetSize;
  int  numNodes;
  int  numQPs;
  int  numDims;
  bool enableTransient;

  // Work space FCs
  Kokkos::DynRankView<ScalarT, PHX::Device> F_inv;
  Kokkos::DynRankView<ScalarT, PHX::Device> F_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> JF_invT;
};
}  // namespace LCM

#endif
