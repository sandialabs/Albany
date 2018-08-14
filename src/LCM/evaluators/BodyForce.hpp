//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(BODY_FORCE_HPP)
#define BODY_FORCE_HPP

#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {

//
// Body force evaluator
//
template <typename EvalT, typename Traits>
class BodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  BodyForce(Teuchos::ParameterList& p, Teuchos::RCP<Albany::Layouts> dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  int num_qp_{0};

  int num_dim_{0};

  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coordinates_;

  PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights_;
  // PHX::MDField<const ScalarT, Cell> density_;
  const RealType density_;

  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> body_force_;

  bool is_constant_{true};

  Teuchos::Array<RealType> constant_value_;

  Teuchos::Array<RealType> rotation_center_;

  Teuchos::Array<RealType> rotation_axis_;

  RealType angular_frequency_;
};

}  // namespace LCM

#endif  // BODY_FORCE_HPP
