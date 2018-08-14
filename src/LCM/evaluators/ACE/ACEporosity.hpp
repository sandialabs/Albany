//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEporosity_hpp)
#define ACEporosity_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates porosity at integration points
///
template <typename EvalT, typename Traits>
class ACEporosity : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>,
                    public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ACEporosity(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  /// Calculates depth-dependent porosity
  void
  evaluateFields(typename Traits::EvalData workset);

  /// Gets the input deck entered values
  ScalarT&
  getValue(const std::string& n);

 private:
  int num_qps_{0};
  int num_dims_{0};

  // contains the depth-dependent porosity value
  PHX::MDField<ScalarT, Cell, QuadPoint> porosity_;

  // parameters to calculate porosity
  ScalarT surface_porosity_{0.75};
  ScalarT efolding_depth_{10.0};
  ScalarT constant_value_{1.0};

  bool is_constant_{false};
};
}  // namespace LCM

#endif  // ACEporosity_hpp
