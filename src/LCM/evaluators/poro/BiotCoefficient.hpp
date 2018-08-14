//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef BIOT_COEFFICIENT_HPP
#define BIOT_COEFFICIENT_HPP

#include "Albany_config.h"

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"

namespace LCM {
/**
 * \brief Evaluates Biot's Coefficient, either as a constant or a truncated
 * KL expansion.
 */

template <typename EvalT, typename Traits>
class BiotCoefficient : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  BiotCoefficient(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(const std::string& n);

 private:
  int                                                   numQPs;
  int                                                   numDims;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coordVec;
  PHX::MDField<ScalarT, Cell, QuadPoint>                biotCoefficient;

  //! Is Biot Coefficient constant, or random field
  bool is_constant;

  //! Constant value
  ScalarT constant_value;

  //! Optional dependence on Bulk modulus of the skeleton (Kskeleton) and the
  //! solid grains (Kgrain)  (B = 1 - K/K_{s}). Notice that K can be dependent
  //! of temperature;
  // PHX::MDField<ScalarT,Cell,QuadPoint> porePressure;

  bool    isPoroElastic;
  ScalarT Kskeleton_value;
  ScalarT Kgrain_value;

#ifdef ALBANY_STOKHOS
  //! Exponential random field
  Teuchos::RCP<Stokhos::KL::ExponentialRandomField<RealType>> exp_rf_kl;
#endif

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;
};
}  // namespace LCM

#endif
