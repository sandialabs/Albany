//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_FLOW_RATE_HPP
#define LANDICE_FLOW_RATE_HPP 1

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, typename TempST>
class FlowRate : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;

  FlowRate (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const RealType,Cell> given_flow_rate;
  PHX::MDField<const TempST,Cell> temperature;

  // Output:
  PHX::MDField<TempST,Cell> flowRate;

  double A;
  enum FlowRateType {UNIFORM, GIVEN_FIELD, TEMPERATURE_BASED};
  FlowRateType flowRate_type;
};

} // Namespace LandIce

#endif // LANDICE_FLOW_RATE_HPP
