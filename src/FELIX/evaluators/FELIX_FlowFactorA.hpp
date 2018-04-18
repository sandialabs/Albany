//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_FLOW_FACTOR_A_HPP
#define FELIX_FLOW_FACTOR_A_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, bool ThermoCoupled>
class FlowFactorA : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  FlowFactorA (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<ThermoCoupled, ScalarT, ParamScalarT>::type TempScalarT;

  // Input:
  PHX::MDField<const ParamScalarT, Cell>  given_flow_factor;
  PHX::MDField<const TempScalarT, Cell>   temperature;
  PHX::MDField<const ScalarT, Dim>        flowFactorParam;

  // Output:
  PHX::MDField<TempScalarT, Cell> flowFactor;

  enum FlowFactorType {UNIFORM, GIVEN_FIELD, TEMPERATURE_BASED};
  FlowFactorType flowFactor_type;
};

} // Namespace FELIX

#endif // FELIX_FLOW_FACTOR_A_HPP
