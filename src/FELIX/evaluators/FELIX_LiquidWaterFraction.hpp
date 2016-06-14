/*
 * FELIX_LiquidWaterFraction.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_LIQUIDWATERFRACTION_HPP_
#define FELIX_LIQUIDWATERFRACTION_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Liquid Water Fraction

    This evaluator computes the liquid water fraction
*/

template<typename EvalT, typename Traits, typename Type>
class LiquidWaterFraction: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  //typedef typename EvalT::ParamScalarT ParamScalarT;
  typedef typename EvalT::ScalarT ScalarT;

  LiquidWaterFraction (const Teuchos::ParameterList& p,
               	   	   const Teuchos::RCP<Albany::Layouts>& dl_basal);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<Type,Cell,Side,Node> enthalpyHs;
  PHX::MDField<ScalarT,Cell,Side,Node> enthalpy;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,Node> omega;

  std::string sideSetName;
  int numSideNodes;

  double L;
};

} // Namespace FELIX

#endif /* FELIX_LIQUIDWATERFRACTION_HPP_ */
