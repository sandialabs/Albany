/*
 * FELIX_Temperature.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_TEMPERATURE_HPP_
#define FELIX_TEMPERATURE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Temperature

    This evaluator computes the temperature from the enthalpy
*/

template<typename EvalT, typename Traits, typename Type>
class Temperature: public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;
  typedef typename EvalT::ScalarT ScalarT;

  Temperature (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<Type,Cell,Node> meltingTemp;
  PHX::MDField<Type,Cell,Node> enthalpyHs;
  PHX::MDField<ScalarT,Cell,Node> enthalpy;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> temperature;
  PHX::MDField<ScalarT,Cell,Node> diffEnth;
  PHX::MDField<ScalarT,Cell,Node> tempIce;

  int numNodes;

  double c_i, rho_i;
  double T0;
};

} // Namespace FELIX

#endif /* FELIX_TEMPERATURE_HPP_ */
