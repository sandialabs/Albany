/*
 * FELIX_PressureMeltingEnthalpy.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_PRESSUREMELTINGENTHALPY_HPP_
#define FELIX_PRESSUREMELTINGENTHALPY_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Pressure-melting enthalpy

    This evaluator computes enthalpy of the ice at pressure-melting temperature Tm(p).
*/

template<typename EvalT, typename Traits, typename Type>
class PressureMeltingEnthalpy: public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  PressureMeltingEnthalpy (const Teuchos::ParameterList& p,
                       	   const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<Type,Cell,Node> meltingTemp;

  // Output:
  PHX::MDField<Type,Cell,Node> enthalpyHs;

  int numNodes;

  double c_i, rho_i, T0;
};

} // Namespace FELIX


#endif /* FELIX_PRESSUREMELTINGENTHALPY_HPP_ */
