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

    This evaluator computes the enthalpy of the ice at the pressure-melting temperature Tm(p).
*/

template<typename EvalT, typename Traits, typename Type>
class PressureMeltingEnthalpy: public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  //typedef typename EvalT::ParamScalarT ParamScalarT;
  //typedef typename EvalT::MeshScalarT MeshScalarT;

  PressureMeltingEnthalpy (const Teuchos::ParameterList& p,
                       	   const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<Type,Cell,QuadPoint> meltingTemp;

  // Output:
  PHX::MDField<Type,Cell,QuadPoint> enthalpyHs;

  int numQPs;

  double c_i;
  double T0;
};

} // Namespace FELIX



#endif /* FELIX_PRESSUREMELTINGENTHALPY_HPP_ */
