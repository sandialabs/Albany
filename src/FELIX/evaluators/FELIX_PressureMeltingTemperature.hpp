/*
 * FELIX_PressureMeltingTemperature.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_PRESSUREMELTINGTEMPERATURE_HPP_
#define FELIX_PRESSUREMELTINGTEMPERATURE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Pressure-melting temperature

    This evaluator computes the pressure-melting temperature Tm(p) via the hydrostatic approximation of the pressure.
*/

template<typename EvalT, typename Traits, typename Type>
class PressureMeltingTemperature: public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  //typedef typename EvalT::ParamScalarT ParamScalarT;
  //typedef typename EvalT::MeshScalarT MeshScalarT;

  PressureMeltingTemperature (const Teuchos::ParameterList& p,
                       	   	  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<Type,Cell,Node> pressure;

  // Output:
  PHX::MDField<Type,Cell,Node> meltingTemp;

  int numNodes;

  double beta;
};

} // Namespace FELIX

#endif /* FELIX_PRESSUREMELTINGTEMPERATURE_HPP_ */
