/*
 * FELIX_HydrostaticPressure.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_HYDROSTATICPRESSURE_HPP_
#define FELIX_HYDROSTATICPRESSURE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrostatic pressure

    This evaluator evaluates the hydrostatic approximation of the pressure to compute the pressure-melting point Tm(p)
*/

template<typename EvalT, typename Traits, typename Type>
class HydrostaticPressure : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  //typedef typename EvalT::ParamScalarT ParamScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HydrostaticPressure (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,Dim> z;
  PHX::MDField<const Type,Cell,Node> s; //surface height

  // Output:
  PHX::MDField<Type,Cell,Node> pressure;

  int numNodes;

  double rho_i;
  double g;
  double p_atm;
};

} // Namespace FELIX

#endif /* FELIX_HYDROSTATICPRESSURE_HPP_ */
