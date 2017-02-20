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
    typedef typename EvalT::ParamScalarT ParamScalarT;
    // Input:
    PHX::MDField<Type,Cell,Node> meltingTemp; //[K]
    PHX::MDField<ParamScalarT,Cell,Node> surfaceTemp; //[K]

    // Output:
    PHX::MDField<Type,Cell,Node> enthalpyHs;  //[MW s m^{-3}]
    PHX::MDField<ParamScalarT,Cell,Node> surfaceEnthalpy;  //[MW s m^{-3}]

    int numNodes;

    double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
    double rho_i; //[kg m^{-3}]
    double T0;    //[K]
  };

} // Namespace FELIX


#endif /* FELIX_PRESSUREMELTINGENTHALPY_HPP_ */
