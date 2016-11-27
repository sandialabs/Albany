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

    This evaluator computes the temperature from enthalpy
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
    PHX::MDField<Type,Cell,Node> 		meltingTemp; //[K]
    PHX::MDField<Type,Cell,Node> 		enthalpyHs;  //[MW s m^{-3}]
    PHX::MDField<ScalarT,Cell,Node> 	enthalpy;  //[MW s m^{-3}]

    // Output:
    PHX::MDField<ScalarT,Cell,Node> 	temperature; //[K]
    PHX::MDField<ScalarT,Cell,Node> 	diffEnth;    //[MW s m^{-3}]

    int numNodes;

    double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
    double rho_i; //[kg m^{-3}]
    double T0;    //[K]
  };

} // Namespace FELIX

#endif /* FELIX_TEMPERATURE_HPP_ */
