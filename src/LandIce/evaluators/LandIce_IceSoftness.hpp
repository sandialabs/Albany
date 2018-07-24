//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_ICE_SOFTNESS_HPP
#define LANDICE_ICE_SOFTNESS_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, bool ThermoCoupled>
class IceSoftness : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  IceSoftness (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<ThermoCoupled, ScalarT, ParamScalarT>::type TempScalarT;

  // Input:
  PHX::MDField<const ParamScalarT, Cell>  given_ice_softness;
  PHX::MDField<const TempScalarT, Cell>   temperature;
  double A;

  // Output:
  PHX::MDField<TempScalarT, Cell> ice_softness;

  enum IceSoftnessType {UNIFORM, GIVEN_FIELD, TEMPERATURE_BASED};
  IceSoftnessType ice_softness_type;
};

} // Namespace LandIce

#endif // LANDICE_ICE_SOFTNESS_HPP
