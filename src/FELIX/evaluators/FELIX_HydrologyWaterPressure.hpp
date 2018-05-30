//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_WATER_PRESSURE_HPP
#define FELIX_HYDROLOGY_WATER_PRESSURE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Ice overburden

    This evaluator evaluates the basal potential phi = \rho_w * g * z_b at the basal side
*/

template<typename EvalT, typename Traits, bool IsStokes>
class HydrologyWaterPressure : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologyWaterPressure (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const ParamScalarT>  H;
  PHX::MDField<const ParamScalarT>  z_s;
  PHX::MDField<const ScalarT>       phi;
  PHX::MDField<const ScalarT>       h;

  // Output:
  PHX::MDField<ScalarT>             P_w;

  std::string basalSideName;  // Only if IsStokes  is true

  int numPts;

  bool use_h;

  double rho_w;
  double g;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_WATER_PRESSURE_HPP
