//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
#define LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology Basal Potential

    This evaluator evaluates the basal potential phi = \rho_w * g * z_b at the basal side
*/

template<typename EvalT, typename Traits, bool IsStokes>
class BasalGravitationalWaterPotential : public PHX::EvaluatorWithBaseImpl<Traits>,
                                         public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  BasalGravitationalWaterPotential (const Teuchos::ParameterList& p,
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

  // Output:
  PHX::MDField<ParamScalarT>  phi_0;

  std::string basalSideName;

  unsigned int numNodes;

  double rho_w;
  double g;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
