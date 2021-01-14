//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
#define LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP 1

#include "Albany_Layouts.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/** \brief Hydrology Basal Potential

    This evaluator evaluates the basal potential phi = \rho_w * g * z_b at the basal side
*/

template<typename EvalT, typename Traits>
class BasalGravitationalWaterPotential : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  BasalGravitationalWaterPotential (const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const ParamScalarT>  H;
  PHX::MDField<const ParamScalarT>  z_s;

  // Output:
  PHX::MDField<ParamScalarT>  phi_0;

  bool eval_on_side;

  int numPts;

  double rho_w;
  double g;

  std::string sideSetName; // Only needed if eval_on_side=true
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
