//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
#define LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP 1

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

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

  void evaluatePotential(unsigned int cell);

  // Input:
  PHX::MDField<const RealType>  H;
  PHX::MDField<const RealType>  z_s;

  // Output:
  PHX::MDField<RealType>  phi_0;

  bool eval_on_side;

  Albany::LocalSideSetInfo sideSet;

  int numPts;
  unsigned int worksetSize;

  double rho_w;
  double g;

  std::string sideSetName; // Only needed if eval_on_side=true
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_BASAL_GRAVITATIONAL_WATER_POTENTIAL_HPP
