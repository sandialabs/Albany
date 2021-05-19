//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDRAULIC_POTENTIAL_HPP
#define LANDICE_HYDRAULIC_POTENTIAL_HPP 1

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/** \brief Ice overburden

    This evaluator evaluates the basal potential phi = \rho_w * g * z_b at the basal side
*/

template<typename EvalT, typename Traits>
class HydraulicPotential : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydraulicPotential (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluatePotential(unsigned int cell);

  // Input:
  PHX::MDField<const ScalarT>   P_w;
  PHX::MDField<const RealType>  phi_0;
  PHX::MDField<const ScalarT>   h;

  // Output:
  PHX::MDField<ScalarT>         phi;

  bool eval_on_side;
  std::string sideSetName;  // Only used if eval_on_side=true

  Albany::LocalSideSetInfo sideSet;

  unsigned int numPts;
  unsigned int worksetSize;

  bool use_h;

  double rho_w;
  double g;
};

} // Namespace LandIce

#endif // LANDICE_HYDRAULIC_POTENTIAL_HPP
