//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_HYDROSTATIC_POTENTIAL_HPP
#define FELIX_HYDROLOGY_HYDROSTATIC_POTENTIAL_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits>
class HydrologyHydrostaticPotential : public PHX::EvaluatorWithBaseImpl<Traits>,
                                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologyHydrostaticPotential (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ParamScalarT,Cell,QuadPoint>      z_s;
  PHX::MDField<ParamScalarT,Cell,QuadPoint>      H;

  // Output:
  PHX::MDField<ParamScalarT,Cell,QuadPoint>      phi_H;

  int numQPs;

  double rho_i;
  double rho_w;
  double g;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_HYDROSTATIC_POTENTIAL_HPP
