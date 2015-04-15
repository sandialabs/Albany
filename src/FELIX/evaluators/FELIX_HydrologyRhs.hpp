//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_RHS_HPP
#define FELIX_HYDROLOGY_RHS_HPP 1

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
class HydrologyRhs : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  HydrologyRhs (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu_i;
  PHX::MDField<ScalarT,Cell,QuadPoint> h;
  PHX::MDField<ScalarT,Cell,QuadPoint> H;
  PHX::MDField<ScalarT,Cell,QuadPoint> z_b;
  PHX::MDField<ScalarT,Cell,QuadPoint> u_b;
  PHX::MDField<ScalarT,Cell,QuadPoint> beta;
  PHX::MDField<ScalarT,Cell,QuadPoint> omega;
  PHX::MDField<ScalarT,Cell,QuadPoint> G;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> rhs;

  unsigned int numQPs;

  ScalarT R;

  double mu_w;
  double rho_i;
  double rho_w;
  double rho_combo;
  double L;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RHS_HPP
