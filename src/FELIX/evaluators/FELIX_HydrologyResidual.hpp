//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_RESIDUAL_HPP
#define FELIX_HYDROLOGY_RESIDUAL_HPP 1

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
class HydrologyResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  HydrologyResidual (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> phi;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradPhi;
  PHX::MDField<ScalarT,Cell,QuadPoint> mu_i;
  PHX::MDField<ScalarT,Cell,QuadPoint> h;
  PHX::MDField<ScalarT,Cell,QuadPoint> rhs;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;

  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;

  ScalarT k_0;

  double mu_w;
  double rho_i;
  double rho_w;
  double rho_combo;
  double L;
  double nonlin_coeff;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RESIDUAL_HPP
