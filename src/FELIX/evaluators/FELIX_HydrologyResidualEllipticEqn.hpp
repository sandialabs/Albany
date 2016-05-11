//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_RESIDUAL_ELLIPTIC_EQN_HPP
#define FELIX_HYDROLOGY_RESIDUAL_ELLIPTIC_EQN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology ResidualEllipticEqn Evaluator

    This evaluator evaluates the residual of the Hydrology model (quasi-static formulation)
*/

template<typename EvalT, typename Traits>
class HydrologyResidualEllipticEqn : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  HydrologyResidualEllipticEqn (const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ScalarT     ScalarT;

  // Input:
  PHX::MDField<RealType>        BF;
  PHX::MDField<RealType>        GradBF;
  PHX::MDField<MeshScalarT>     w_measure;
  PHX::MDField<ScalarT>         N;
  PHX::MDField<ScalarT>         q;
  PHX::MDField<ScalarT>         h;
  PHX::MDField<ScalarT>         m;
  PHX::MDField<ScalarT>         omega;
  PHX::MDField<ScalarT>         u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;

  int numNodes;
  int numQPs;
  int numDims;

  double mu_w;
  double rho_combo;
  double h_r;
  double l_r;
  double n;
  double A;

  // Variables necessary for stokes coupling
  bool                            stokes_coupling;
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RESIDUAL_ELLIPTIC_EQN_HPP
