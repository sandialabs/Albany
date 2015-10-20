//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_RESIDUAL_EVOLUTION_EQN_H_HPP
#define FELIX_HYDROLOGY_RESIDUAL_EVOLUTION_EQN_H_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Mass Equation Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits>
class HydrologyResidualEvolutionEqn : public PHX::EvaluatorWithBaseImpl<Traits>,
                                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  HydrologyResidualEvolutionEqn (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ScalarT     ScalarT;

  // Input:
  PHX::MDField<RealType>      BF;
  PHX::MDField<MeshScalarT>   w_measure;
  PHX::MDField<ScalarT>       h;
  PHX::MDField<ScalarT>       h_dot;
  PHX::MDField<ScalarT>       N;
  PHX::MDField<ScalarT>       m;
  PHX::MDField<ScalarT>       u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;

  int numNodes;
  int numQPs;
  int numDims;

  double rho_i;
  double h_r;
  double l_r;
  double A;
  double n;

  // Variables necessary for stokes coupling
  bool                            stokes_coupling;
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RESIDUAL_EVOLUTION_EQN_H_HPP
