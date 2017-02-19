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

template<typename EvalT, typename Traits, bool IsStokes>
class HydrologyResidualThicknessEqn;

// Partial specialization for the hydrology only problem
template<typename EvalT, typename Traits>
class HydrologyResidualThicknessEqn<EvalT,Traits,false> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  HydrologyResidualThicknessEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT         ScalarT;
  typedef typename EvalT::MeshScalarT     MeshScalarT;
  typedef typename EvalT::ParamScalarT    ParamScalarT;

  // Input:
  PHX::MDField<const RealType,Cell,Node,QuadPoint>  BF;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint>    w_measure;
  PHX::MDField<const ScalarT,Cell,QuadPoint>        h;
  PHX::MDField<const ScalarT,Cell,QuadPoint>        h_dot;
  PHX::MDField<const ScalarT,Cell,QuadPoint>        N;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>   m;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>   u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;
  PHX::MDField<ScalarT,Cell,QuadPoint>        h_dot_eval;

  int numNodes;
  int numQPs;
  int numDims;

  double rho_i;
  double h_r;
  double l_r;
  double A;
  double n;

  bool unsteady;
};

// Partial specialization for StokesFO coupling
template<typename EvalT, typename Traits>
class HydrologyResidualThicknessEqn<EvalT,Traits,true> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  HydrologyResidualThicknessEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ScalarT     ScalarT;

  // Input:
  PHX::MDField<const RealType,Cell,Side,Node,QuadPoint>  BF;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>    w_measure;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>        h;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>        h_dot;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>        N;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>        m;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>        u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>        h_dot_eval;

  int numNodes;
  int numQPs;
  int numDims;

  double rho_i;
  double h_r;
  double l_r;
  double A;
  double n;

  bool unsteady;

  // Variables necessary for stokes coupling
  bool                            stokes_coupling;
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RESIDUAL_EVOLUTION_EQN_H_HPP
