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

/** \brief Hydrology ResidualPotentialEqn Evaluator

    This evaluator evaluates the residual of the Hydrology model (quasi-static formulation)
*/

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
class HydrologyResidualPotentialEqn;

// Partial specialization for Hydrology problem
template<typename EvalT, typename Traits, bool HasThicknessEqn>
class HydrologyResidualPotentialEqn<EvalT,Traits,HasThicknessEqn,false> :
      public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename EvalT::ScalarT       ScalarT;

  typedef typename std::conditional<HasThicknessEqn,ScalarT,ParamScalarT>::type hScalarT;

  HydrologyResidualPotentialEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const RealType,Cell,Node,QuadPoint>      BF;
  PHX::MDField<const RealType,Cell,Node,QuadPoint,Dim>  GradBF;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint>        w_measure;
  PHX::MDField<const ScalarT,Cell,QuadPoint>            N;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim>        q;
  PHX::MDField<const hScalarT,Cell,QuadPoint>           h;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>       m;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>       omega;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>       u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;

  int numNodes;
  int numQPs;
  int numDims;

  double mu_w;
  double rho_combo;
  double h_r;
  double l_r;
  double A;
};

// Partial specialization for StokesFO coupling
template<typename EvalT, typename Traits, bool HasThicknessEqn>
class HydrologyResidualPotentialEqn<EvalT,Traits,HasThicknessEqn,true> :
        public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename EvalT::ScalarT       ScalarT;

  typedef typename std::conditional<HasThicknessEqn,ScalarT,ParamScalarT>::type hScalarT;

  HydrologyResidualPotentialEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const RealType,Cell,Side,Node,QuadPoint>       BF;
  PHX::MDField<const RealType,Cell,Side,Node,QuadPoint,Dim>   GradBF;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>         w_measure;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>             N;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint,Dim>         q;
  PHX::MDField<const hScalarT,Cell,Side,QuadPoint>            h;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>             m;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>             omega;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>             u_b;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> residual;

  int numNodes;
  int numQPs;
  int numDims;

  double mu_w;
  double rho_combo;
  double h_r;
  double l_r;
  double A;

  // Variables necessary for stokes coupling
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_RESIDUAL_ELLIPTIC_EQN_HPP
