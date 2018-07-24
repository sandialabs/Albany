//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP
#define LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology Mass Equation Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

/*
 *  The (water) thickness equation has the following (strong) form
 *
 *     dh/dt = m/rho_i + (h_r-h)*|u_b|/l_r - c_creep*A*h*N^3
 *
 *  where h is the water thickness, m the melting rate of the ice,
 *  h_r/l_r typical height/length of bed bumps, u_b the sliding
 *  velocity of the ice, A is the ice softness, N is the
 *  effective pressure, and c_creep is a tuning coefficient.
 *  Also, dh/dt denotes the *partial* time derivative.
 */

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
class HydrologyResidualCavitiesEqn : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT         ScalarT;
  typedef typename EvalT::MeshScalarT     MeshScalarT;
  typedef typename EvalT::ParamScalarT    ParamScalarT;

  typedef typename std::conditional<IsStokes,ScalarT,ParamScalarT>::type       IceScalarT;
  typedef typename std::conditional<ThermoCoupled,ScalarT,ParamScalarT>::type  TempScalarT;

  HydrologyResidualCavitiesEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const RealType>      BF;
  PHX::MDField<const MeshScalarT>   w_measure;
  PHX::MDField<const ScalarT>       h;
  PHX::MDField<const ScalarT>       h_node;
  PHX::MDField<const ScalarT>       h_dot;
  PHX::MDField<const ScalarT>       P_dot;
  PHX::MDField<const ScalarT>       N;
  PHX::MDField<const ScalarT>       m;
  PHX::MDField<const IceScalarT>    u_b;
  PHX::MDField<const TempScalarT>   ice_softness;

  // Output:
  PHX::MDField<ScalarT>             residual;

  int numNodes;
  int numQPs;

  double rho_i;
  double phi0;
  double h_r;
  double l_r;
  double c_creep;
  double scaling_h_t;
  double penalization_coeff;

  bool unsteady;
  bool has_p_dot;
  bool use_melting;
  bool nodal_equation;
  bool penalization;

  // Variables necessary for stokes coupling
  bool                            stokes_coupling;
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP
