//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_RESIDUAL_MASS_EQN_HPP
#define LANDICE_HYDROLOGY_RESIDUAL_MASS_EQN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

/** \brief Hydrology ResidualMassEqn Evaluator

    This evaluator evaluates the residual of the mass conservation for the Hydrology model
*/

/*
 *  The mass conservation equation has the following (strong) form
 *
 *     dh/dt + div(q) = m/rho_w + omega
 *
 *  where q is the water discharge, h the water thickness, rho_w is the water density,
 *  m the melting rate of the ice (due to geothermal flow and sliding), and omega
 *  is  the water source (water reaching the bed from the surface, through crevasses)
 */


template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
class HydrologyResidualMassEqn : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename EvalT::ScalarT       ScalarT;

  typedef typename std::conditional<IsStokesCoupling,ScalarT,ParamScalarT>::type  uScalarT;
  typedef typename std::conditional<ThermoCoupled,ScalarT,ParamScalarT>::type     tScalarT;

  HydrologyResidualMassEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const RealType>      BF;
  PHX::MDField<const MeshScalarT>   GradBF;
  PHX::MDField<const MeshScalarT>   w_measure;
  PHX::MDField<const ScalarT>       q;
  PHX::MDField<const ScalarT>       m;
  PHX::MDField<const ParamScalarT>  omega;
  PHX::MDField<const ScalarT>       h_dot;
  PHX::MDField<const ScalarT>       h_till_dot;
  PHX::MDField<const ScalarT>       P_w;  // Water pressure (for penalization)
  PHX::MDField<const ParamScalarT>  P_o;  // Overburden (for penalization)

  // Input only needed if equation is on a sideset
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>   metric;

  // Output:
  PHX::MDField<ScalarT>       residual;

  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;

  double rho_w;
  double scaling_omega;
  double scaling_q;
  double scaling_h_dot;
  double penalization_coeff;

  bool mass_lumping;
  bool penalization;
  bool use_melting;
  bool unsteady;
  bool has_h_till;

  // Variables necessary for stokes coupling
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_RESIDUAL_MASS_EQN_HPP
