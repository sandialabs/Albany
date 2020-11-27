//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_RESIDUAL_TILL_STORAGE_EQN_HPP
#define LANDICE_HYDROLOGY_RESIDUAL_TILL_STORAGE_EQN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology ResidualTillStorageEqn Evaluator

    This evaluator evaluates the residual of the till water storage for the Hydrology model
*/

/*
 *  The till water strorage equation has the following (strong) form
 *
 *     dh_till/dt = m/rho_w + omega - C_drain
 *
 *  where h_till the till water storage thickness, rho_w is the water density,
 *  m the melting rate of the ice (due to geothermal flow and sliding), omega
 *  is  the water source (water reaching the bed from the surface, through crevasses)
 *  and C_drain is a fixed rate, that makes the till drain in absence of water input
 */


template<typename EvalT, typename Traits, bool IsStokesCoupling>
class HydrologyResidualTillStorageEqn : public PHX::EvaluatorWithBaseImpl<Traits>,
                                        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename EvalT::ScalarT       ScalarT;

  HydrologyResidualTillStorageEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const RealType>      BF;
  PHX::MDField<const MeshScalarT>   w_measure;
  PHX::MDField<const ScalarT>       m;
  PHX::MDField<const ParamScalarT>  omega;
  PHX::MDField<const ScalarT>       h_till_dot;

  // Output:
  PHX::MDField<ScalarT>             residual;

  unsigned int numNodes;
  unsigned int numQPs;

  double rho_w;
  double scaling_omega;
  double scaling_h_dot;
  double C_drain;

  bool mass_lumping;
  bool use_melting;

  // Variables necessary for stokes coupling
  std::string                     sideSetName;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_RESIDUAL_TILL_STORAGE_EQN_HPP
