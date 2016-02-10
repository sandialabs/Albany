//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ENERGY_DOT_HPP
#define ENERGY_DOT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace AMP {
///
/// \brief  Rate of energy evaluator
///
/// This evaluator computes the rate of energy
///
template<typename EvalT, typename Traits>
class EnergyDot : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  EnergyDot(const Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<ScalarT,Cell,QuadPoint> T_;
  PHX::MDField<ScalarT,Cell,QuadPoint> T_dot_;
  PHX::MDField<ScalarT,Cell,QuadPoint> phi_;
  PHX::MDField<ScalarT,Cell,QuadPoint> psi_;
  PHX::MDField<ScalarT,Cell,QuadPoint> rho_Cp_;
  PHX::MDField<ScalarT,Dummy> time_;
  PHX::MDField<ScalarT,Dummy> deltaTime_;

  PHX::MDField<ScalarT,Cell,QuadPoint> energyDot_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;
  
  // Volumetric heat capacity in liquid
  ScalarT Cl_;
  // Latent heat of fusion/melting
  ScalarT L_;
  // Melting temperature
  ScalarT Tm_;
  // Delta temperature to compute phi
  ScalarT Tc_;
  
// old temperature name
  std::string Temperature_Name_;

  Teuchos::RCP<const Teuchos::ParameterList>
    getValidEnergyDotParameters() const;
};
}

#endif
