//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ENERGY_HPP
#define ENERGY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "PhaseProblem.hpp"

namespace AMP {
///
/// \brief  energy evaluator
///
/// This evaluator computes the total energy
///
template<typename EvalT, typename Traits>
class Energy : 
  public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Energy(Teuchos::ParameterList& p,const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void postEvaluate(typename Traits::PostEvalData d);  

private:
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<const ScalarT,Cell,QuadPoint> T_;
  PHX::MDField<const ScalarT,Cell,QuadPoint> phi_;
  PHX::MDField<const ScalarT,Cell,QuadPoint> rho_Cp_;
  PHX::MDField<const ScalarT,Cell,QuadPoint> laser_source_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;
  
  ScalarT Cl_;	  // Volumetric heat capacity in liquid
  ScalarT L_;  	  // Latent heat of fusion/melting
  ScalarT Tm_;    // Melting temperature
  ScalarT Tc_;    // Delta temperature to compute phi
  
  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;
};
}

#endif
