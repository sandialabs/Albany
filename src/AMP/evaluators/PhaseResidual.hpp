//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHASERESIDUAL_HPP
#define PHASERESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace AMP {
///
/// \brief  Phase Residual
///
/// This evaluator computes the residual to a 
/// phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class PhaseResidual : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  PhaseResidual(const Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> w_bf_;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> w_grad_bf_;
  PHX::MDField<ScalarT,Cell,QuadPoint> T_;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> T_grad_;
  PHX::MDField<ScalarT,Cell,QuadPoint> T_dot_;
  PHX::MDField<ScalarT,Cell,QuadPoint> k_;
  PHX::MDField<ScalarT,Cell,QuadPoint> rho_cp_;
  PHX::MDField<ScalarT,Cell,QuadPoint> source_;
  PHX::MDField<ScalarT,Cell,QuadPoint> laser_source_;
  PHX::MDField<ScalarT,Dummy> time;
  PHX::MDField<ScalarT,Dummy> deltaTime;

  PHX::MDField<ScalarT,Cell,Node> residual_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  bool enable_transient_;
  std::string Temperature_Name_;
  Intrepid2::FieldContainer<ScalarT> term1_;
  Intrepid2::FieldContainer<ScalarT> term2_;

};
}

#endif
