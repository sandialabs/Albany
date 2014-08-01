//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef NONLINEARPOISSONADJOINT_HPP
#define NONLINEARPOISSONADJOINT_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace SEE {
///
/// \brief Nonlinear Poisson Adjoint
///
/// This evaluator computes the residual of the
/// adjoint of a nonliner Poisson's problem
///
template<typename EvalT, typename Traits>
class NonlinearPoissonAdjoint : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  NonlinearPoissonAdjoint(const Teuchos::ParameterList& p,
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
  PHX::MDField<ScalarT,Cell,QuadPoint> u_;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> u_grad_;
  PHX::MDField<ScalarT,Cell,QuadPoint> lambda_;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> lambda_grad_;

  PHX::MDField<ScalarT,Cell,Node> residual_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

};
}

#endif
