//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef NONLINEARPOISSONRESIDUAL_HPP
#define NONLINEARPOISSONRESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace AMP {
///
/// \brief Nonlinear Poisson Residual
///
/// This evaluator computes the residual to a nonliner
/// Poisson's problem
///
template<typename EvalT, typename Traits>
class NonlinearPoissonResidual : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  ///
  /// Constructor
  ///
  NonlinearPoissonResidual(const Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Input: Weighted Basis Functions
  ///
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> w_bf_;

  ///
  /// Input: Weighted Basis Function Gradients
  ///
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> w_grad_bf_;

  ///
  /// Input: Unknown
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint> u_;

  ///
  /// Input: Unknown Gradient
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> u_grad_;



  PHX::MDField<ScalarT,Cell,QuadPoint> Tdot;
  PHX::MDField<ScalarT,Cell,QuadPoint> ThermalCond;
  PHX::MDField<ScalarT,Cell,QuadPoint> Source;
  Teuchos::Array<double> convectionVels;
  PHX::MDField<ScalarT,Cell,QuadPoint> rhoCp;
  PHX::MDField<ScalarT,Cell,QuadPoint> Absorption;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> TResidual;

  bool haveSource;
  bool haveConvection;
  bool haveAbsorption;
  bool enableTransient;
  bool haverhoCp;
  unsigned int numQPs, numDims, numNodes, worksetSize;
  Intrepid::FieldContainer<ScalarT> flux;
  Intrepid::FieldContainer<ScalarT> aterm;
};
}

#endif
