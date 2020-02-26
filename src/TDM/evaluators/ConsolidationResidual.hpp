//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TDM_CONSOLIDATIONRESIDUAL_HPP
#define TDM_CONSOLIDATIONRESIDUAL_HPP

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "AAdapt_RC_Field.hpp"
#include "Albany_Layouts.hpp"

namespace TDM {
///
/// \brief Consolidation Residual
///
/// This evaluator computes the residual due to consolidation of 
/// the powder layer during melting in the Selective Laser Melting
/// additive manufacturing process
///
template<typename EvalT, typename Traits>
class ConsolidationResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits> {
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Constructor
  ///
  ConsolidationResidual(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  //Inputs
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> w_bf_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;
  PHX::MDField<ScalarT,Cell,QuadPoint> psi_;
  PHX::MDField<ScalarT,Cell,QuadPoint> porosity_;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim,Dim> GradU;

  //Outputs
  PHX::MDField<ScalarT, Cell, Node, Dim> residual_;

  int num_nodes_;
  int num_pts_;
  int num_dims_;
  int workset_size_;

  ScalarT DetF;
  ScalarT Initial_porosity;  

  
 //Not sure if the Kokkos stuff is needed, it is not in the AMP PhaseResidual evaluator
 public:  // Kokkos
  struct residual_Tag
  {
  };
  struct residual_haveBodyForce_Tag
  {
  };
  struct residual_haveBodyForce_and_dynamic_Tag
  {
  };
  struct residual_have_dynamic_Tag
  {
  };

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, residual_Tag> residual_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, residual_haveBodyForce_Tag>
      residual_haveBodyForce_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, residual_haveBodyForce_and_dynamic_Tag>
          residual_haveBodyForce_and_dynamic_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, residual_have_dynamic_Tag>
      residual_have_dynamic_Policy;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_haveBodyForce_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_haveBodyForce_and_dynamic_Tag& tag, const int& i)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_have_dynamic_Tag& tag, const int& i) const;


};
}

#endif
