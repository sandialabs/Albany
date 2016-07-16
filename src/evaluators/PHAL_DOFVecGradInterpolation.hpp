//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFVECGRAD_INTERPOLATION_HPP
#define PHAL_DOFVECGRAD_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFVec values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename Type>
class DOFVecGradInterpolation : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef Type ScalarT;

  typedef typename EvalT::MeshScalarT MeshScalarT;


  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFVecGradInterpolation_Residual_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFVecGradInterpolation_Residual_Tag> DOFVecGradInterpolation_Residual_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFVecGradInterpolation_Residual_Tag& tag, const int& cell) const;

#endif

};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
 	public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:

  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;


  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  std::size_t offset;

//KOKKOS:
 #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
 
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFVecGradInterpolation_Jacobian_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFVecGradInterpolation_Jacobian_Tag> DOFVecGradInterpolation_Jacobian_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFVecGradInterpolation_Jacobian_Tag& tag, const int& cell) const;

  int num_dof, neq;

#endif
};

#ifdef ALBANY_SG
//! Specialization for SGJacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, SGFadType>\
      : public PHX::EvaluatorWithBaseImpl<Traits>,
 	public PHX::EvaluatorDerived<PHAL::AlbanyTraits::SGJacobian, Traits>  {

public:

  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::SGJacobian::MeshScalarT MeshScalarT;


  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  std::size_t offset;
};
#endif

#ifdef ALBANY_ENSEMBLE
//! Specialization for MPJacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, MPFadType>\
      : public PHX::EvaluatorWithBaseImpl<Traits>,
 	public PHX::EvaluatorDerived<PHAL::AlbanyTraits::MPJacobian, Traits>  {

public:

  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::MPJacobian::MeshScalarT MeshScalarT;


  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  std::size_t offset;
};
#endif

}

#endif
