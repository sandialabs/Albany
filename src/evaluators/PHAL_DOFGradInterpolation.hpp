//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFGRAD_INTERPOLATION_HPP
#define PHAL_DOFGRAD_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFGradInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFGradInterpolationBase_Residual_Tag{};


#ifdef KOKKOS_OPTIMIZED
  typedef Kokkos::TeamPolicy<ExecutionSpace>              team_policy ;
  typedef team_policy::member_type team_member ;
  const int work_size=256;
  int numCells;
  int threads_per_team; //=worksize/numQP
  int numTeams; //#of elements/threads_per_team

  void operator()( const team_member & thread) const;

#else
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFGradInterpolationBase_Residual_Tag> DOFGradInterpolationBase_Residual_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFGradInterpolationBase_Residual_Tag& tag, const int& cell) const;
#endif
#endif

};

template<typename Traits>
class DOFGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:

  DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t offset;

 #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFGradInterpolationBase_Jacobian_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFGradInterpolationBase_Jacobian_Tag> DOFGradInterpolationBase_Jacobian_Policy;

  int num_dof;
  int neq;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const;

#endif

};

#ifdef ALBANY_SG
template<typename Traits>
class DOFGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::SGJacobian, Traits>  {

public:

  DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::SGJacobian::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t offset;
};
#endif

#ifdef ALBANY_ENSEMBLE
template<typename Traits>
class DOFGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::MPJacobian, Traits>  {

public:

  DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::MPJacobian::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t offset;
};
#endif

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFGradInterpolation = DOFGradInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFGradInterpolationMesh = DOFGradInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFGradInterpolationParam = DOFGradInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOFGRAD_INTERPOLATION_HPP
