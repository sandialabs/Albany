//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_INTERPOLATION_HPP
#define PHAL_DOF_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFInterpolationBase (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
};
/*
//! Specialization for Jacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:

  DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t offset;
};
*/
#ifdef ALBANY_SG
//! Specialization for SGJacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::SGJacobian, Traits>  {

public:

  DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t offset;
};
#endif

#ifdef ALBANY_ENSEMBLE
//! Specialization for MPJacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::MPJacobian, Traits>  {

public:

  DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t offset;
};
#endif

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFInterpolation = DOFInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationMesh = DOFInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationParam = DOFInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_INTERPOLATION_HPP
