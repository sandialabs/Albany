//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


//IK, 9/13/14: only Epetra is SG and MP 

#ifndef PHAL_GATHER_SOLUTION_HPP
#define PHAL_GATHER_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "Kokkos_Vector.hpp"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherSolutionBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  GatherSolutionBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:
  typedef typename EvalT::ScalarT ScalarT;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val_dot;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val_dotdot;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec_dot;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec_dotdot;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor_dot;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor_dotdot;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq
  unsigned short int tensorRank;
  bool enableTransient;
  bool enableAcceleration;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
protected:
  Albany::AbstractDiscretization::WorksetConn nodeID;
  Kokkos::View<const ST*, PHX::Device> xT_constView, xdotT_constView, xdotdotT_constView;

  typedef Kokkos::vector<Kokkos::DynRankView<ScalarT, PHX::Device>, PHX::Device> KV;
  KV val_kokkos, val_dot_kokkos, val_dotdot_kokkos;
  typename KV::t_dev d_val, d_val_dot, d_val_dotdot;

#endif
};

template<typename EvalT, typename Traits> class GatherSolution;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Residual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  const int numFields;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
public:
  struct PHAL_GatherSolRank2_Tag{};
  struct PHAL_GatherSolRank2_Transient_Tag{};
  struct PHAL_GatherSolRank2_Acceleration_Tag{};

  struct PHAL_GatherSolRank1_Tag{};
  struct PHAL_GatherSolRank1_Transient_Tag{};
  struct PHAL_GatherSolRank1_Acceleration_Tag{};

  struct PHAL_GatherSolRank0_Tag{};
  struct PHAL_GatherSolRank0_Transient_Tag{};
  struct PHAL_GatherSolRank0_Acceleration_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Acceleration_Tag&, const int& cell) const;

private:
  int numDim;

  typedef GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::nodeID;
  using Base::xT_constView;
  using Base::xdotT_constView;
  using Base::xdotdotT_constView;
  using Base::val_kokkos;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Tag> PHAL_GatherSolRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Transient_Tag> PHAL_GatherSolRank2_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Acceleration_Tag> PHAL_GatherSolRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Tag> PHAL_GatherSolRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Transient_Tag> PHAL_GatherSolRank1_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Acceleration_Tag> PHAL_GatherSolRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Tag> PHAL_GatherSolRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Transient_Tag> PHAL_GatherSolRank0_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Acceleration_Tag> PHAL_GatherSolRank0_Acceleration_Policy;

#endif
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  const int numFields;
 
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
public:
  struct PHAL_GatherJacRank2_Tag{};
  struct PHAL_GatherJacRank2_Transient_Tag{};
  struct PHAL_GatherJacRank2_Acceleration_Tag{};

  struct PHAL_GatherJacRank1_Tag{};
  struct PHAL_GatherJacRank1_Transient_Tag{};
  struct PHAL_GatherJacRank1_Acceleration_Tag{};

  struct PHAL_GatherJacRank0_Tag{};
  struct PHAL_GatherJacRank0_Transient_Tag{};
  struct PHAL_GatherJacRank0_Acceleration_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Acceleration_Tag&, const int& cell) const;
 
private:
  int neq, numDim;
  double j_coeff, n_coeff, m_coeff;

  typedef GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::nodeID;
  using Base::xT_constView;
  using Base::xdotT_constView;
  using Base::xdotdotT_constView;
  using Base::val_kokkos;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Tag> PHAL_GatherJacRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Transient_Tag> PHAL_GatherJacRank2_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Acceleration_Tag> PHAL_GatherJacRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Tag> PHAL_GatherJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Transient_Tag> PHAL_GatherJacRank1_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Acceleration_Tag> PHAL_GatherJacRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Tag> PHAL_GatherJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Transient_Tag> PHAL_GatherJacRank0_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Acceleration_Tag> PHAL_GatherJacRank0_Acceleration_Policy;

#endif
};


// **************************************************************
// Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Tangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  typedef typename Kokkos::View<ScalarT*, PHX::Device>::reference_type reference_type;
  const std::size_t numFields;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************

#ifdef ALBANY_SG
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGResidual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGResidual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  const std::size_t numFields;
};


// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGTangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGTangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  typedef typename Kokkos::View<ScalarT*, PHX::Device>::reference_type reference_type;
  const std::size_t numFields;
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual
// **************************************************************

template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPResidual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  const std::size_t numFields;
};


// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPTangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPTangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  const std::size_t numFields;
};
#endif

// **************************************************************
}

#endif
