//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_SOLUTION_HPP
#define PHAL_GATHER_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

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
class GatherSolutionBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
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
  Albany::WorksetConn nodeID;
  Albany::DeviceView1d<const ST> x_constView, xdot_constView, xdotdot_constView;

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
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;
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
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;
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
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the GatherSolution Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to gather the solution for the computation of:
 * <ul>
 *  <li> The @f$ H_{xx}(g)v_{x} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{xx}(g)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{x} g(x+ r\,v_{x},p_1, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{xp_1}(g)v_{p_1} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{xp_1}(g)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} g(x, p_1+ r\,v_{p_1}, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(g)v_{x} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{p_2x}(g)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} g(x+ r\,v_{x},p_1, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{xx}(f,z)v_{x} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{xx}(f,z)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{x} \left\langle f(x+ r\,v_{x},p_1, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{xp_1}(f,z)v_{p_1} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{xp_1}(f,z)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} \left\langle f(x, p_1+ r\,v_{p_1}, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(f,z)v_{x} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{p_2x}(f,z)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} \left\langle f(x+ r\,v_{x},p_1, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  @f$ x @f$  is the solution,  @f$ p_1 @f$  is a first distributed parameter,  @f$ p_2 @f$  is a potentially different second distributed parameter,
 *  @f$  g @f$  is the response function, @f$  f @f$  is the residual, @f$  z  @f$ is the Lagrange multiplier vector,  @f$ v_{x} @f$  is a direction vector
 *  with the same dimension as the vector  @f$ x @f$, and @f$ v_{p_1} @f$  is a direction vector with the same dimension as the vector  @f$ p_1 @f$.
 */

template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
  const std::size_t numFields;
};

} // namespace PHAL

#endif // PHAL_GATHER_SOLUTION_HPP
