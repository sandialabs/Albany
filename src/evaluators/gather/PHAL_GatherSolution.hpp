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
#include "Albany_DualView.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Albany {
  class DOFManager;
}

namespace PHAL {
/** \brief Gathers solution values from the Thyra vector into the PHX Field

    Currently makes an assumption that dofs are contiguous, possibly with
    an initial offset.
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
  GatherSolutionBase (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  using ref_t = typename PHAL::Ref<typename EvalT::ScalarT>::type;

  // These functions are used to select the correct field based on rank.
  // They are called from *inside* for loops, but the switch statement
  // is constant for all iterations, so the compiler branch predictor
  // can easily guess the correct branch, making the conditional jump
  // cheap.
  ref_t get_ref (const int cell, const int node, const int eq) const {
    switch (tensorRank) {
      case 0:
        return val[eq](cell,node);
      case 1:
        return valVec(cell,node,eq);
      case 2:
        return valTensor(cell,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  ref_t get_ref_dot (const int cell, const int node, const int eq) const {
    switch (tensorRank) {
      case 0:
        return val_dot[eq](cell,node);
      case 1:
        return valVec_dot(cell,node,eq);
      case 2:
        return valTensor_dot(cell,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  ref_t get_ref_dotdot (const int cell, const int node, const int eq) const {
    switch (tensorRank) {
      case 0:
        return val_dotdot[eq](cell,node);
      case 1:
        return valVec_dotdot(cell,node,eq);
      case 2:
        return valTensor_dotdot(cell,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  void gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr);

  // Offsets of solution field(s) inside a single element, as per the DOFManager
  // The only reason we have this view is that the offsets returned by the
  // dof manager would have the node striding faster. By doing the transposition
  // once, we can get better cached/coalesced memory access during the evaluation
  Albany::DualView<int**> m_fields_offsets;

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

  int numDim;
  int numNodes;
  int numFields; // Number of fields gathered in this call
  int offset;    // Offset of first DOF being gathered when numFields<neq
  int tensorRank;

  bool enableTransient;
  bool enableAcceleration;

  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;

public:
  struct SolAccessor {
    using DynRankView = Kokkos::DynRankView<ScalarT, PHX::Device>;
    using DRVDualView = Kokkos::DualView<DynRankView*, PHX::Device>;
    using t_dev = typename DRVDualView::t_dev;
    
    KOKKOS_INLINE_FUNCTION
    ref_t get_ref (const int cell, const int node, const int eq) const {
      switch (tensorRank) {
        case 0:
          return d_val[eq](cell,node);
        case 1:
          return d_valVec(cell,node,eq);
        case 2:
          return d_valTensor(cell,node,eq/numDim,eq%numDim);
      }
      Kokkos::abort("Unsupported tensor rank");
    }

    KOKKOS_INLINE_FUNCTION
    ref_t get_ref_dot (const int cell, const int node, const int eq) const {
      switch (tensorRank) {
        case 0:
          return d_val_dot[eq](cell,node);
        case 1:
          return d_valVec_dot(cell,node,eq);
        case 2:
          return d_valTensor_dot(cell,node,eq/numDim,eq%numDim);
      }
      Kokkos::abort("Unsupported tensor rank");
    }

    KOKKOS_INLINE_FUNCTION
    ref_t get_ref_dotdot (const int cell, const int node, const int eq) const {
      switch (tensorRank) {
        case 0:
          return d_val_dotdot[eq](cell,node);
        case 1:
          return d_valVec_dotdot(cell,node,eq);
        case 2:
          return d_valTensor_dotdot(cell,node,eq/numDim,eq%numDim);
      }
      Kokkos::abort("Unsupported tensor rank");
    }

    int tensorRank;
    int numDim;
    DRVDualView val_kokkos, val_dot_kokkos, val_dotdot_kokkos;
    t_dev d_val, d_val_dot, d_val_dotdot;
    DynRankView d_valVec, d_valVec_dot, d_valVec_dotdot;
    DynRankView d_valTensor, d_valTensor_dot, d_valTensor_dotdot;
  };
  SolAccessor device_sol;
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
   : public GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>
{
public:
  GatherSolution( const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits> Base;

  using ref_t = typename Base::ref_t;
  using Base::get_ref;
  using Base::get_ref_dot;
  using Base::get_ref_dotdot;
  using Base::numFields;
  using Base::m_fields_offsets;

private:
  using RangePolicy = typename Base::RangePolicy;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;

  using ref_t = typename Base::ref_t;
  using Base::get_ref;
  using Base::get_ref_dot;
  using Base::get_ref_dotdot;
  using Base::numFields;
  using Base::m_fields_offsets;

private:
  using RangePolicy = typename Base::RangePolicy;
};


// **************************************************************
// Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Tangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>
{
public:
  GatherSolution (const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:

  typedef GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits> Base;

  using ref_t = typename Base::ref_t;
  using Base::get_ref;
  using Base::get_ref_dot;
  using Base::get_ref_dotdot;
  using Base::numFields;
  using Base::m_fields_offsets;

protected:
  using RangePolicy = typename Base::RangePolicy;

};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> Base;

  using ref_t = typename Base::ref_t;
  using Base::get_ref;
  using Base::get_ref_dot;
  using Base::get_ref_dotdot;
  using Base::numFields;
  using Base::m_fields_offsets;

protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the GatherSolution Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to gather the solution for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$ \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$ \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first parameter, \f$\boldsymbol{p}_2\f$ is a potentially different second parameter,
 * \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1} \f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This gather is used when calling: 
 * <ul>
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResponse_HessVecProd_pp,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResidual_HessVecProd_pp.
 * </ul>
 */

template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>
{
public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);

  /**
   * @brief Gather the solution for the Hessian-vector product computations.
   * 
   * The PHAL::AlbanyTraits::HessianVec::ScalarT is a nested Sacado::FAD type with two levels of
   * differentiation. 
   * 
   * This member function behaves in four different ways depending on which Hessian-vector product
   * contribution is currently computed:
   * 
   * <ol>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   * </ul>
   * the first derivative and the second derivative are w.r.t the solution and the values of the FAD types have to be
   * initialized for both first and second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution, the first derivative, and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0.4
   * <tr><th>dx(0)                <td>1          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = true;
   *  is_x_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   * </ul>
   * only the first derivative is w.r.t the solution and the values of the FAD types have to be
   * initialized for the first derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution and the first derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0
   * <tr><th>dx(0)                <td>1          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = true;
   *  is_x_direction_active = false;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   * </ul>
   * only the second derivative is w.r.t the solution and the values of the FAD types have to be
   * initialized for the second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0.4
   * <tr><th>dx(0)                <td>0          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = false;
   *  is_x_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   * </ul>
   * none of the derivative is w.r.t the solution.
   * The values of the derivatives of the FAD types should not be initialized during this function call.
   * 
   * Such a case is illustrated in the following table; only the value of the solution has to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0
   * <tr><th>dx(0)                <td>0          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt> is_x_active = false;
   *  is_x_direction_active = false;
   * </tt>
   * 
   * </li>
   * </ol>
   */
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef GatherSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits> Base;

  using ref_t = typename Base::ref_t;
  using Base::get_ref;
  using Base::get_ref_dot;
  using Base::get_ref_dotdot;
  using Base::numFields;
  using Base::m_fields_offsets;

protected:
  using RangePolicy = typename Base::RangePolicy;
};

} // namespace PHAL

#endif // PHAL_GATHER_SOLUTION_HPP
