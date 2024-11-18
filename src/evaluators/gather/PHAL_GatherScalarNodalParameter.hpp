//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP
#define PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Gathers parameter values from distributed vectors into
    scalar nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherScalarNodalParameterBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  GatherScalarNodalParameterBase(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;
  virtual ~GatherScalarNodalParameterBase() = default;

protected:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  const int numNodes;
  const std::string param_name;

  // Output:
  PHX::MDField<ParamScalarT,Cell,Node> val;

  MDFieldMemoizer<Traits> memoizer;
};

// General version for most evaluation types
template<typename EvalT, typename Traits>
class GatherScalarNodalParameter :
    public GatherScalarNodalParameterBase<EvalT, Traits>  {

public:
  GatherScalarNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherScalarNodalParameter(const Teuchos::ParameterList& p);
//  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ParamScalarT ParamScalarT;
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

// General version for most evaluation types
template<typename EvalT, typename Traits>
class GatherScalarExtruded2DNodalParameter :
    public GatherScalarNodalParameterBase<EvalT, Traits>  {

public:
  GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ParamScalarT ParamScalarT;
  const int fieldLevel;
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};


// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// DistParamDeriv
// **************************************************************
template<typename Traits>
class GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>  {

public:
  GatherScalarNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherScalarNodalParameter(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ParamScalarT ParamScalarT;
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};


template<typename Traits>
class GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>  {

public:
  GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ParamScalarT ParamScalarT;
  const int fieldLevel;
};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the GatherScalarNodalParameter Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to gather the distributed parameter for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle f(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle f(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle f(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first parameter,  \f$\boldsymbol{p}_2 \f$ is a potentially different second parameter,
 *  \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector,  \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector  \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
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
class GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::HessianVec,Traits>  {

public:
  GatherScalarNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherScalarNodalParameter(const Teuchos::ParameterList& p);

  /**
   * @brief Gather the parameter for the Hessian-vector product computations.
   * 
   * The PHAL::AlbanyTraits::HessianVec::ScalarT is a nested Sacado::FAD type with two levels of
   * differentiation. 
   * 
   * This member function behaves in four different ways depending on which Hessian-vector product
   * contribution is currently computed, for a current parameter \f$\boldsymbol{p}_1\f$:
   * 
   * <ol>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   * </ul>
   * the first derivative and the second derivative are w.r.t the parameter \f$\boldsymbol{p}_1\f$ and the values of the FAD types have to be
   * initialized for both first and second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the parameter, the first derivative, and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a parameter</caption>
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
   *  is_p_active = true;
   *  is_p_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$ where \f$j\neq1\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$ where \f$j\neq1\f$,
   * </ul>
   * only the first derivative is w.r.t the current parameter \f$\boldsymbol{p}_1\f$ and the values of the FAD types have to be
   * initialized for the first derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the parameter and the first derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a parameter</caption>
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
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ where \f$i\neq1\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ where \f$i\neq1\f$,
   * </ul>
   * only the second derivative is w.r.t the current parameter \f$\boldsymbol{p}_1\f$ and the values of the FAD types have to be
   * initialized for the second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the parameter and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a parameter</caption>
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
   *  is_p_active = false;
   *  is_p_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$  where \f$j\neq1\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$  where \f$j\neq1\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$  where \f$i\neq1\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$  where \f$i\neq1\neq j\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$  where \f$i\neq1\neq j\f$,
   * </ul>
   * none of the derivative is w.r.t the current parameter \f$\boldsymbol{p}_1\f$. 
   * The values of the derivatives of the FAD types should not be initialized during this function call.
   * 
   * Such a case is illustrated in the following table; only the value of the parameter has to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a parameter</caption>
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
   * <tt> is_p_active = false;
   *  is_p_direction_active = false;
   * </tt>
   * 
   * </li>
   * </ol>
   */
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ParamScalarT ParamScalarT;
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};


/**
 * @brief Template specialization of the GatherScalarExtruded2DNodalParameter Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to gather the extruded distributed parameter for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle f(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle f(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle f(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first parameter which can be extruded, \f$\boldsymbol{p}_2\f$ is a potentially different second parameter which can be extruded too,
 *  \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
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
class GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::HessianVec,Traits>  {

public:
  GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  /**
   * @brief Gather the extruded parameter for the Hessian-vector product computations.
   * 
   * See PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits> for more details on the implementation.
   */
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ParamScalarT ParamScalarT;
  const int fieldLevel;
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

} // namespace PHAL

#endif // PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP
