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

  const std::size_t numNodes;
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
 *  where \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first distributed parameter,  \f$\boldsymbol{p}_2 \f$ is a potentially different second distributed parameter,
 *  \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector,  \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector  \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This gather is used when calling: 
 * <ul>
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_xx,
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_xp,
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_px, 
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_pp,
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
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ParamScalarT ParamScalarT;
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
 *  where  \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first distributed parameter which can be extruded, \f$\boldsymbol{p}_2\f$ is a potentially different second distributed parameter which can be extruded too,
 *  \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This gather is used when calling: 
 * <ul>
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_xx,
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_xp,
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_px, 
 *   <li> Albany::Application::evaluateResponseDistParamHessVecProd_pp,
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
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ParamScalarT ParamScalarT;
  const int fieldLevel;
};

} // namespace PHAL

#endif // PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP
