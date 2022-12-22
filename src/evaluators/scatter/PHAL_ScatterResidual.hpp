//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_RESIDUAL_HPP
#define PHAL_SCATTER_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Albany_KokkosTypes.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_KokkosUtils.hpp"

#include "Teuchos_ParameterList.hpp"

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
#include "Kokkos_Vector.hpp"
#endif

namespace Albany {
class DOFManager;
}

namespace PHAL {
/** \brief Scatters result from the residual fields into the
    global (epetra) data structures.  This includes the
    post-processing of the AD data type for all evaluation
    types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class ScatterResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

protected:
  using ScalarT = typename EvalT::ScalarT;
  void gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr);

  // These functions are used to select the correct field based on rank.
  // They are called from *inside* for loops, but the switch statement
  // is constant for all iterations, so the compiler branch predictor
  // can easily guess the correct branch, making the conditional jump
  // cheap.
  KOKKOS_INLINE_FUNCTION
  ScalarT get_resid (const int cell, const int node, const int eq) const {
    switch (tensorRank) {
      case 0:
        KOKKOS_IF_ON_HOST  (return val[eq](cell,node););
        KOKKOS_IF_ON_DEVICE(return d_val[eq](cell,node););
      case 1:
      {  return valVec(cell,node,eq); }
      case 2:
      {  return valTensor(cell,node,eq/numDim,eq%numDim); }
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  Teuchos::RCP<PHX::FieldTag> scatter_operation;

  std::vector<PHX::MDField<const ScalarT,Cell,Node>>  val;
  PHX::MDField<const ScalarT,Cell,Node,Dim>           valVec;
  PHX::MDField<const ScalarT,Cell,Node,Dim,Dim>       valTensor;

  // Offsets of solution field(s) inside a single element, as per the DOFManager
  // The only reason we have this view is that the offsets returned by the
  // dof manager would have the node striding faster. By doing the transposition
  // once, we can get better cached/coalesced memory access during the evaluation
  Albany::DualView<int**> m_fields_offsets;

  int numNodes;
  int numFields;  // Number of fields gathered in this call
  int offset;     // Offset of first DOF being gathered when numFields<neq
  int numDim;
  int tensorRank;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;

  Albany::DeviceView1d<ST> f_data;
  typedef Kokkos::vector<Kokkos::DynRankView<const ScalarT, PHX::Device>, PHX::Device> KV;
  KV val_kokkos;
  typename KV::t_dev d_val;
#endif
};

template<typename EvalT, typename Traits> class ScatterResidual;

template<typename EvalT, typename Traits>
class ScatterResidualWithExtrudedParams
  : public ScatterResidual<EvalT, Traits> {

public:

  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
                                ScatterResidual<EvalT, Traits>(p,dl) {
    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    ScatterResidual<EvalT, Traits>::postRegistrationSetup(d,vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    ScatterResidual<EvalT, Traits>::evaluateFields(d);
  }

protected:

  using Base = ScatterResidualBase<EvalT, Traits>;
  using ScalarT = typename Base::ScalarT;
  using Base::get_resid;
  using Base::m_fields_offsets;
  using Base::numNodes;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterResidual<AlbanyTraits::Residual,Traits>
  : public ScatterResidualBase<AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidualBase<AlbanyTraits::Residual, Traits>;
  using ExecutionSpace = typename Base::ExecutionSpace;
  using RangePolicy = typename Base::RangePolicy;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::numFields;
  using Base::numNodes;
  using Base::m_fields_offsets;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<AlbanyTraits::Jacobian,Traits>
  : public ScatterResidualBase<AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:

  void evaluateFieldsDevice(typename Traits::EvalData d);
  void evaluateFieldsHost(typename Traits::EvalData d);

  using Base = ScatterResidualBase<AlbanyTraits::Jacobian, Traits>;
  using ScalarT = typename Base::ScalarT;
  using ExecutionSpace = typename Base::ExecutionSpace;
  using RangePolicy = typename Base::RangePolicy;

  using Base::get_resid;
  using Base::numFields;
  using Base::numNodes;
  using Base::m_fields_offsets;

  static constexpr bool is_atomic = KU::NeedsAtomic<PHX::Device::execution_space>::value;

  Albany::DualView<int*> m_volume_eqns;
  Albany::DualView<int*> m_volume_eqns_offsets;
  Albany::DualView<int*> m_lids;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<AlbanyTraits::Tangent,Traits>
  : public ScatterResidualBase<AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidualBase<AlbanyTraits::Tangent, Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::numFields;
  using Base::numNodes;
  using Base::m_fields_offsets;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidual<AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidualBase<AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidualBase<AlbanyTraits::DistParamDeriv, Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::numNodes;
  using Base::numFields;
  using Base::m_fields_offsets;
};

template<typename Traits>
class ScatterResidualWithExtrudedParams<AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>
{
public:
  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::m_fields_offsets;
  using Base::numNodes;
  using Base::numFields;

  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the ScatterResidual Class for AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to scatter the residual for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 * where \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first parameter, \f$\boldsymbol{p}_2\f$ is a potentially different second parameter,
 * \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 * with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This scatter is used when calling:
 * <ul>
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResidual_HessVecProd_pp.
 * </ul>
 */
template<typename Traits>
class ScatterResidual<AlbanyTraits::HessianVec,Traits>
  : public ScatterResidualBase<AlbanyTraits::HessianVec, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidualBase<AlbanyTraits::HessianVec, Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::m_fields_offsets;
  using Base::numNodes;
  using Base::numFields;
};

/**
 * @brief Template specialization of the ScatterResidualWithExtrudedParams Class for AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to scatter the residual for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  \f$\boldsymbol{x}\f$  is the solution, \f$\boldsymbol{p}_1\f$  is a first parameter, \f$\boldsymbol{p}_2\f$  is a potentially different second parameter
 *  which is extruded, \f$\boldsymbol{f}\f$  is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This scatter is used when calling:
 * <ul>
 *   <li> Albany::Application::evaluateResidual_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResidual_HessVecProd_pp.
 * </ul>
 */
template<typename Traits>
class ScatterResidualWithExtrudedParams<AlbanyTraits::HessianVec,Traits>
  : public ScatterResidual<AlbanyTraits::HessianVec, Traits>
{
public:
  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluate2DFieldsDerivativesDueToExtrudedParams(typename Traits::EvalData d);
  void evaluateFields(typename Traits::EvalData d);
protected:
  using Base = ScatterResidual<AlbanyTraits::HessianVec, Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::get_resid;
  using Base::m_fields_offsets;
  using Base::numNodes;
  using Base::numFields;

  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************

} // namespace PHAL

#endif // PHAL_SCATTER_RESIDUAL_HPP
