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

namespace PHAL {
/** \brief Scatters result from the residual fields into the
    global (epetra) data structurs.  This includes the
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
  typedef typename EvalT::ScalarT ScalarT;
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<ScalarT const,Cell,Node> > val;
  PHX::MDField<ScalarT const,Cell,Node,Dim>  valVec;
  PHX::MDField<ScalarT const,Cell,Node,Dim,Dim> valTensor;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq

  unsigned short int tensorRank;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
protected:
  Albany::WorksetConn nodeID;
  Albany::DeviceView1d<ST> f_kokkos;
  Kokkos::vector<Kokkos::DynRankView<const ScalarT, PHX::Device>, PHX::Device> val_kokkos;

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

  typedef typename EvalT::ScalarT ScalarT;
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
class ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_ScatterResRank0_Tag{};
  struct PHAL_ScatterResRank1_Tag{};
  struct PHAL_ScatterResRank2_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const;

private:
  int numDims;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::nodeID;
  using Base::f_kokkos;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag> PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag> PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag> PHAL_ScatterResRank2_Policy;

#endif
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_ScatterResRank0_Tag{};
  struct PHAL_ScatterJacRank0_Adjoint_Tag{};
  struct PHAL_ScatterJacRank0_Tag{};
  struct PHAL_ScatterResRank1_Tag{};
  struct PHAL_ScatterJacRank1_Adjoint_Tag{};
  struct PHAL_ScatterJacRank1_Tag{};
  struct PHAL_ScatterResRank2_Tag{};
  struct PHAL_ScatterJacRank2_Adjoint_Tag{};
  struct PHAL_ScatterJacRank2_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank0_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank0_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank1_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank1_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank2_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank2_Tag&, const int& cell) const;

private:
  void evaluateFieldsDevice(typename Traits::EvalData d);
  void evaluateFieldsHost(typename Traits::EvalData d);
  int neq, nunk, numDims;
  Albany::DeviceLocalMatrix<ST> Jac_kokkos;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::nodeID;
  using Base::f_kokkos;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  static constexpr bool is_atomic = KU::NeedsAtomic<ExecutionSpace>::value;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag> PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Adjoint_Tag> PHAL_ScatterJacRank0_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Tag> PHAL_ScatterJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag> PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Adjoint_Tag> PHAL_ScatterJacRank1_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Tag> PHAL_ScatterJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag> PHAL_ScatterResRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Adjoint_Tag> PHAL_ScatterJacRank2_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Tag> PHAL_ScatterJacRank2_Policy;

#endif
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

template<typename Traits>
class ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl) {
    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the ScatterResidual Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
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
class ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::HessianVec, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
};

/**
 * @brief Template specialization of the ScatterResidualWithExtrudedParams Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
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
class ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::HessianVec, Traits>  {
public:
  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    ScatterResidual<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl) {
    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    ScatterResidual<PHAL::AlbanyTraits::HessianVec, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluate2DFieldsDerivativesDueToExtrudedParams(typename Traits::EvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************

} // namespace PHAL

#endif // PHAL_SCATTER_RESIDUAL_HPP
