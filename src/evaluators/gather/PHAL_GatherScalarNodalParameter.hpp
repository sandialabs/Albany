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
 *  <li> The @f$ H_{xp_1}(g)v_{p_1} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{xp_1}(g)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} g(x, p_1+ r\,v_{p_1}, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(g)v_{x} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{p_2x}(g)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} g(x+ r\,v_{x},p_1, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2p_1}(g)v_{p_1} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{p_2p_1}(g)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{p_2} g(x, p_1+ r\,v_{p_1}, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{xp_1}(f,z)v_{p_1} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{xp_1}(f,z)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} \left\langle f(x, p_1+ r\,v_{p_1}, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(f,z)v_{x} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{p_2x}(f,z)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} \left\langle f(x+ r\,v_{x},p_1, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2p_1}(f,z)v_{p_1} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{p_2p_1}(f,z)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{p_2} \left\langle f(x, p_1+ r\,v_{p_1}, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  @f$ x @f$  is the solution,  @f$ p_1 @f$  is a first distributed parameter,  @f$ p_2 @f$  is a potentially different second distributed parameter,
 *  @f$  g @f$  is the response function, @f$  f @f$  is the residual, @f$  z  @f$ is the Lagrange multiplier vector,  @f$ v_{x} @f$  is a direction vector
 *  with the same dimension as the vector  @f$ x @f$, and @f$ v_{p_1} @f$  is a direction vector with the same dimension as the vector  @f$ p_1 @f$.
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
 *  <li> The @f$ H_{xp_1}(g)v_{p_1} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{xp_1}(g)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} g(x, p_1+ r\,v_{p_1}, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(g)v_{x} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{p_2x}(g)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} g(x+ r\,v_{x},p_1, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2p_1}(g)v_{p_1} @f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         H_{p_2p_1}(g)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{p_2} g(x, p_1+ r\,v_{p_1}, p_2)\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{xp_1}(f,z)v_{p_1} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{xp_1}(f,z)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{x} \left\langle f(x, p_1+ r\,v_{p_1}, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2x}(f,z)v_{x} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{p_2x}(f,z)v_{x}=\left.\frac{\partial}{\partial r} \nabla_{p_2} \left\langle f(x+ r\,v_{x},p_1, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The @f$ H_{p_2p_1}(f,z)v_{p_1} @f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         H_{p_2p_1}(f,z)v_{p_1}=\left.\frac{\partial}{\partial r} \nabla_{p_2} \left\langle f(x, p_1+ r\,v_{p_1}, p_2),z\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  @f$ x @f$  is the solution,  @f$ p_1 @f$  is a first distributed parameter which can be extruded,  @f$ p_2 @f$  is a potentially different second distributed parameter which can be extruded too,
 *  @f$  g @f$  is the response function, @f$  f @f$  is the residual, @f$  z  @f$ is the Lagrange multiplier vector,  @f$ v_{x} @f$  is a direction vector
 *  with the same dimension as the vector  @f$ x @f$, and @f$ v_{p_1} @f$  is a direction vector with the same dimension as the vector  @f$ p_1 @f$.
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
