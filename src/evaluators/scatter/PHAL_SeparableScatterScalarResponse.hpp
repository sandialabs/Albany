//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP
#define PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP

#include "PHAL_ScatterScalarResponse.hpp"

#include "Shards_CellTopologyData.h"
#include "Teuchos_TestForException.hpp"

#include <string>

namespace PHAL {

/** \brief Handles scattering of separable scalar response functions into (Tpetra)
 * data structures.
 *
 * Base implementation useable by specializations below
 */
template<typename EvalT, typename Traits>
class SeparableScatterScalarResponseBase
  : public virtual PHX::EvaluatorWithBaseImpl<Traits>,
    public virtual PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  SeparableScatterScalarResponseBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData /* d */) {}

protected:

  // Default constructor for child classes
  SeparableScatterScalarResponseBase() {}

  // Child classes should call setup once p is filled out
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);

protected:

  typedef typename EvalT::ScalarT ScalarT;
  bool stand_alone;
  // Note: these two fields have the same Field Name, so they alias each other.
  //       The former is used to fill the Thyra response vectors, while the
  //       latter is computed by derived classes. We really only need the
  //       former in case the response field is provided externally. In practice,
  //       all derived classes *compute* the response field.
  PHX::MDField<const ScalarT> local_response;
  PHX::MDField<ScalarT> local_response_eval;
};

/** \brief Handles scattering of separable scalar response functions into Tpetra
 * data structures.
 *
 * A separable response function is one that is a sum of responses across cells.
 * In this case we can compute the Jacobian in a generic fashion.
 */
template <typename EvalT, typename Traits>
class SeparableScatterScalarResponse :
    public ScatterScalarResponse<EvalT, Traits>,
    public SeparableScatterScalarResponseBase<EvalT,Traits> {
public:
  SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
    ScatterScalarResponse<EvalT,Traits>(p,dl) {}

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBase<EvalT,Traits>::postRegistrationSetup(d,vm);
  }

  void evaluateFields(typename Traits::EvalData /* d */) {}

  void evaluate2DFieldsDerivativesDueToColumnContraction(typename Traits::EvalData /* d */,
                                                        const std::string& /* sidesetName */ ) {}

protected:
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponse<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
};


template<typename EvalT, typename Traits> class SeparableScatterScalarResponse;

template<typename EvalT, typename Traits>
class SeparableScatterScalarResponseWithExtrudedParams
  : public SeparableScatterScalarResponse<EvalT, Traits> {

public:

  SeparableScatterScalarResponseWithExtrudedParams(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
                                SeparableScatterScalarResponse<EvalT, Traits>(p,dl) {
//    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d,vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(d);
  }

protected:

  typedef typename EvalT::ScalarT ScalarT;
  //Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;

  SeparableScatterScalarResponseWithExtrudedParams() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
  }
};


// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// Equation offset and level of a 2D solution field defined at one level of an extruded mesh,
// read from the "Parameters From Problem" list of p ("Extruded 2D Field Offset", "Extruded 2D
// Field Level"), or -1 if none (see SeparableScatterScalarResponse<Jacobian>).
inline void
getExtruded2DField (const Teuchos::ParameterList& p, int& offset, int& level)
{
  using PL = Teuchos::ParameterList;

  offset = -1;
  level  = -1;

  const std::string pfp_name = "Parameters From Problem";
  if (!p.isType<Teuchos::RCP<PL>>(pfp_name)) {
    return;
  }
  const auto pfp = p.get<Teuchos::RCP<PL>>(pfp_name);
  if (pfp.is_null() || !pfp->isParameter("Extruded 2D Field Offset")) {
    return;
  }
  offset = pfp->get<int>("Extruded 2D Field Offset");
  level  = pfp->get<int>("Extruded 2D Field Level");
  TEUCHOS_TEST_FOR_EXCEPTION (offset<0 || level<0, std::logic_error,
      "Error! Invalid extruded 2D field for the responses: offset " << offset
      << ", level " << level << ".\n");
}

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponse<AlbanyTraits::Jacobian,Traits>
  : public ScatterScalarResponseBase<AlbanyTraits::Jacobian, Traits>,
    public SeparableScatterScalarResponseBase<AlbanyTraits::Jacobian, Traits> {
public:
  SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBase<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void evaluate2DFieldsDerivativesDueToColumnContraction(typename Traits::EvalData d, std::string& sideset);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef AlbanyTraits::Jacobian EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
    setupExtruded2DField(p);
  }

  // A 2D field defined at one level of an extruded mesh (e.g., the thickness of
  // LandIce::StokesFOThickness) has dofs at all the nodes, but only the ones at its level are
  // the dofs of its equation: its evaluators (e.g., LandIce::Gather2DField) gather the field at
  // any node of a column from the dof at the field level, storing the derivative in the slot of
  // the node. The derivatives w.r.t. such a field are then scattered to the dofs at the field
  // level (as PHAL::ScatterResidualWithExtrudedField does for the residual), rather than to the
  // dofs of the nodes. The field is given by the equation offset and level in the "Parameters
  // From Problem" list ("Extruded 2D Field Offset", "Extruded 2D Field Level"), if any.
  void setupExtruded2DField (const Teuchos::ParameterList& p);
  void evaluateExtruded2DFieldDerivatives (typename Traits::EvalData d);
protected:
  int numNodes;
  int offset2DField = -1;  // equation offset of the extruded 2D field (-1 if none)
  int fieldLevel2D  = -1;  // level of the extruded 2D field
  using ExecutionSpace = typename PHX::ExecutionSpace;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
private:
  typedef typename AlbanyTraits::Jacobian::ScalarT ScalarT;
protected:
  MDFieldVectorRight<const ScalarT> global_response_reader;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterScalarResponseBase<AlbanyTraits::DistParamDeriv, Traits>,
    public SeparableScatterScalarResponseBase<AlbanyTraits::DistParamDeriv, Traits> {
public:
  SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBase<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void evaluate2DFieldsDerivativesDueToColumnContraction(typename Traits::EvalData /* d */,
                                                        const std::string& /* sidesetName */) {}
  void postEvaluate(typename Traits::PostEvalData d);

protected:
  typedef AlbanyTraits::DistParamDeriv EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
  }
  int numNodes;

private:
  typedef typename AlbanyTraits::DistParamDeriv::ScalarT ScalarT;

protected:
  using ExecutionSpace = typename PHX::ExecutionSpace;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
  MDFieldVectorRight<const ScalarT> global_response_reader;
  
};


template<typename Traits>
class SeparableScatterScalarResponseWithExtrudedParams<AlbanyTraits::DistParamDeriv,Traits>
  : public SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>  {
public:
  SeparableScatterScalarResponseWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
   : Base(p,dl)
  {
    const auto prob_params = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
    extruded_params_levels = prob_params->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  }

  void evaluateFields(typename Traits::EvalData d);

protected:
  using Base = SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>;
  SeparableScatterScalarResponseWithExtrudedParams() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv,Traits>::setup(p,dl);
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  }

private:
  typedef typename AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;

};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the SeparableScatterScalarResponse Class for AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to scatter the solution for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1,\boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where  \f$\boldsymbol{x}\f$  is the solution, \f$\boldsymbol{p}_1\f$  is a first parameter, \f$\boldsymbol{p}_2\f$ is a potentially different second parameter,
 *  \f$g\f$  is the response function, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$  is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1}\f$  is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This scatter is used when calling:
 * <ul>
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResponse_HessVecProd_pp.
 * </ul>
 */

template<typename Traits>
class SeparableScatterScalarResponse<AlbanyTraits::HessianVec,Traits>
  : public ScatterScalarResponseBase<AlbanyTraits::HessianVec, Traits>,
    public SeparableScatterScalarResponseBase<AlbanyTraits::HessianVec, Traits> {
public:
  SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBase<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void evaluate2DFieldsDerivativesDueToColumnContraction(typename Traits::EvalData d, std::string& sideset);
  void postEvaluate(typename Traits::PostEvalData d);

protected:
  typedef AlbanyTraits::HessianVec EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
    getExtruded2DField(p,offset2DField,fieldLevel2D);
  }
  int numNodes;
  // Extruded 2D field
  int offset2DField = -1;
  int fieldLevel2D  = -1;

private:
  typedef typename AlbanyTraits::HessianVec::ScalarT ScalarT;

protected:
  using ExecutionSpace = typename PHX::ExecutionSpace;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
  MDFieldVectorRight<const ScalarT> global_response_reader;
};


template<typename Traits>
class SeparableScatterScalarResponseWithExtrudedParams<AlbanyTraits::HessianVec,Traits>
  : public SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>  {
public:
  SeparableScatterScalarResponseWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>(p,dl) {
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluateFields(typename Traits::EvalData d);

protected:
  using Base = SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>;
  SeparableScatterScalarResponseWithExtrudedParams() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    SeparableScatterScalarResponse<AlbanyTraits::HessianVec,Traits>::setup(p,dl);
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  }

private:
  typedef typename AlbanyTraits::HessianVec::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;

};

// **************************************************************
} // namespace PHAL

#endif // PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP
