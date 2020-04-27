//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP
#define PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP

#include "PHAL_ScatterScalarResponse.hpp"

#include "Shards_CellTopologyData.h"

namespace PHAL {

/** \brief Handles scattering of separable scalar response functions into epetra
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
  PHX::MDField<const ScalarT> local_response;
  PHX::MDField<ScalarT> local_response_eval;
};

/** \brief Handles scattering of separable scalar response functions into epetra
 * data structures.
 *
 * A separable response function is one that is a sum of respones across cells.
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

  void evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData /* d */,
                                                        const std::string& /* sidesetName */,
                                                        Teuchos::RCP<const CellTopologyData> /* cellTopo */) {}

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

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>,
    public SeparableScatterScalarResponseBase<PHAL::AlbanyTraits::Jacobian, Traits> {
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
  void evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData d, std::string& sideset, Teuchos::RCP<const CellTopologyData> cellTopo);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::Jacobian EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
  }
protected:
  int numNodes;
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>,
    public SeparableScatterScalarResponseBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
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
  void evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData /* d */,
                                                        const std::string& /* sidesetName */,
                                                        Teuchos::RCP<const CellTopologyData> /* cellTopo */) {}
  void postEvaluate(typename Traits::PostEvalData d);

protected:
  typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
  }
  int numNodes;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};


template<typename Traits>
class SeparableScatterScalarResponseWithExtrudedParams<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  SeparableScatterScalarResponseWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl) {
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluateFields(typename Traits::EvalData d);

protected:
  SeparableScatterScalarResponseWithExtrudedParams() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits>::setup(p,dl);
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  }

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;

};

// **************************************************************
// HessianVec
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::HessianVec, Traits>,
    public SeparableScatterScalarResponseBase<PHAL::AlbanyTraits::HessianVec, Traits> {
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
  void evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData d, std::string& sideset, Teuchos::RCP<const CellTopologyData> cellTopo);
  void postEvaluate(typename Traits::PostEvalData d);

protected:
  typedef PHAL::AlbanyTraits::HessianVec EvalT;
  SeparableScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->extent(1);
  }
  int numNodes;

private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
};


template<typename Traits>
class SeparableScatterScalarResponseWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>
  : public SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>  {
public:
  SeparableScatterScalarResponseWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl) {
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluateFields(typename Traits::EvalData d);

protected:
  SeparableScatterScalarResponseWithExtrudedParams() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>::setup(p,dl);
    extruded_params_levels = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  }

private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;

};

// **************************************************************
} // namespace PHAL

#endif // PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSE_HPP
