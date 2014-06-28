//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_SCALAR_RESPONSE_HPP
#define PHAL_SCATTER_SCALAR_RESPONSE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Albany_ProblemUtils.hpp"

namespace PHAL {

/** \brief Handles scattering of scalar response functions into epetra
 * data structures.
 *
 * Base implementation useable by specializations below
 */
template<typename EvalT, typename Traits>
class ScatterScalarResponseBase
  : public virtual PHX::EvaluatorWithBaseImpl<Traits>,
    public virtual PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ScatterScalarResponseBase(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d) {}

  //! Get tag for response field (to determine the number of responses)
  Teuchos::RCP<const PHX::FieldTag>
  getResponseFieldTag() const {
    return global_response.fieldTag().clone();
  }

  //! Get tag for evaluated field (required field for field manager)
  Teuchos::RCP<const PHX::FieldTag>
  getEvaluatedFieldTag() const {
    return scatter_operation;
  }

protected:

  // Default constructor for child classes
  ScatterScalarResponseBase() {}

  // Child classes should call setup once p is filled out
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);

  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

protected:

  typedef typename EvalT::ScalarT ScalarT;
  PHX::MDField<ScalarT> global_response;
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
};

template<typename EvalT, typename Traits> class ScatterScalarResponse {};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::Residual EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian -- No implementation can be provided
// **************************************************************

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::Tangent EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed Parameter Derivative -- No implementation can be provided
// **************************************************************

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::SGResidual,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::SGResidual EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Jacobian -- no implementation can be provided
// **************************************************************

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::SGTangent,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::SGTangent EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::MPResidual EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Jacobian -- No implementation can be provided
// **************************************************************

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class ScatterScalarResponse<PHAL::AlbanyTraits::MPTangent,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  ScatterScalarResponse(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::MPTangent EvalT;
  ScatterScalarResponse() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
  }
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif //ALBANY_SG_MP

// **************************************************************
}

#endif
