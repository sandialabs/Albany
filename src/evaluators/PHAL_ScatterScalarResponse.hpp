/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
// Stochastic Galerkin Residual 
// **************************************************************
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

// **************************************************************
}

#endif
