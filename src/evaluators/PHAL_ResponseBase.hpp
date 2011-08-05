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


#ifndef PHAL_RESPONSEBASE_HPP
#define PHAL_RESPONSEBASE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_EvaluatedResponseFunction.hpp"

namespace PHAL
{
/** 
 * \brief Base Class for Response evaluators
 */
  template<typename EvalT, typename Traits>
  class ResponseBaseCommon : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    void setPostProcessingParams(const Teuchos::ParameterList& p) {
      responseFn->setPostProcessingParams(p); };

  protected: 
    typedef typename EvalT::ScalarT ScalarT;
    ResponseBaseCommon(Teuchos::ParameterList& p);

    Teuchos::RCP<PHX::FieldTag> response_operation;
    Teuchos::RCP<Albany::EvaluatedResponseFunction> responseFn;
    int responseIndex;
  };



template<typename EvalT, typename Traits> class ResponseBase;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual 
// **************************************************************
  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::Residual,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::Residual, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
    
    //! Copies local_g to/from workset.  Must be called at the beginning
    //  and end of evaluateFields(...) function in derived classes.
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    
    //! Sets initial values of the responses.  This must be called
    //  by every derived class (usually in its constructor), as it 
    //  sets the number of responses.
    void setInitialValues(const std::vector<double>& initialVals);

    //! local copy of responses for derived classes to use
    std::vector<ScalarT> local_g;
  };


// **************************************************************
// Jacobian
// **************************************************************

  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::Jacobian,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::Jacobian, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p); 
    
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };

// **************************************************************
// Tangent
// **************************************************************

  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::Tangent,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::Tangent, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
    
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };


// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************

  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::SGResidual,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::SGResidual, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
    
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };


// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************

  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::SGJacobian,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::SGJacobian, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
    
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };


// **************************************************************
// Multi-point Residual 
// **************************************************************

  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::MPResidual,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::MPResidual, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
    
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };


// **************************************************************
// Multi-point Jacobian
// **************************************************************
  
  template<typename Traits>
  class ResponseBase<PHAL::AlbanyTraits::MPJacobian,Traits>
    : public ResponseBaseCommon<PHAL::AlbanyTraits::MPJacobian, Traits>  {
  protected:
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    ResponseBase(Teuchos::ParameterList& p);
 
    void beginEvaluateFields(typename Traits::EvalData d);
    void endEvaluateFields(typename Traits::EvalData d);
    void setInitialValues(const std::vector<double>& initialVals);

    std::vector<ScalarT> local_g;
  };

	
}

#endif
