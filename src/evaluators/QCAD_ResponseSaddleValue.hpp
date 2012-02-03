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


#ifndef QCAD_RESPONSESADDLEVALUE_HPP
#define QCAD_RESPONSESADDLEVALUE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "QCAD_EvaluatorTools.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"


/** 
 * \brief Response Description
 */
namespace QCAD 
{
  template<typename EvalT, typename Traits>
  class ResponseSaddleValue : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>,
    public EvaluatorTools<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseSaddleValue(Teuchos::ParameterList& p,
			const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);
  
    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);

    /*int numResponses() const { 
      return 5; 
      }*/
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;
    void getCellQuantities(const std::size_t cell, ScalarT& cellVol, 
			   typename EvalT::ScalarT& fieldVal, typename EvalT::ScalarT& retFieldVal, 
			   std::vector<typename EvalT::ScalarT>& fieldGrad) const;

    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVertices;
  
    Teuchos::RCP<QCAD::SaddleValueResponseFunction> svResponseFn;
  
    PHX::MDField<ScalarT> field;
    PHX::MDField<ScalarT> fieldGradient;
    PHX::MDField<ScalarT> retField;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec_vertices; //not currently needed
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    
    std::string fieldName;
    std::string fieldGradientName;
    std::string retFieldName;
    
    bool bReturnSameField;
    double scaling, retScaling;
    

    Teuchos::RCP<PHX::FieldTag> response_operation;
  };
	
}

#endif
