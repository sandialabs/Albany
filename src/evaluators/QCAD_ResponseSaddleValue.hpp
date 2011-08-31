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

#include "PHAL_ResponseBase.hpp"
#include "QCAD_EvaluatorTools.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"

/** 
 * \brief Response Description
 */
namespace QCAD 
{
  template<typename EvalT, typename Traits>
  class ResponseSaddleValue : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public EvaluatorTools<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseSaddleValue(Teuchos::ParameterList& p);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);
  
    void evaluateFields(typename Traits::EvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVertices;
  
    Teuchos::RCP<QCAD::SaddleValueResponseFunction> svResponseFn;
  
    PHX::MDField<ScalarT> field;
    PHX::MDField<ScalarT> retField;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec_vertices;
    PHX::MDField<ScalarT,Cell,QuadPoint> weights;
    
    std::string fieldName;
    std::string retFieldName;
    std::string domain;
    std::string ebName;

    bool bReturnSameField;
    bool limitX, limitY, limitZ;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double retScaling;
    bool bLateralVolumes;

    Teuchos::RCP<PHX::FieldTag> response_operation;
  };
	
}

#endif
