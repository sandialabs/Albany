//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_RESPONSEFIELDAVERAGE_HPP
#define QCAD_RESPONSEFIELDAVERAGE_HPP

#include "QCAD_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace QCAD {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseFieldAverage : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseFieldAverage(Teuchos::ParameterList& p,
			 const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::size_t numQPs;
    std::size_t numDims;
    
    PHX::MDField<const ScalarT> field;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint> weights;
    
    std::string fieldName;
    Teuchos::RCP< MeshRegion<EvalT, Traits> > opRegion;
  };
	
}

#endif
