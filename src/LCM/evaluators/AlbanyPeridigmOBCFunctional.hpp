//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP
#define ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace PHAL {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class AlbanyPeridigmOBCFunctional : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    AlbanyPeridigmOBCFunctional(Teuchos::ParameterList& p,
				const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    PHX::MDField<MeshScalarT> referenceCoordinates;
    PHX::MDField<ScalarT> displacement;

    std::size_t numQPs;
    std::size_t numDims;
  };
	
}

#endif
