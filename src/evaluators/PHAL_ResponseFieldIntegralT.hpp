//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_FIELD_INTEGRALT_HPP
#define PHAL_RESPONSE_FIELD_INTEGRALT_HPP

#include "PHAL_SeparableScatterScalarResponseT.hpp"

namespace PHAL {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseFieldIntegralT : 
    public PHAL::SeparableScatterScalarResponseT<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ResponseFieldIntegralT(Teuchos::ParameterList& p,
			  const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    PHX::MDField<ScalarT> field;
    PHX::MDField<MeshScalarT> coordVec;
    PHX::MDField<MeshScalarT> weights;
    PHX::DataLayout::size_type field_rank;
    std::vector<PHX::DataLayout::size_type> field_dims;
    Teuchos::Array<int> field_components;
    std::size_t numQPs;
    std::size_t numDims;

    std::vector<std::string> ebNames;    
    double scaling;
    bool limitX, limitY, limitZ;
    double xmin, xmax, ymin, ymax, zmin, zmax;
  };
	
}

#endif
