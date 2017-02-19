//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_FIELD_INTEGRAL_HPP
#define PHAL_RESPONSE_FIELD_INTEGRAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace PHAL {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseFieldIntegral : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ResponseFieldIntegral(Teuchos::ParameterList& p,
			  const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    PHX::MDField<const ScalarT> field;
    PHX::MDField<const MeshScalarT> coordVec;
    PHX::MDField<const MeshScalarT> weights;
    PHX::index_size_type field_rank;
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
