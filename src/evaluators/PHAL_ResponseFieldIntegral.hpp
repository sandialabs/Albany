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
