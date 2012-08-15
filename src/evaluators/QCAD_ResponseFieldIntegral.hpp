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


#ifndef QCAD_RESPONSEFIELDINTEGRAL_HPP
#define QCAD_RESPONSEFIELDINTEGRAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace QCAD {

  const int MAX_FIELDNAMES_IN_INTEGRAL = 10;
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

    std::vector<std::string> fieldNames;
    std::vector<std::string> ebNames;
    
    std::size_t numQPs;
    std::size_t numDims;
    
    std::vector<PHX::MDField<ScalarT,Cell,QuadPoint> > fields;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    Teuchos::Array<int> field_components;
    
    double length_unit_in_m; // length unit for input and output mesh
    double scaling;          // scaling factor due to difference in mesh and integrand units
    bool bPositiveOnly;
    bool limitX, limitY, limitZ;
    double xmin, xmax, ymin, ymax, zmin, zmax;

  };
	
}

#endif
