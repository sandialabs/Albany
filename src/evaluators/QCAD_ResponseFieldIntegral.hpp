//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_RESPONSEFIELDINTEGRAL_HPP
#define QCAD_RESPONSEFIELDINTEGRAL_HPP

#include "QCAD_MeshRegion.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace QCAD {

  //Maximum number of fields that can be multiplied together as an integrand
  //  (Note: must be less than the number of bits in an integer)
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
    std::vector<std::string> fieldNames_Imag;
    bool bReturnImagPart;
    
    std::size_t numQPs;
    std::size_t numDims;
    
    std::vector<bool> fieldIsComplex;
    std::vector<bool> conjugateFieldFlag;
    std::vector<PHX::MDField<const ScalarT,Cell,QuadPoint> > fields;
    std::vector<PHX::MDField<const ScalarT,Cell,QuadPoint> > fields_Imag;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint> weights;
    Teuchos::Array<int> field_components;
    
    double length_unit_in_m; // length unit for input and output mesh
    double scaling;          // scaling factor due to difference in mesh and integrand units
    bool bPositiveOnly;
    Teuchos::RCP< MeshRegion<EvalT, Traits> > opRegion;

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  };
	
}

#endif
