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


#ifndef QCAD_RESPONSEFIELDVALUE_HPP
#define QCAD_RESPONSEFIELDVALUE_HPP

#include "PHAL_ResponseBase.hpp"

/** 
 * \brief Response Description
 */
namespace QCAD 
{
  template<typename EvalT, typename Traits>
  class ResponseFieldValue : 
    public PHAL::ResponseBase<EvalT, Traits>
  {
     public:
  	  typedef typename EvalT::ScalarT ScalarT;
  	  typedef typename EvalT::MeshScalarT MeshScalarT;
    
          ResponseFieldValue(Teuchos::ParameterList& p);
  
	  void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);
  
	  void evaluateFields(typename Traits::EvalData d);
	  
	private:
	  Teuchos::RCP<const Teuchos::ParameterList>
	  getValidResponseParameters() const;

	  std::size_t numQPs;
	  std::size_t numDims;
	  PHX::MDField<ScalarT> opField;
	  PHX::MDField<ScalarT> retField;
	  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
	  PHX::MDField<ScalarT,Cell,QuadPoint> weights;
          bool bOpFieldIsVector, bRetFieldIsVector;

          std::string operation;
          std::string opFieldName;
          std::string retFieldName;
          std::string opDomain;
          std::string ebName;

          bool bReturnOpField;
          bool opX, opY, opZ;
          bool limitX, limitY, limitZ;
          double xmin, xmax, ymin, ymax, zmin, zmax;
  };
	
}

#endif
