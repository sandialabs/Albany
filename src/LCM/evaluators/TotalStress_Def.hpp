//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
TotalStress<EvalT, Traits>::
TotalStress(const Teuchos::ParameterList& p) :
  effStress           (p.get<std::string>                   ("Effective Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  biotCoefficient  (p.get<std::string>                   ("Biot Coefficient Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  porePressure    (p.get<std::string>                   ("QP Variable Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Total Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(effStress);
  this->addDependentField(biotCoefficient);
  this->addDependentField(porePressure);

  this->addEvaluatedField(stress);

  this->setName("TotalStress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void TotalStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(effStress,fm);
  this->utils.setFieldData(biotCoefficient,fm);
  this->utils.setFieldData(porePressure,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TotalStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  for (std::size_t dim=0; dim<numDims; ++ dim) {
    		  for (std::size_t j=0; j<numDims; ++ j) {
	              stress(cell,qp,dim,j) = effStress(cell,qp, dim,j);
    		  }
    	  }
      }
    }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  for (std::size_t dim=0; dim<numDims; ++ dim) {
	              stress(cell,qp,dim,dim) -= biotCoefficient(cell,qp)*
	            		                     porePressure(cell,qp);
    	  }
      }
    }
/*  ScalarT lambda, mu; // B is the Biot coefficient 1 - K/K(s) where

  switch (numDims) {
  case 1:
    Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT>(stress, elasticModulus, strain);
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
        	  stress(cell, qp) = stress(cell, qp) - porePressure(cell,qp);
          }
    }
    break;
  case 2:
    // Compute Stress (with the plane strain assumption for now)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	lambda = ( elasticModulus(cell,qp) * poissonsRatio(cell,qp) ) / ( ( 1 + poissonsRatio(cell,qp) ) * ( 1 - 2 * poissonsRatio(cell,qp) ) );
	mu = elasticModulus(cell,qp) / ( 2 * ( 1 + poissonsRatio(cell,qp) ) );
	stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) ) - biotCoefficient(cell,qp)*porePressure(cell,qp) ;
	stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) ) - biotCoefficient(cell,qp)*porePressure(cell,qp);
	stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );
	stress(cell,qp,1,0) = stress(cell,qp,0,1); 
      }
    }
    break;
  case 3:
    // Compute Stress
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	lambda = ( elasticModulus(cell,qp) * poissonsRatio(cell,qp) ) / ( ( 1 + poissonsRatio(cell,qp) ) * ( 1 - 2 * poissonsRatio(cell,qp) ) );
	mu = elasticModulus(cell,qp) / ( 2 * ( 1 + poissonsRatio(cell,qp) ) );
	stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) )
		                  - biotCoefficient(cell,qp)*porePressure(cell,qp);
	stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) )
		                  - biotCoefficient(cell,qp)*porePressure(cell,qp);
	stress(cell,qp,2,2) = 2.0 * mu * ( strain(cell,qp,2,2) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) )
		                  - biotCoefficient(cell,qp)*porePressure(cell,qp);
	stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );
	stress(cell,qp,1,2) = 2.0 * mu * ( strain(cell,qp,1,2) );
	stress(cell,qp,2,0) = 2.0 * mu * ( strain(cell,qp,2,0) );
	stress(cell,qp,1,0) = stress(cell,qp,0,1); 
	stress(cell,qp,2,1) = stress(cell,qp,1,2); 
	stress(cell,qp,0,2) = stress(cell,qp,2,0); 
      }
    }
    break;
  }
  */
}

//**********************************************************************
}

