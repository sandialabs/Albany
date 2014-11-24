//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
Stress<EvalT, Traits>::
Stress(const Teuchos::ParameterList& p) :
  strain           (p.get<std::string>                   ("Strain Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  elasticModulus   (p.get<double>                        ("Elastic Modulus")),
  poissonsRatio    (p.get<double>                        ("Poissons Ratio"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(strain);
  this->addEvaluatedField(stress);

  this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void Stress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(strain,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Stress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT lambda, mu;

  switch (numDims) {
  case 1:
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	lambda = ( elasticModulus * poissonsRatio ) / ( ( 1 + poissonsRatio ) * ( 1 - 2 * poissonsRatio ) );
	mu = elasticModulus / ( 2 * ( 1 + poissonsRatio ) );
	stress(cell,qp,0,0) = (lambda + 2.0 * mu) * strain(cell,qp,0,0);
      }
    }
    break;
  case 2:
    // Compute Stress (with the plane strain assumption for now)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	lambda = ( elasticModulus * poissonsRatio ) / ( ( 1 + poissonsRatio ) * ( 1 - 2 * poissonsRatio ) );
	mu = elasticModulus / ( 2 * ( 1 + poissonsRatio ) );
	stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );
	stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );
	stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );
	stress(cell,qp,1,0) = stress(cell,qp,0,1); 
      }
    }
    break;
  case 3:
    // Compute Stress
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	lambda = ( elasticModulus * poissonsRatio ) / ( ( 1 + poissonsRatio ) * ( 1 - 2 * poissonsRatio ) );
	mu = elasticModulus / ( 2 * ( 1 + poissonsRatio ) );
	stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
	stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
	stress(cell,qp,2,2) = 2.0 * mu * ( strain(cell,qp,2,2) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
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
}

//**********************************************************************
}

