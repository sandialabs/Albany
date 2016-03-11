//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace HYD {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrideStress<EvalT, Traits>::
HydrideStress(const Teuchos::ParameterList& p) :
  strain           (p.get<std::string>                   ("Strain Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  c                (p.get<std::string>                   ("C QP Variable Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{

   e = p.get<double>("e Value");


  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate and initialize elasticity tensor
  if(numDims == 2){

    ElastTensor.resize(3, 3);  // C_{ijkl} => C_{ij}  i,j = 1..3

    ElastTensor(0, 0) = 2.0;  // C_{1111}
    ElastTensor(0, 1) = 1.0;  // C_{1122}
    ElastTensor(1, 0) = 1.0;  // C_{2211}
    ElastTensor(2, 2) = 10.0;  // C_{1212}

  }
  else {

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Hydride only works in 2D presently: "
            << "See line " << __LINE__ << " in file " << __FILE__ << std::endl);

    ElastTensor.resize(6, 6);  // C_{ijkl} => C_{ij}  i,j = 1..6

  }

  this->addDependentField(c);
  this->addDependentField(strain);

  this->addEvaluatedField(stress);

  this->setName("HydrideStress"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(c,fm);
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(strain,fm);
}

//**********************************************************************
#if 0
template<typename EvalT, typename Traits>
void HydrideStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT lambda, mu;

  switch (numDims) {
  case 1:
    Intrepid2::FunctionSpaceTools::tensorMultiplyDataData<ScalarT>(stress, elasticModulus, strain);
    break;
  case 2:
    // Compute Stress (with the plane strain assumption for now)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {

        lambda = ( elasticModulus(cell,qp) * poissonsRatio(cell,qp) ) 
          / ( ( 1 + poissonsRatio(cell,qp) ) * ( 1 - 2 * poissonsRatio(cell,qp) ) );

        mu = elasticModulus(cell,qp) / ( 2 * ( 1 + poissonsRatio(cell,qp) ) );

        stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) 
          + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );

        stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) 
          + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );

        stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );

        stress(cell,qp,1,0) = stress(cell,qp,0,1); 

      }
    }
    break;

  case 3:
    // Compute Stress
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {

        lambda = ( elasticModulus(cell,qp) * poissonsRatio(cell,qp) ) 
          / ( ( 1 + poissonsRatio(cell,qp) ) * ( 1 - 2 * poissonsRatio(cell,qp) ) );

        mu = elasticModulus(cell,qp) / ( 2 * ( 1 + poissonsRatio(cell,qp) ) );

        stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) 
          + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );

        stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) 
          + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );

        stress(cell,qp,2,2) = 2.0 * mu * ( strain(cell,qp,2,2) ) 
          + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );

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
#endif

template<typename EvalT, typename Traits>
void HydrideStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  switch (numDims) {
  case 1:
    // not implemented
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Hydride only works in 2D presently: "
            << "See line " << __LINE__ << " in file " << __FILE__ << std::endl);
    break;
  case 2:
    // Compute Stress (Garcke Eq. 2.3)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {

        stress(cell,qp,0,0) = ElastTensor(0,0) * (strain(cell,qp,0,0) - e * c(cell, qp))
          + ElastTensor(0, 1) * (strain(cell,qp,1,1) - e * c(cell, qp))
          + ElastTensor(0, 2) * 2.0 * strain(cell,qp,0,1);

        stress(cell,qp,1,1) = ElastTensor(1,0) * (strain(cell,qp,0,0) - e * c(cell, qp))
          + ElastTensor(1, 1) * (strain(cell,qp,1,1) - e * c(cell, qp))
          + ElastTensor(1, 2) * 2.0 * strain(cell,qp,0,1);

        stress(cell,qp,0,1) = ElastTensor(2,0) * (strain(cell,qp,0,0) - e * c(cell, qp))
          + ElastTensor(2, 1) * (strain(cell,qp,1,1) - e * c(cell, qp))
          + ElastTensor(2, 2) * 2.0 * strain(cell,qp,0,1);

        stress(cell,qp,1,0) = stress(cell,qp,0,1); 

      }
    }
    break;

  case 3:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Hydride only works in 2D presently: "
            << "See line " << __LINE__ << " in file " << __FILE__ << std::endl);
    break;
  }
}

template<typename EvalT, typename Traits>
typename HydrideStress<EvalT, Traits>::ScalarT&
HydrideStress<EvalT, Traits>::getValue(const std::string &n) {

  if (n == "e") 

    return e;

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
				"Error! Logic error in getting parameter " << n <<
				" in HydrideStress::getValue()" << std::endl);
    return e;
  }

}


//**********************************************************************
}

