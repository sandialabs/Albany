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
EnergyPotential<EvalT, Traits>::
EnergyPotential(const Teuchos::ParameterList& p) :
  defgrad          (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  energy           (p.get<std::string>                   ("EnergyPotential Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(defgrad);
  this->addDependentField(J);
  this->addDependentField(elasticModulus);
  this->addDependentField(poissonsRatio);

  this->addEvaluatedField(energy);

  this->setName("Energy Potential"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void EnergyPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(energy,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(elasticModulus,fm);
  this->utils.setFieldData(poissonsRatio,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EnergyPotential<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT kappa;
  ScalarT mu;
  ScalarT Jm23;
  ScalarT trace;

  numCells = workset.numCells;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      kappa = elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
      mu    = elasticModulus(cell,qp) / ( 2. * ( 1. - poissonsRatio(cell,qp) ) );
      Jm23  = std::pow(J(cell,qp), -2./3.);
      trace = 0.0;

      for (std::size_t i=0; i < numDims; ++i) 
	for (std::size_t j=0; j < numDims; ++j) 
	  trace += defgrad(cell,qp,i,j) * defgrad(cell,qp,i,j);
      
      energy(cell,qp) = 0.5 * kappa * ( 0.5 * ( J(cell,qp) * J(cell,qp) - 1.0 ) - std::log( J(cell,qp) ) )
	+ 0.5 * mu * Jm23 * ( trace - 3.0 );	  
    }
  }
}

//**********************************************************************
}

