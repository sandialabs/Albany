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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
PisdWdF<EvalT, Traits>::
PisdWdF(const Teuchos::ParameterList& p) :
  defgrad          (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  P                (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(defgrad);
  this->addDependentField(elasticModulus);
  this->addDependentField(poissonsRatio);

  this->addEvaluatedField(P);

  this->setName("P by AD of dWdF"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PisdWdF<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(P,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(elasticModulus,fm);
  this->utils.setFieldData(poissonsRatio,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PisdWdF<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT kappa;
  ScalarT mu;

  // Leading dimension of 1 added so we can use Intrepid::det
  Intrepid::FieldContainer<EnergyFadType> F(1,numDims,numDims);

  // Allocate F ( = defgrad of derivative types) and seed with identity derivs
  for (std::size_t i=0; i < numDims; ++i) 
  {
    for (std::size_t j=0; j < numDims; ++j) 
    {
      F(0,i,j) = EnergyFadType(numDims*numDims, 0.0); // 0.0 will be overwriten below
      F(0,i,j).fastAccessDx(i*numDims + j) = 1.0;
    }
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      kappa = elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
      mu    = elasticModulus(cell,qp) / ( 2. * ( 1. + poissonsRatio(cell,qp) ) );

      // Fill F with defgrad for value. Derivs already seeded with identity.
      for (std::size_t i=0; i < numDims; ++i) 
        for (std::size_t j=0; j < numDims; ++j) 
           F(0,i,j).val() = defgrad(cell, qp, i, j);

      // Call energy funtional (can make a library of these)
      EnergyFadType W = computeEnergy(kappa, mu, F);

      // Extract stress from derivs of energy
      for (std::size_t i=0; i < numDims; ++i) 
        for (std::size_t j=0; j < numDims; ++j) 
           P(cell, qp, i, j) = W.fastAccessDx(i*numDims + j);
      
    }
  }
}

//**********************************************************************

template<typename EvalT, typename Traits>
typename PisdWdF<EvalT, Traits>::EnergyFadType
PisdWdF<EvalT, Traits>::computeEnergy(ScalarT& kappa, ScalarT& mu, Intrepid::FieldContainer<EnergyFadType>& F) 
{
  // array of length 1 so Intrepid::det can be called.
  Intrepid::FieldContainer<EnergyFadType> Jvec(1);
  Intrepid::RealSpaceTools<EnergyFadType>::det(Jvec, F);
  EnergyFadType& J =  Jvec(0);
  EnergyFadType Jm23  = std::pow(J, -2./3.);
  EnergyFadType trace = 0.0;

  for (std::size_t i=0; i < numDims; ++i) 
    for (std::size_t j=0; j < numDims; ++j) 
      trace += F(0,i,j) * F(0,i,j);
      
  EnergyFadType kappa_div_2 = EnergyFadType(0.5 * kappa);
  EnergyFadType mu_div_2 = EnergyFadType(0.5 * mu);

  return ( kappa_div_2 * ( 0.5 * ( J * J - 1.0 ) - std::log(J) )
	   + mu_div_2 * ( Jm23 * trace - 3.0 ));
}

} // LCM
