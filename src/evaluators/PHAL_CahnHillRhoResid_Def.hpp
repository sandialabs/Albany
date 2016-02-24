//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
CahnHillRhoResid<EvalT, Traits>::
CahnHillRhoResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  rhoGrad     (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  chemTerm    (p.get<std::string>                   ("Chemical Energy Term"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rhoResidual (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{

  haveNoise = p.get<bool>("Have Noise");

  this->addDependentField(wBF);
  this->addDependentField(rhoGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(chemTerm);
  this->addEvaluatedField(rhoResidual);

  if(haveNoise){
    noiseTerm = PHX::MDField<ScalarT, Cell, QuadPoint> (p.get<std::string>("Langevin Noise Term"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    this->addDependentField(noiseTerm);
  }

  gamma = p.get<double>("gamma Value");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  gamma_term.resize(worksetSize, numQPs, numDims);

  this->setName("CahnHillRhoResid" );

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillRhoResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(rhoGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(chemTerm,fm);
  if(haveNoise)
    this->utils.setFieldData(noiseTerm,fm);

  this->utils.setFieldData(rhoResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillRhoResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

// Form Equation 2.2

  typedef Intrepid2::FunctionSpaceTools FST;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    for (std::size_t qp=0; qp < numQPs; ++qp) 
      for (std::size_t i=0; i < numDims; ++i) 

        gamma_term(cell, qp, i) = rhoGrad(cell,qp,i) * gamma; 

  FST::integrate<ScalarT>(rhoResidual, gamma_term, wGradBF, Intrepid2::COMP_CPP, false); // "false" overwrites

  FST::integrate<ScalarT>(rhoResidual, chemTerm, wBF, Intrepid2::COMP_CPP, true); // "true" sums into

  if(haveNoise)

    FST::integrate<ScalarT>(rhoResidual, noiseTerm, wBF, Intrepid2::COMP_CPP, true); // "true" sums into


}

template<typename EvalT, typename Traits>
typename CahnHillRhoResid<EvalT, Traits>::ScalarT&
CahnHillRhoResid<EvalT, Traits>::getValue(const std::string &n) {

  if (n == "gamma") 

    return gamma;

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
				"Error! Logic error in getting parameter " << n <<
				" in CahnHillRhoResid::getValue()" << std::endl);
    return gamma;
  }

}

//**********************************************************************
}

