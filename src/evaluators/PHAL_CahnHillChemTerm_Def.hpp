//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"


template<typename ScalarT>
inline ScalarT Sqr (const ScalarT& num) {
  return num * num;
}

namespace PHAL {


//**********************************************************************
template<typename EvalT, typename Traits>
CahnHillChemTerm<EvalT, Traits>::
CahnHillChemTerm(const Teuchos::ParameterList& p) :
  rho        (p.get<std::string>                   ("Rho QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  w          (p.get<std::string>                   ("W QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  chemTerm   (p.get<std::string>                ("Chemical Energy Term"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
 
{

  b = p.get<double>("b Value");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(rho);
  this->addDependentField(w);

  this->addEvaluatedField(chemTerm);

  this->setName("CahnHillChemTerm" );

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillChemTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(w,fm);

  this->utils.setFieldData(chemTerm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillChemTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

// Equations 1.1 and 2.2 in Garcke, Rumpf, and Weikard
// psi(rho) = 0.25 * (rho^2 - b^2)^2

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    for (std::size_t qp=0; qp < numQPs; ++qp)

      // chemTerm(cell, qp) = 0.25 * Sqr(Sqr(rho(cell, qp)) - Sqr(b)) - w(cell, qp);
      chemTerm(cell, qp) = ( Sqr<ScalarT>(rho(cell, qp)) - Sqr<ScalarT>(b) ) * rho(cell, qp) - w(cell, qp);

}

template<typename EvalT, typename Traits>
typename CahnHillChemTerm<EvalT, Traits>::ScalarT&
CahnHillChemTerm<EvalT, Traits>::getValue(const std::string &n) {

  if (n == "b") 

    return b;

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
				"Error! Logic error in getting parameter " << n <<
				" in CahnHillChemTerm::getValue()" << std::endl);
    return b;
  }

}


}

