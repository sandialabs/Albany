/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
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
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

#define SQR(X) ((X)*(X))

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

  this->addDependentField(rho);
  this->addDependentField(w);

  this->addEvaluatedField(chemTerm);

  b = p.get<double>("b Value");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("CahnHillChemTerm"+PHX::TypeString<EvalT>::value);
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
      for (std::size_t i=0; i < numDims; ++i) 

          chemTerm(cell,qp,i) = w(cell,qp,i) - 0.25 * SQR(SQR(rho(cell,qp,i)) - SQR(b));

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

