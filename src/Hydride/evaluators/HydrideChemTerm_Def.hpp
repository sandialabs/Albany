//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

namespace HYD {


//**********************************************************************
template<typename EvalT, typename Traits>
HydrideChemTerm<EvalT, Traits>::
HydrideChemTerm(const Teuchos::ParameterList& p) :
  c          (p.get<std::string>                   ("C QP Variable Name"),
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

  this->addDependentField(c);
  this->addDependentField(w);

  this->addEvaluatedField(chemTerm);

  this->setName("HydrideChemTerm"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideChemTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(c,fm);
  this->utils.setFieldData(w,fm);

  this->utils.setFieldData(chemTerm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideChemTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

// Equations 1.1 and 2.2 in Garcke, Rumpf, and Weikard
// psi(c) = 0.25 * (c^2 - b^2)^2

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    for (std::size_t qp=0; qp < numQPs; ++qp)

//        chemTerm(cell, qp) = 0.25 * Sqr(Sqr(c(cell, qp)) - Sqr(b)) - w(cell, qp);
      chemTerm(cell, qp) = ( Sqr<ScalarT>(c(cell, qp)) - Sqr<ScalarT>(b) ) * c(cell, qp) - w(cell, qp);

}

template<typename EvalT, typename Traits>
typename HydrideChemTerm<EvalT, Traits>::ScalarT&
HydrideChemTerm<EvalT, Traits>::getValue(const std::string &n) {

  if (n == "b") 

    return b;

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
				"Error! Logic error in getting parameter " << n <<
				" in HydrideChemTerm::getValue()" << std::endl);
    return b;
  }

}


}

