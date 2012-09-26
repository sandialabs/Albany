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


namespace HYD {


//**********************************************************************
template<typename EvalT, typename Traits>
HydrideStressTerm<EvalT, Traits>::
HydrideStressTerm(const Teuchos::ParameterList& p) :

  stress       (p.get<std::string>                   ("Stress Name"),
	             p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  stressTerm   (p.get<std::string>                ("Stress Term"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
 
{

  e = p.get<double>("e Value");

  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];


  this->addDependentField(stress);

  this->addEvaluatedField(stressTerm);

  this->setName("HydrideStressTerm"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideStressTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);

  this->utils.setFieldData(stressTerm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideStressTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

// Equations 2.2 in Garcke, Rumpf, and Weikard
// S:E^\prime(c)

// Calculate the negative of the stress term as this will be what is added to the residual

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    for (std::size_t qp=0; qp < numQPs; ++qp){

      stressTerm(cell, qp) = 0.0;

      for (std::size_t dim=0; dim < numDims; ++dim)

        stressTerm(cell, qp) -= stress(cell, qp, dim, dim) * e;

    }

}

template<typename EvalT, typename Traits>
typename HydrideStressTerm<EvalT, Traits>::ScalarT&
HydrideStressTerm<EvalT, Traits>::getValue(const std::string &n) {

  if (n == "e") 

    return e;

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
				"Error! Logic error in getting parameter " << n <<
				" in HydrideStressTerm::getValue()" << std::endl);
    return e;
  }

}


}

