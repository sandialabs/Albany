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
#include "Intrepid_RealSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
DetDefGrad<EvalT, Traits>::
DetDefGrad(const Teuchos::ParameterList& p) :
  J         (p.get<std::string>                   ("DetDefGrad Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  defgrad   (p.get<std::string>                   ("DefGrad Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  this->addDependentField(defgrad);

  this->addEvaluatedField(J);

  this->setName("DetDefGrad");

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DetDefGrad<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(defgrad,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DetDefGrad<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //typedef Intrepid::RealSpaceTools RST;
  //int numCells = workset.numCells;
  Intrepid::RealSpaceTools<ScalarT>::det(J, defgrad);
}

//**********************************************************************
}

