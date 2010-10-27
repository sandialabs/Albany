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

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
LCG<EvalT, Traits>::
LCG(const Teuchos::ParameterList& p) :
  lcg       (p.get<std::string>                   ("LCG Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  defgrad   (p.get<std::string>                   ("DefGrad Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  this->addDependentField(defgrad);

  this->addEvaluatedField(lcg);

  this->setName("LCG"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LCG<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(lcg,fm);
  this->utils.setFieldData(defgrad,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LCG<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  // Compute lcg tensor from deformation gradient
  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i < numDims; ++i) {
        for (std::size_t j=0; j < numDims; ++j) {
	  lcg(cell,qp,i,j) = 0.0;
	  for (std::size_t k=0; k < numDims; ++k) {
	    lcg(cell,qp,i,j) += defgrad(cell,qp,i,k) * defgrad(cell,qp,j,k);
	  }
        }
      }
    }
  }

}

//**********************************************************************
}

