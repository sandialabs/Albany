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
RCG<EvalT, Traits>::
RCG(const Teuchos::ParameterList& p) :
  defgrad   (p.get<std::string>                   ("DefGrad Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  rcg       (p.get<std::string>                   ("RCG Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  this->addDependentField(defgrad);

  this->addEvaluatedField(rcg);

  this->setName("RCG"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void RCG<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rcg,fm);
  this->utils.setFieldData(defgrad,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void RCG<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t numCells = workset.numCells;

  // Compute rcg tensor from deformation gradient
  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i < numDims; ++i) {
        for (std::size_t j=0; j < numDims; ++j) {
	  rcg(cell,qp,i,j) = 0.0;
	  for (std::size_t k=0; k < numDims; ++k) {
	    rcg(cell,qp,i,j) += defgrad(cell,qp,k,i) * defgrad(cell,qp,k,j);
	  }
        }
      }
    }
  }

}

//**********************************************************************
}

