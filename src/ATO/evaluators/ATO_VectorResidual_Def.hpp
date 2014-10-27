//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
VectorResidual<EvalT, Traits>::
VectorResidual(const Teuchos::ParameterList& p) :
  vector      (p.get<std::string>                   ("Vector Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  ExResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(vector);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(ExResidual);

  this->setName("VectorResidual"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void VectorResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(vector,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(ExResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void VectorResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
        ExResidual(cell,node)=0.0;
        for (std::size_t qp=0; qp < numQPs; ++qp)
          for (std::size_t dim=0; dim<numDims; dim++)
            ExResidual(cell,node) += vector(cell, qp, dim) * wGradBF(cell, node, qp, dim);
    } 
  }
}

//**********************************************************************
}

