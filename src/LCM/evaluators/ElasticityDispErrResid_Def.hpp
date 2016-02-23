//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ElasticityDispErrResid<EvalT, Traits>::
ElasticityDispErrResid(const Teuchos::ParameterList& p) :
  ErrorStress  (p.get<std::string>                   ("Error Stress Name"),
	              p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  wGradBF      (p.get<std::string>                   ("Weighted Gradient BF Name"),
	              p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
  DispResid    (p.get<std::string>                   ("Displacement Residual Name"),
                p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout") ),
  ExResidual   (p.get<std::string>                   ("Residual Name"),
	              p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout") )
{
  this->addDependentField(ErrorStress);
  this->addDependentField(wGradBF);
  this->addDependentField(DispResid);

  this->addEvaluatedField(ExResidual);

  this->setName("ElasticityDispErrResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ElasticityDispErrResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(ErrorStress,fm);
  this->utils.setFieldData(DispResid,fm);
  this->utils.setFieldData(wGradBF,fm);
  
  this->utils.setFieldData(ExResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ElasticityDispErrResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int dim=0; dim<numDims; dim++)  ExResidual(cell,node,dim)=0.0;
      for (int qp=0; qp < numQPs; ++qp) {
        for (int i=0; i<numDims; i++) {
          ExResidual(cell,node,i) += DispResid(cell,node,i);
          for (int dim=0; dim<numDims; dim++) {
            ExResidual(cell,node,i) += ErrorStress(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
    } } } } }
}

//**********************************************************************
}

