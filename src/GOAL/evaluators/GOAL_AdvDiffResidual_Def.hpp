//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace GOAL {

//**********************************************************************
template<typename EvalT, typename Traits>
AdvDiffResidual<EvalT, Traits>::
AdvDiffResidual(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  u        (p.get<std::string>("U Name"), dl->qp_scalar),
  gradU    (p.get<std::string>("Gradient U Name"), dl->qp_gradient),
  wBF      (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_gradient),
  residual (p.get<std::string>("Residual Name"), dl->node_scalar)
{
  this->addDependentField(u);
  this->addDependentField(gradU);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(residual);

  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];
  
  this->setName("AdvDiffResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResidual<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u, fm);
  this->utils.setFieldData(gradU, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(residual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

}
