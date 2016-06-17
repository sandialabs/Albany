//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesMomentumResid<EvalT, Traits>::
StokesMomentumResid(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF       (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar), 
  wGradBF   (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_vector),
  VGrad     (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_tensor),
  P         (p.get<std::string> ("Pressure QP Variable Name"), dl->qp_scalar),
  force     (p.get<std::string> ("Body Force Name"), dl->qp_vector),
  muFELIX   (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar),
  MResidual (p.get<std::string> ("Residual Name"),dl->node_vector)
{

  this->addDependentField(wBF);  
  this->addDependentField(VGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(P);
  this->addDependentField(force);
  this->addDependentField(muFELIX);
  
  this->addEvaluatedField(MResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];
  
  this->setName("StokesMomentumResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesMomentumResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(wGradBF,fm); 
  this->utils.setFieldData(P,fm);
  this->utils.setFieldData(force,fm);
  this->utils.setFieldData(muFELIX,fm);
  
  this->utils.setFieldData(MResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesMomentumResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {          
      for (int i=0; i<numDims; i++) {
	MResidual(cell,node,i) = 0.0;
	for (int qp=0; qp < numQPs; ++qp) {
	  MResidual(cell,node,i) += 
	    force(cell,qp,i)*wBF(cell,node,qp) -
	     P(cell,qp)*wGradBF(cell,node,qp,i);          
	  for (int j=0; j < numDims; ++j) { 
	    MResidual(cell,node,i) += 
	      muFELIX(cell,qp)*(VGrad(cell,qp,i,j)+VGrad(cell,qp,j,i))*wGradBF(cell,node,qp,j);
//	      muFELIX(cell,qp)*VGrad(cell,qp,i,j)*wGradBF(cell,node,qp,j);
	  }  
	}
      }
    }
  }
  
 
}

}

