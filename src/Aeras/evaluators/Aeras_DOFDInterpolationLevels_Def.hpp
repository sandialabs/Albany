//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFDInterpolationLevels<EvalT, Traits>::
DOFDInterpolationLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node   (p.get<std::string>   ("Variable Name"),            dl->node_vector_level),
  GradBF     (p.get<std::string>   ("Gradient BF Name"),         dl->node_qp_gradient),
  d_val_qp   (p.get<std::string>   ("Derivative Variable Name"), dl->qp_vector_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(d_val_qp);

  this->setName("Aeras::DOFDInterpolationLevels"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFDInterpolationLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(d_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFDInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < d_val_qp.size(); ++i) d_val_qp(i)=0.0;
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int qp=0; qp < numQPs; ++qp) 
      for (int node= 0 ; node < numNodes; ++node) 
        for (int level=0; level < numLevels; ++level) 
          for (int dim=0; dim<numDims; dim++) {
            d_val_qp(cell,qp,level,dim) += val_node(cell,node,level,dim) * GradBF(cell,node,qp,dim);
}

}

}

