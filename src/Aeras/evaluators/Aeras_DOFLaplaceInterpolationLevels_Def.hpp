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
DOFLaplaceInterpolationLevels<EvalT, Traits>::
DOFLaplaceInterpolationLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"),          dl->node_vector_level),
  GradGradBF  (p.get<std::string>   ("Gradient Gradient BF Name"), dl->node_qp_tensor),
  Laplace_val_qp (p.get<std::string>   ("Laplace Variable Name"), dl->qp_scalar_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradGradBF);
  this->addEvaluatedField(Laplace_val_qp);

  this->setName("Aeras::DOFLaplaceInterpolationLevels"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFLaplaceInterpolationLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradGradBF,fm);
  this->utils.setFieldData(Laplace_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFLaplaceInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < Laplace_val_qp.size(); ++i) Laplace_val_qp(i)=0.0;
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int qp=0; qp < numQPs; ++qp) 
      for (int level=0; level < numLevels; ++level) 
        for (int node=0 ; node < numNodes; ++node) 
          for (int dim=0; dim<numDims; dim++) 
            Laplace_val_qp(cell,qp,level) += val_node(cell, node, level, dim) * GradGradBF(cell, node, qp, dim, dim);
}
}

