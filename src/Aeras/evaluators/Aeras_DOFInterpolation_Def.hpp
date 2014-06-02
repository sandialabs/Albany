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
DOFInterpolation<EvalT, Traits>::
DOFInterpolation(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), 
               p.get<Teuchos::RCP<PHX::DataLayout> >("Nodal Variable Layout",     dl->node_scalar_level)),
  BF          (p.get<std::string>   ("BF Name"),                                  dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), 
               p.get<Teuchos::RCP<PHX::DataLayout> >("Quadpoint Variable Layout", dl->qp_scalar_level)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  numRank    (val_node.fieldTag().dataLayout().rank())
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("Aeras::DOFInterpolation"+PHX::TypeString<EvalT>::value);

  TEUCHOS_TEST_FOR_EXCEPTION( (numRank!=2 && numRank!=3),
     std::logic_error,"Aeras::DOFGradInterpolation supports scalar or vector only");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      if (2==numRank) {
        ScalarT& vqp = val_qp(cell,qp) = 0;
        for (int node=0; node < numNodes; ++node) {
          vqp += val_node(cell, node) * BF(cell, node, qp);
        }
      } else {
        for (int level=0; level < numLevels; ++level) {
          ScalarT& vqp = val_qp(cell,qp,level) = 0;
          for (int node=0; node < numNodes; ++node) {
            vqp += val_node(cell, node, level) * BF(cell, node, qp);
          }
        }
      } 
    }
  }
}
}

