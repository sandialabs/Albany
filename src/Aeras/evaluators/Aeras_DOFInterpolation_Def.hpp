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
               p.get<Teuchos::RCP<PHX::DataLayout> >("Nodal Variable Layout", dl->node_scalar_level)),
  BF          (p.get<std::string>   ("BF Name"),       dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), 
               p.get<Teuchos::RCP<PHX::DataLayout> >("Quadpoint Variable Layout", dl->qp_scalar_level))
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation"+PHX::TypeString<EvalT>::value);

  

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  val_node.fieldTag().dataLayout().dimensions(dims);
  const int rank = val_node.fieldTag().dataLayout().rank();
  numLevels = dims[2];
  numTracers=  3<rank ? dims[3] : 0;
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
      for (int level=0; level < numLevels; ++level) {
        if (numTracers) {
          for (int t=0; t < numTracers; ++t) {
            ScalarT& vqp = val_qp(cell,qp,level,t);
            vqp = val_node(cell, 0, level, t) * BF(cell, 0, qp);
            for (int node=1; node < numNodes; ++node) {
              vqp += val_node(cell, node, level, t) * BF(cell, node, qp);
            }
          }
        } else {
          ScalarT& vqp = val_qp(cell,qp,level);
          vqp = val_node(cell, 0, level) * BF(cell, 0, qp);
          for (int node=1; node < numNodes; ++node) {
            vqp += val_node(cell, node, level) * BF(cell, node, qp);
          }
        }
      }
    }
  }
}
}

