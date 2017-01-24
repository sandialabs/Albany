//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
DOFDivInterpolationSideBase(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl_side->node_vector),
  gradBF      (p.get<std::string> ("Gradient BF Name"), dl_side->node_qp_gradient),
  val_qp      (p.get<std::string> ("Divergence Variable Name"), dl_side->qp_scalar )
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(gradBF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFDivInterpolationSideBase" );

  numSideNodes = dl_side->node_qp_gradient->dimension(2);
  numSideQPs   = dl_side->node_qp_gradient->dimension(3);
  numDims      = dl_side->node_qp_gradient->dimension(4);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int qp=0; qp<numSideQPs; ++qp)
    {
      val_qp(cell,side,qp) = 0.;
      for (int dim=0; dim<numDims; ++dim)
      {
        for (int node=0; node<numSideNodes; ++node)
        {
          val_qp(cell,side,qp) += val_node(cell,side,node,dim) * gradBF(cell,side,node,qp,dim);
        }
      }
    }
  }
}

} // Namespace PHAL
