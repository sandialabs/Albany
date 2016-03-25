//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL
{

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::
DOFVecGradInterpolationSideBase(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl->side_node_vector),
  gradBF      (p.get<std::string> ("Gradient BF Name"), dl->side_node_qp_gradient),
  grad_qp     (p.get<std::string> ("Gradient Variable Name"), dl->side_qp_vecgradient )
{
  this->addDependentField(val_node);
  this->addDependentField(gradBF);
  this->addEvaluatedField(grad_qp);

  this->setName("DOFVecGradInterpolationSideBase" );

  numSideNodes = dl->side_node_qp_gradient->dimension(2);
  numSideQPs   = dl->side_node_qp_gradient->dimension(3);
  numDims      = dl->side_node_qp_gradient->dimension(4);
  vecDim       = dl->side_node_vector->dimension(3);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(grad_qp,fm);
}

// *********************************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFVecGradInterpolationSideBase<EvalT, Traits, ScalarT>::
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
      for (int comp=0; comp<vecDim; ++comp)
      {
        for (int dim=0; dim<numDims; ++dim)
        {
          grad_qp(cell,side,qp,comp,dim) = 0.;
          for (int node=0; node<numSideNodes; ++node)
          {
            grad_qp(cell,side,qp,comp,dim) += val_node(cell,side,node,comp) * gradBF(cell,side,node,qp,dim);
          }
        }
      }
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
DOFVecGradInterpolationSide<EvalT, Traits>::
DOFVecGradInterpolationSide (const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl) :
  DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>(p,dl)
{
  // Nothing to be done here
}

//**********************************************************************
template<typename EvalT, typename Traits>
DOFVecGradInterpolationSide_noDeriv<EvalT, Traits>::
DOFVecGradInterpolationSide_noDeriv (const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl) :
  DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>(p,dl)
{
  // Nothing to be done here
}

} // Namespace PHAL
