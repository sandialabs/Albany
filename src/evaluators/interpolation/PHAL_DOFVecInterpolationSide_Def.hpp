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
template<typename EvalT, typename Traits, typename Type>
DOFVecInterpolationSideBase<EvalT, Traits, Type>::
DOFVecInterpolationSideBase(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl_side->node_vector),
  BF          (p.get<std::string> ("BF Name"), dl_side->node_qp_scalar),
  val_qp      (p.get<std::string> ("Variable Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_vector_sideset : dl_side->qp_vector)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl_side->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layouts structure does not appear to be that of a side set.\n");

  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolationSide"+PHX::print<EvalT>());

  numSideNodes = dl_side->node_qp_scalar->extent(2);
  numSideQPs   = dl_side->node_qp_scalar->extent(3);
  vecDim       = dl_side->qp_vector->extent(3);

  useCollapsedSidesets = dl_side->useCollapsedSidesets;
}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
evaluateFields(typename Traits::EvalData workset)
{
  // if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
  //   return;
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  // const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    // Get the local data of side and cell
    // const int cell = it_side.elem_LID;
    // const int side = it_side.side_local_id;
    const int cell = sideSet.elem_LID(sideSet_idx);
    const int side = sideSet.side_local_id(sideSet_idx);

    for (int dim=0; dim<vecDim; ++dim)
    {
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        if (useCollapsedSidesets) {
          val_qp(sideSet_idx,qp,dim) = val_node(cell,side,0,dim) * BF(cell,side,0,qp);
        }
        else {
          val_qp(cell,side,qp,dim) = val_node(cell,side,0,dim) * BF(cell,side,0,qp);
        }
        
        for (int node=1; node<numSideNodes; ++node)
        {
          if (useCollapsedSidesets) {
            val_qp(sideSet_idx,qp,dim) += val_node(cell,side,node,dim) * BF(cell,side,node,qp);
          }
          else {
            val_qp(cell,side,qp,dim) += val_node(cell,side,node,dim) * BF(cell,side,node,qp);
          }
          
        }
      }
    }
  }
}

} // Namespace PHAL
