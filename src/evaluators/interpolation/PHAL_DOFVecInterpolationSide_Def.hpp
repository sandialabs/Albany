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
  val_qp      (p.get<std::string> ("Variable Name"), dl_side->qp_vector)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl_side->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layouts structure does not appear to be that of a side set.\n");

  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolationSide"+PHX::print<EvalT>());

  numSideNodes = dl_side->node_qp_scalar->extent(1);
  numSideQPs   = dl_side->node_qp_scalar->extent(2);
  vecDim       = dl_side->qp_vector->extent(2);
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

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename Type>
KOKKOS_INLINE_FUNCTION
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
operator() (const VecInterpolationSide_Tag&, const int& sideSet_idx) const {
  
  for (int dim=0; dim<vecDim; ++dim) {
    for (int qp=0; qp<numSideQPs; ++qp) {
      val_qp(sideSet_idx,qp,dim) = val_node(sideSet_idx,0,dim) * BF(sideSet_idx,0,qp);
      for (int node=1; node<numSideNodes; ++node) {
        val_qp(sideSet_idx,qp,dim) += val_node(sideSet_idx,node,dim) * BF(sideSet_idx,node,qp);
      }
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);

  Kokkos::parallel_for(VecInterpolationSide_Policy(0, sideSet.size), *this);
}

} // Namespace PHAL
