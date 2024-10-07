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
DOFInterpolationSideBase<EvalT, Traits, ScalarT>::
DOFInterpolationSideBase (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl_side->node_scalar),
  BF          (p.get<std::string> ("BF Name"), dl_side->node_qp_scalar),
  val_qp      (p.get<std::string> ("Variable Name"), dl_side->qp_scalar)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl_side->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layouts structure does not appear to be that of a side set.\n");

  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationSide"+PHX::print<EvalT>());

  numSideNodes = dl_side->node_qp_scalar->extent(1);
  numSideQPs   = dl_side->node_qp_scalar->extent(2);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFInterpolationSideBase<EvalT, Traits, ScalarT>::
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
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void DOFInterpolationSideBase<EvalT, Traits, ScalarT>::
operator() (const InterpolationSide_Tag&, const int& sideSet_idx) const {

  for (int qp=0; qp<numSideQPs; ++qp) {
    val_qp(sideSet_idx,qp) = 0;
    for (int node=0; node<numSideNodes; ++node) {
      val_qp(sideSet_idx,qp) += val_node(sideSet_idx,node) * BF(sideSet_idx,node,qp);
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFInterpolationSideBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);
  
  Kokkos::parallel_for(InterpolationSide_Policy(0, sideSet.size), *this);
}

} // Namespace PHAL
