//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include "PHAL_DOFGradInterpolationSide.hpp"
#include "Albany_DiscretizationUtils.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFGradInterpolationSideBase<EvalT, Traits, ScalarT>::
DOFGradInterpolationSideBase(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl_side->node_scalar),
  gradBF      (p.get<std::string> ("Gradient BF Name"), dl_side->node_qp_gradient),
  grad_qp      (p.get<std::string> ("Gradient Variable Name"), dl_side->qp_gradient)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl_side->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layouts structure does not appear to be that of a side set.\n");

  this->addNonConstDependentField(val_node.fieldTag());
  this->addNonConstDependentField(gradBF.fieldTag());
  this->addEvaluatedField(grad_qp);

  this->setName("DOFGradInterpolationSide("+p.get<std::string>("Variable Name") + ")"+PHX::print<EvalT>());

  numSideNodes = dl_side->node_qp_gradient->extent(1);
  numSideQPs   = dl_side->node_qp_gradient->extent(2);
  numDims      = dl_side->node_qp_gradient->extent(3);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFGradInterpolationSideBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(grad_qp,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolationSideBase<EvalT, Traits, ScalarT>::
operator() (const GradInterpolationSide_Tag&, const int& sideSet_idx) const {
  
  for (int qp=0; qp<numSideQPs; ++qp) {
    for (int dim=0; dim<numDims; ++dim) {
      grad_qp(sideSet_idx,qp,dim) = 0.;
      for (int node=0; node<numSideNodes; ++node) {
        grad_qp(sideSet_idx,qp,dim) += val_node(sideSet_idx,node) * gradBF(sideSet_idx,node,qp,dim);
      }
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFGradInterpolationSideBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);
  
  Kokkos::parallel_for(GradInterpolationSide_Policy(0, sideSet.size), *this);    
}

} // Namespace PHAL
