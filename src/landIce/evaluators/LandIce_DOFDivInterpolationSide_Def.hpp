//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
DOFDivInterpolationSideBase(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl_side->node_vector),
  gradBF      (p.get<std::string> ("Gradient BF Name"), dl_side->node_qp_gradient),
  tangents    (p.get<std::string> ("Tangents Name"), dl_side->qp_tensor_cd_sd),
  val_qp      (p.get<std::string> ("Divergence Variable Name"), dl_side->qp_scalar)
{
  this->addDependentField(val_node);
  this->addDependentField(gradBF);
  this->addDependentField(tangents);
  this->addEvaluatedField(val_qp);

  this->setName("DOFDivInterpolationSideBase" );

  numSideNodes = dl_side->node_qp_gradient->extent(1);
  numSideQPs   = dl_side->node_qp_gradient->extent(2);
  numDims      = dl_side->node_qp_vector->extent(3);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(tangents,fm);
  this->utils.setFieldData(val_qp,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
operator() (const DivInterpolation_Tag&, const int& sideSet_idx) const {
  
  for (unsigned int qp=0; qp<numSideQPs; ++qp)
  {
    val_qp(sideSet_idx,qp) = 0.;
    for (unsigned int dim=0; dim<numDims; ++dim)
    {
      for (unsigned int node=0; node<numSideNodes; ++node)
      {
        MeshScalarT gradBF_non_intrinsic = 0.0;
        for (unsigned int itan=0; itan<numDims; ++itan)
        {
          gradBF_non_intrinsic += tangents(sideSet_idx,qp,dim,itan)*gradBF(sideSet_idx,node,qp,itan);
        }
        val_qp(sideSet_idx,qp) += val_node(sideSet_idx,node,dim) * gradBF_non_intrinsic;
      }
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFDivInterpolationSideBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);

  Kokkos::parallel_for(DivInterpolation_Policy(0, sideSet.size), *this);  
}

} // Namespace LandIce
