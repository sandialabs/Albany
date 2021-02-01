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
  val_node    (p.get<std::string> ("Variable Name"), (dl_side->useCollapsedSidesets) ? dl_side->node_vector_sideset : dl_side->node_vector),
  gradBF      (p.get<std::string> ("Gradient BF Name"), (dl_side->useCollapsedSidesets) ? dl_side->node_qp_gradient_sideset : dl_side->node_qp_gradient),
  tangents    (p.get<std::string> ("Tangents Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_tensor_cd_sd_sideset : dl_side->qp_tensor_cd_sd),
  val_qp      (p.get<std::string> ("Divergence Variable Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_scalar_sideset : dl_side->qp_scalar )
{
  useCollapsedSidesets = dl_side->useCollapsedSidesets;

  this->addDependentField(val_node);
  this->addDependentField(gradBF);
  this->addDependentField(tangents);
  this->addEvaluatedField(val_qp);

  this->setName("DOFDivInterpolationSideBase" );

  numSideNodes = dl_side->node_qp_gradient->extent(2);
  numSideQPs   = dl_side->node_qp_gradient->extent(3);
  numDims      = dl_side->node_qp_vector->extent(4);
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
operator() (const DivInterpolation_Tag& tag, const int& sideSet_idx) const {
  
  for (int qp=0; qp<numSideQPs; ++qp)
  {
    val_qp(sideSet_idx,qp) = 0.;
    for (int dim=0; dim<numDims; ++dim)
    {
      for (int node=0; node<numSideNodes; ++node)
      {
        MeshScalarT gradBF_non_intrinsic = 0.0;
        for (int itan=0; itan<numDims; ++itan)
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
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return;

  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);
  if (useCollapsedSidesets) {
    Kokkos::parallel_for(DivInterpolation_Policy(0, sideSet.size), *this);
  } else {
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.elem_LID(sideSet_idx);
      const int side = sideSet.side_local_id(sideSet_idx);

      for (int qp=0; qp<numSideQPs; ++qp)
      {
        val_qp(cell,side,qp) = 0.;
        for (int dim=0; dim<numDims; ++dim)
        {
          for (int node=0; node<numSideNodes; ++node)
          {
            MeshScalarT gradBF_non_intrinsic = 0.0;
            for (int itan=0; itan<numDims; ++itan)
            {
              gradBF_non_intrinsic += tangents(cell,side,qp,dim,itan)*gradBF(cell,side,node,qp,itan);
            }
            val_qp(cell,side,qp) += val_node(cell,side,node,dim) * gradBF_non_intrinsic;
          }
        }
      }
    }
  }
  
}

} // Namespace LandIce
