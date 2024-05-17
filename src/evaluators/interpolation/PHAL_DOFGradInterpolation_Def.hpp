//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(GradBF.fieldTag());
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase"+PHX::print<EvalT>());

 // std::vector<PHX::DataLayout::size_type> dims;
  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor Residual
#ifdef KOKKOS_OPTIMIZED
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
operator()( const team_member & thread) const{

  const int thread_idx = thread.league_rank() * threads_per_team;
  const int end_loop= thread_idx+threads_per_team>(numCells*numQPs)?(numCells*numQPs):(thread_idx+threads_per_team);
  ScalarT gradVal_tmp;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, thread_idx, end_loop), [=](int& indx){
      const int cell =indx/numCells;
      const int qp =indx=indx/numCells;
      const int vector_range=numNodes-1;
          for (int dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);

           /* Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, vector_range),
                               [&](const int& lk, ScalarT& gradVal){
                           const int node=1+lk;
                           gradVal += val_node(cell, node) * GradBF(cell, node, qp, dim);
                }, gradVal_tmp);
            Kokkos::single(Kokkos::PerThread(thread),[&](){
               grad_val_qp(cell,qp,dim)=val_node(cell, 0) * GradBF(cell, 0, qp, dim)+gradVal_tmp;
           });
           */
            for (int node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      });


}

#else
  template<typename EvalT, typename Traits, typename ScalarT>
  KOKKOS_INLINE_FUNCTION
  void DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
  operator() (const DOFGradInterpolationBase_Residual_Tag& tag, const int& cell) const {

   for (size_t qp=0; qp < numQPs; ++qp) {
          for (size_t dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
            for (size_t node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      }
}
#endif
// ***************************************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

//  Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);

#ifdef KOKKOS_OPTIMIZED

  threads_per_team=work_size;
  numTeams=(workset.numCells*numQPs+threads_per_team-1)/threads_per_team;
  numCells=workset.numCells;

  const team_policy policy(numTeams, 1, 16);

   Kokkos::parallel_for(policy, *this);

#else
 Kokkos::parallel_for(this->getName(),DOFGradInterpolationBase_Residual_Policy(0,workset.numCells),*this);
#endif
}
//Specialization for Jacobian evaluation taking advantage of the sparsity of the derivatives
//Note, this assumes that the mesh is not FAD type
// *********************************************************************

// Kokkos kernel for Jacobian
template< typename Traits>
KOKKOS_INLINE_FUNCTION
void FastSolutionGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
operator() (const FastSolutionGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const {
  const int num_dof = this->val_node(0,0).size();
  for (size_t qp=0; qp < this->numQPs; ++qp) {
    for (size_t dim=0; dim<this->numDims; dim++) {
      this->grad_val_qp(cell,qp,dim) = ScalarT(num_dof, this->val_node(cell, 0).val() * this->GradBF(cell, 0, qp, dim));
      (this->grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = this->val_node(cell, 0).fastAccessDx(offset) * this->GradBF(cell, 0, qp, dim);
      for (size_t node= 1 ; node < this->numNodes; ++node) {
        (this->grad_val_qp(cell,qp,dim)).val() += this->val_node(cell, node).val() * this->GradBF(cell, node, qp, dim);
        (this->grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += this->val_node(cell, node).fastAccessDx(neq*node+offset) * this->GradBF(cell, node, qp, dim);
      }
    }
  }
}
//**********************************************************************

template<typename Traits>
void FastSolutionGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{

  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

 neq = workset.disc->getDOFManager()->getNumFields();

 Kokkos::parallel_for(this->getName(),FastSolutionGradInterpolationBase_Jacobian_Policy(0,workset.numCells),*this);
}

//**********************************************************************

} // Namespace PHAL
