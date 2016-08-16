//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifdef ALBANY_TIMER
#include <chrono>
#endif

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

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
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase" );

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
}

// *********************************************************************
// Kokkos functor Reesidual
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
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

   for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
            for (int node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      }
}
#endif
#endif
// ***************************************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFGradInterpolationBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
#else

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto start = std::chrono::high_resolution_clock::now();
#endif
//  Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);

#ifdef KOKKOS_OPTIMIZED

  threads_per_team=work_size;
  numTeams=(workset.numCells*numQPs+threads_per_team-1)/threads_per_team;
  numCells=workset.numCells;

  const team_policy policy(numTeams, 1, 16);

   Kokkos::parallel_for(policy, *this);

#else
 Kokkos::parallel_for(DOFGradInterpolationBase_Residual_Policy(0,workset.numCells),*this);
#endif

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFGradInterpolationBase Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif

}

//**********************************************************************
template<typename Traits>
DOFGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase Jacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}
// *********************************************************************
// Kokkos kernel for Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  template< typename Traits>
  KOKKOS_INLINE_FUNCTION
  void DOFGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  operator() (const DOFGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const {

    for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = ScalarT(num_dof, val_node(cell, 0).val() * GradBF(cell, 0, qp, dim));
            (grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * GradBF(cell, 0, qp, dim);
            for (int node= 1 ; node < numNodes; ++node) {
              (grad_val_qp(cell,qp,dim)).val() += val_node(cell, node).val() * GradBF(cell, node, qp, dim);
              (grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * GradBF(cell, node, qp, dim);
          }
        }
      }
}
#endif
//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
#else
            grad_val_qp(cell,qp,dim) = ScalarT(num_dof, val_node(cell, 0).val() * GradBF(cell, 0, qp, dim));
            (grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * GradBF(cell, 0, qp, dim);
#endif
            for (std::size_t node= 1 ; node < numNodes; ++node) {
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
#else
              (grad_val_qp(cell,qp,dim)).val() += val_node(cell, node).val() * GradBF(cell, node, qp, dim);
              (grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * GradBF(cell, node, qp, dim);
#endif
          }
        }
      }
    }
#else

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto start = std::chrono::high_resolution_clock::now();
#endif

 num_dof = val_node(0,0).size();
 neq = workset.wsElNodeEqID[0][0].size();

 Kokkos::parallel_for(DOFGradInterpolationBase_Jacobian_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFGradInterpolationBase Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif


}

#ifdef ALBANY_SG
//**********************************************************************
template<typename Traits>
DOFGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase SGJacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);


  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = ScalarT(num_dof, val_node(cell, 0).val() * GradBF(cell, 0, qp, dim));
            (grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              (grad_val_qp(cell,qp,dim)).val() += val_node(cell, node).val() * GradBF(cell, node, qp, dim);
              (grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
}
#endif

#ifdef ALBANY_ENSEMBLE
//**********************************************************************
template<typename Traits>
DOFGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
DOFGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolationBase MPJacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);


  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            grad_val_qp(cell,qp,dim) = ScalarT(num_dof, val_node(cell, 0).val() * GradBF(cell, 0, qp, dim));
            (grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              (grad_val_qp(cell,qp,dim)).val() += val_node(cell, node).val() * GradBF(cell, node, qp, dim);
              (grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
}
#endif

//**********************************************************************

} // Namespace PHAL
