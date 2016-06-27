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
template<typename EvalT, typename Traits, typename Type>
DOFGradInterpolation<EvalT, Traits, Type>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation" );

 // std::vector<PHX::DataLayout::size_type> dims;
  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFGradInterpolation<EvalT, Traits, Type>::
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
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolation<EvalT, Traits>::
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
  template<typename EvalT, typename Traits>
  KOKKOS_INLINE_FUNCTION
  void DOFGradInterpolation<EvalT, Traits>::
  operator() (const DOFGradInterpolation_Residual_Tag& tag, const int& cell) const {

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
template<typename EvalT, typename Traits, typename Type>
void DOFGradInterpolation<EvalT, Traits, Type>::
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
//  Kokkos::deep_copy(grad_val_qp.get_view(), 0.0);

#ifdef KOKKOS_OPTIMIZED

  threads_per_team=work_size;
  numTeams=(workset.numCells*numQPs+threads_per_team-1)/threads_per_team;
  numCells=workset.numCells;

  const team_policy policy(numTeams, 1, 16);

   Kokkos::parallel_for(policy, *this);

#else
 Kokkos::parallel_for(DOFGradInterpolation_Residual_Policy(0,workset.numCells),*this);
#endif

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFGradInterpolation Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif

}

//**********************************************************************
template<typename Traits>
DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation Jacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
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
  void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
  operator() (const DOFGradInterpolation_Jacobian_Tag& tag, const int& cell) const {

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
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
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

 Kokkos::parallel_for(DOFGradInterpolation_Jacobian_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFGradInterpolation Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif


}

//**********************************************************************

template<typename EvalT, typename Traits>
DOFGradInterpolation_noDeriv<EvalT, Traits>::
DOFGradInterpolation_noDeriv(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation_noDeriv" );

  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}
//**********************************************************************
//Kokkos functor GradInt noDeriv
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
operator()(const int& i) const
 {
   for (int qp=0; qp < numQPs; ++qp) {
       for (int dim=0; dim<numDims; dim++) {
           grad_val_qp(i,qp,dim) = val_node(i, 0) * GradBF(i, 0, qp, dim);
            for (int node= 1 ; node < numNodes; ++node) {
              grad_val_qp(i,qp,dim) += val_node(i, node) * GradBF(i, node, qp, dim);
          }
        }
      }
 }
//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

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
  auto start = std::chrono::high_resolution_clock::now();
#endif

  Kokkos::parallel_for(workset.numCells,*this);

#ifdef ALBANY_TIMER
  PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFGradInterpolation Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
}

#ifdef ALBANY_SG
//**********************************************************************
template<typename Traits>
DOFGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation SGJacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
DOFGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation MPJacobian");

  std::vector<PHX::Device::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
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

}

