//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#ifdef ALBANY_TIMER
#include <chrono>
#endif

namespace PHAL {

  //**********************************************************************
  template<typename EvalT, typename Traits, typename ScalarT>
  DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node.fieldTag());
    this->addDependentField(GradBF.fieldTag());
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolationBase"+PHX::print<EvalT>());

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];
  }

  //**********************************************************************
  template<typename EvalT, typename Traits, typename ScalarT>
  void DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
    if (d.memoizer_active()) memoizer.enable_memoizer();
  }

  //*********************************************************************
  //KOKKOS functor Residual

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  template<typename EvalT, typename Traits, typename ScalarT>
  KOKKOS_INLINE_FUNCTION
  void DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  operator() (const DOFVecGradInterpolationBase_Residual_Tag& tag, const int& cell) const
  {
    for (size_t qp=0; qp < numQPs; ++qp)
      for (size_t i=0; i<vecDim; i++)
        for (size_t dim=0; dim<numDims; dim++)
             grad_val_qp(cell,qp,i,dim)=0.0;

    for (size_t qp=0; qp < numQPs; ++qp) {
      for (size_t i=0; i<vecDim; i++) {
        for (size_t dim=0; dim<numDims; dim++) {
          // For node==0, overwrite. Then += for 1 to numNodes.
          grad_val_qp(cell,qp,i,dim) = val_node(cell, 0, i) * GradBF(cell, 0, qp, dim);
          for (size_t node= 1 ; node < numNodes; ++node) {
            grad_val_qp(cell,qp,i,dim) += val_node(cell, node, i) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }
#endif

// *********************************************************************************
  template<typename EvalT, typename Traits, typename ScalarT>
  void DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t i=0; i<vecDim; i++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              grad_val_qp(cell,qp,i,dim) = val_node(cell, 0, i) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                grad_val_qp(cell,qp,i,dim) += val_node(cell, node, i) * GradBF(cell, node, qp, dim);
            }
          }
        }
      }
    }

    //  Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);
#else

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto start = std::chrono::high_resolution_clock::now();
#endif
  //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);
  Kokkos::parallel_for(DOFVecGradInterpolationBase_Residual_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFVecGradInterpolationBase Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
  }

  // Specialization for Jacobian evaluation taking advantage of known sparsity
  // This assumes that mesh coordinates are not FAD types
  //**********************************************************************
  //Kokkos functor Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  template<typename Traits>
  KOKKOS_INLINE_FUNCTION
  void FastSolutionVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  operator() (const FastSolutionVecGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const {
    const int num_dof = this->val_node(0,0,0).size();
    for (size_t qp=0; qp < this->numQPs; ++qp) {
          for (size_t i=0; i<this->vecDim; i++) {
            for (size_t dim=0; dim<this->numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              this->grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, this->val_node(cell, 0, i).val() * this->GradBF(cell, 0, qp, dim));
              (this->grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = this->val_node(cell, 0, i).fastAccessDx(offset+i) * this->GradBF(cell, 0, qp, dim);
              for (size_t node= 1 ; node < this->numNodes; ++node) {
                (this->grad_val_qp(cell,qp,i,dim)).val() += this->val_node(cell, node, i).val() * this->GradBF(cell, node, qp, dim);
                (this->grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += this->val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * this->GradBF(cell, node, qp, dim);
           }
         }
        }
      }

  }
#endif
  //**********************************************************************
  template<typename Traits>
  void FastSolutionVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    const int num_dof = this->val_node(0,0,0).size();
    const int neq = workset.disc->getDOFManager()->getNumFields();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < this->numQPs; ++qp) {
          for (std::size_t i=0; i<this->vecDim; i++) {
            for (std::size_t dim=0; dim<this->numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              this->grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, this->val_node(cell, 0, i).val() * this->GradBF(cell, 0, qp, dim));
              (this->grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = this->val_node(cell, 0, i).fastAccessDx(offset+i) * this->GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < this->numNodes; ++node) {
                (this->grad_val_qp(cell,qp,i,dim)).val() += this->val_node(cell, node, i).val() * this->GradBF(cell, node, qp, dim);
                (this->grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += this->val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * this->GradBF(cell, node, qp, dim);
           }
         }
        }
      }
    }
 //  Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

   neq = workset.disc->getDOFManager()->getNumFields();

   Kokkos::parallel_for(FastSolutionVecGradInterpolationBase_Jacobian_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
  PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "FastSolutionVecGradInterpolationBase Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
  }

} // Namespace PHAL
