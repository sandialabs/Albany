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
  DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolationBase" );

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
  }

  //*********************************************************************
  //KOKKOS functor Residual

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  template<typename EvalT, typename Traits, typename ScalarT>
  KOKKOS_INLINE_FUNCTION
  void DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>::
  operator() (const DOFVecGradInterpolationBase_Residual_Tag& tag, const int& cell) const {


  for (int qp=0; qp < numQPs; ++qp)
    for (int i=0; i<vecDim; i++)
      for (int dim=0; dim<numDims; dim++)
           grad_val_qp(cell,qp,i,dim)=0.0;

   for (int qp=0; qp < numQPs; ++qp) {
          for (int i=0; i<vecDim; i++) {
            for (int dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              grad_val_qp(cell,qp,i,dim) = val_node(cell, 0, i) * GradBF(cell, 0, qp, dim);
              for (int node= 1 ; node < numNodes; ++node) {
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
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    // This is needed, since evaluate currently sums into
   Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);

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

  // Specializations for all 3 Jacobian types are identical
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolationBase Jacobian");

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];

    offset = p.get<int>("Offset of First DOF");
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }
  //**********************************************************************
  //Kokkos functor Jacabian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  template<typename Traits>
  KOKKOS_INLINE_FUNCTION
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  operator() (const DOFVecGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const {
    for (int qp=0; qp < numQPs; ++qp) {
          for (int i=0; i<vecDim; i++) {
            for (int dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, val_node(cell, 0, i).val() * GradBF(cell, 0, qp, dim));
              (grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * GradBF(cell, 0, qp, dim);
              for (int node= 1 ; node < numNodes; ++node) {
                (grad_val_qp(cell,qp,i,dim)).val() += val_node(cell, node, i).val() * GradBF(cell, node, qp, dim);
                (grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * GradBF(cell, node, qp, dim);
           }
         }
        }
      }

  }
#endif
  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    const int num_dof = val_node(0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t i=0; i<vecDim; i++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
              grad_val_qp(cell,qp,i,dim) = val_node(cell, 0, i)* GradBF(cell, 0, qp, dim);
#else
              grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, val_node(cell, 0, i).val() * GradBF(cell, 0, qp, dim));
              (grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * GradBF(cell, 0, qp, dim);
#endif
              for (std::size_t node= 1 ; node < numNodes; ++node) {
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
                grad_val_qp(cell,qp,i,dim) += val_node(cell, node, i) * GradBF(cell, node, qp, dim);
#else
                (grad_val_qp(cell,qp,i,dim)).val() += val_node(cell, node, i).val() * GradBF(cell, node, qp, dim);
                (grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * GradBF(cell, node, qp, dim);
#endif
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

   num_dof = val_node(0,0,0).size();
   neq = workset.wsElNodeEqID[0][0].size();

   Kokkos::parallel_for(DOFVecGradInterpolationBase_Jacobian_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
  PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFVecGradInterpolationBase Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
  }

#ifdef ALBANY_SG
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolationBase SGJacobian");

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];

    offset = p.get<int>("Offset of First DOF");
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }
  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    const int num_dof = val_node(0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t i=0; i<vecDim; i++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, val_node(cell, 0, i).val() * GradBF(cell, 0, qp, dim));
              (grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                (grad_val_qp(cell,qp,i,dim)).val() += val_node(cell, node, i).val() * GradBF(cell, node, qp, dim);
                (grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * GradBF(cell, node, qp, dim);
           }
         }
        }
      }
    }
    //  Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

/*#else

   //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), ScalarT(0.0));
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationBaseJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
*/
  }
#endif

#ifdef ALBANY_ENSEMBLE
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolationBase MPJacobian");

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];

    offset = p.get<int>("Offset of First DOF");
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    const int num_dof = val_node(0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t i=0; i<vecDim; i++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              grad_val_qp(cell,qp,i,dim) = ScalarT(num_dof, val_node(cell, 0, i).val() * GradBF(cell, 0, qp, dim));
              (grad_val_qp(cell,qp,i,dim)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                (grad_val_qp(cell,qp,i,dim)).val() += val_node(cell, node, i).val() * GradBF(cell, node, qp, dim);
                (grad_val_qp(cell,qp,i,dim)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * GradBF(cell, node, qp, dim);
           }
         }
        }
      }
    }
    //  Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

/*#else

   //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), ScalarT(0.0));
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationBaseJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
*/
  }
#endif

} // Namespace PHAL
