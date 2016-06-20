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
  DOFVecGradInterpolation<EvalT, Traits, Type>::
  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolation" );

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];
  }

  //**********************************************************************
  template<typename EvalT, typename Traits,typename Type>
  void DOFVecGradInterpolation<EvalT, Traits, Type>::
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
  template<typename EvalT, typename Traits, typename Type>
  KOKKOS_INLINE_FUNCTION
  void DOFVecGradInterpolation<EvalT, Traits, Type>::
  operator() (const DOFVecGradInterpolation_Residual_Tag& tag, const int& cell) const {


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
  template<typename EvalT, typename Traits, typename Type>
  void DOFVecGradInterpolation<EvalT, Traits, Type>::
  evaluateFields(typename Traits::EvalData workset)
  {
	#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    // This is needed, since evaluate currently sums into    
   Kokkos::deep_copy(grad_val_qp.get_static_view(), 0.0);

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
  Kokkos::parallel_for(DOFVecGradInterpolation_Residual_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFVecGradInterpolation Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
  }
  
  // Specializations for all 3 Jacobian types are identical
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolation Jacobian");

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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
  operator() (const DOFVecGradInterpolation_Jacobian_Tag& tag, const int& cell) const {
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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
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

   Kokkos::parallel_for(DOFVecGradInterpolation_Jacobian_Policy(0,workset.numCells),*this);

#ifdef ALBANY_TIMER
  PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFVecGradInterpolation Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
  }

#ifdef ALBANY_SG
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, FadType>::
  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolation SGJacobian");

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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, FadType>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }
  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, FadType>::
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
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
*/
  }
#endif

#ifdef ALBANY_ENSEMBLE
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, FadType>::
  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolation MPJacobian");

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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, FadType>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, FadType>::
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
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
*/
  }
#endif
  
  //**********************************************************************
}
