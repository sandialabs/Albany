//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef ALBANY_TIMER
#include <chrono>
#endif

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  DOFVecGradInterpolation<EvalT, Traits>::
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
  template<typename EvalT, typename Traits>
  void DOFVecGradInterpolation<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //*********************************************************************
  //KOKKOS functor Residual
  template < class DeviceType, class MDFieldType1, class MDFieldType2, class MDFieldType3 >
  class VecGradInterpolation {
  MDFieldType1 GradBF_;
  MDFieldType2 val_node_;
  MDFieldType3 U_;
  const int numQPs_;
  const int numNodes_;
  const int numDims_;
  const int vecDims_;

  public:
  typedef DeviceType device_type;

  VecGradInterpolation (MDFieldType1 &GradBF,
                        MDFieldType2 &val_node,
                        MDFieldType3 &u,
                        int numQPs,
			int numNodes,
			int numDims,
			int vecDims)
                       : GradBF_(GradBF)
                       , val_node_(val_node)
                       , U_(u)
 		       , numQPs_(numQPs)
                       , numNodes_(numNodes)
                       , numDims_(numDims)
                       , vecDims_(vecDims){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
  for (int j=0; j<numQPs_; j++){
    for (int k=0; k<vecDims_; k++){
      for (int dim=0; dim<numDims_; dim++){
       U_(i,j,k,dim) = val_node_(i, 0, k) * GradBF_(i, 0, j, dim);
       for (int node= 1 ; node < numNodes_; ++node) {
         U_(i,j,k,dim) += val_node_(i, node, k) * GradBF_(i, node, j, dim);
       }
      }
    }
   }
  }
};

  template<typename EvalT, typename Traits>
  void DOFVecGradInterpolation<EvalT, Traits>::
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

    //  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);
#else
 Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), 0.0);

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto start = std::chrono::high_resolution_clock::now(); 
#endif

 Kokkos::parallel_for ( workset.numCells,  VecGradInterpolation < PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim));

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
  DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }
  //**********************************************************************
  //Kokkos functor Jacabian
  template <typename ScalarT, class DeviceType, class MDFieldType, class MDFieldTypeFad1, class MDFieldTypeFad2>
  class VecGradInterpolationJacobian { 
  MDFieldType GradBF_;
  MDFieldTypeFad1 val_node_;
  MDFieldTypeFad2 U_;
  const int numQPs_;
  const int numNodes_;
  const int numDims_;
  const int vecDims_;
  const int offset_;

 public:
 typedef DeviceType device_type;

 VecGradInterpolationJacobian (MDFieldType &GradBF,
                       MDFieldTypeFad1 &val_node,
                       MDFieldTypeFad2 &u,
                       int numQPs,
                       int numNodes,
                       int numDims,
                       int vecDims,
                       int offset)
                       : GradBF_(GradBF)
                       , val_node_(val_node)
                       , U_(u)
                       , numQPs_(numQPs)
                       , numNodes_(numNodes)
                       , numDims_(numDims)
                       , vecDims_(vecDims)
                       , offset_(offset){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
    int num_dof = val_node_(0,0,0).size();
    int neq = num_dof / numNodes_;
    
    for (int qp=0; qp < numQPs_; ++qp) {
       for (int vec=0; vec<vecDims_; vec++) {
           for (int dim=0; dim<numDims_; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes
              U_(i,qp,vec,dim) = ScalarT(num_dof, val_node_(i, 0, vec).val() * GradBF_(i, 0, qp, dim));
              (U_(i,qp,vec,dim)).fastAccessDx(offset_+vec) = val_node_(i, 0, vec).fastAccessDx(offset_+vec) * GradBF_(i, 0, qp, dim);
              for (int node= 1 ; node < numNodes_; ++node) {
                (U_(i,qp,vec,dim)).val() += val_node_(i, node, vec).val() * GradBF_(i, node, qp, dim);
                (U_(i,qp,vec,dim)).fastAccessDx(neq*node+offset_+vec) += val_node_(i, node, vec).fastAccessDx(neq*node+offset_+vec) * GradBF_(i, node, qp, dim);
            }
          }
        }
      }

 }
  };
  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
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
    //  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

   //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), ScalarT(0.0));
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));

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
  DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }
  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
    //  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

#else
  
   //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), ScalarT(0.0));
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
  }
#endif

#ifdef ALBANY_ENSEMBLE
  //**********************************************************************
  template<typename Traits>
  DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFVecGradInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
    //  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

#else
  
   //Kokkos::deep_copy(grad_val_qp.get_kokkos_view(), ScalarT(0.0));
   Kokkos::parallel_for ( workset.numCells,  VecGradInterpolationJacobian <ScalarT,  PHX::Device, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,Node,VecDim>,  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>  >(GradBF, val_node, grad_val_qp, numQPs, numNodes, numDims, vecDim, offset));
#endif
  }
#endif
  
  //**********************************************************************
}
