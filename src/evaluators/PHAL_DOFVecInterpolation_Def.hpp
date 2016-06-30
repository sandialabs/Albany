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
DOFVecInterpolation<EvalT, Traits, Type>::
DOFVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolation" );
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolation<EvalT, Traits, Type>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}
//**********************************************************************
//Kokkos kernel for Residual
template <class DeviceType, class MDFieldType1, class MDFieldType2, class MDFieldType3 >
class VecInterpolation {
 MDFieldType1  BF_;
 MDFieldType2  val_node_;
 MDFieldType3 U_;
 const int numQPs_;
 const int numNodes_;
 const int vecDims_;

 public:
 typedef DeviceType device_type;

 VecInterpolation ( MDFieldType1  &BF,
                    MDFieldType2  &val_node,
                    MDFieldType3 &U,
                    int numQPs,
                    int numNodes,
                    int vecDims)
  : BF_(BF)
  , val_node_(val_node)
  , U_(U)
  , numQPs_(numQPs)
  , numNodes_(numNodes)
  , vecDims_(vecDims){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
   for (int qp=0; qp < numQPs_; ++qp) {
      for (int vec=0; vec<vecDims_; vec++) {
      U_(i,qp,vec) = val_node_(i, 0, vec) * BF_(i, 0, qp);
        for (int node=1; node < numNodes_; ++node) {
        U_(i,qp,vec) += val_node_(i, node, vec) * BF_(i, node, qp);
        }
      }
    }
   }
};

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolation<EvalT, Traits, Type>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        val_qp(cell,qp,i) = val_node(cell, 0, i) * BF(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          val_qp(cell,qp,i) += val_node(cell, node, i) * BF(cell, node, qp);
        } 
      } 
    } 
  }
//  Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(val_qp, val_node, BF);
#else

#ifdef ALBANY_TIMER
auto start = std::chrono::high_resolution_clock::now();
#endif

   Kokkos::parallel_for ( workset.numCells,  VecInterpolation <  PHX::Device,  PHX::MDField<RealType,Cell,Node,QuadPoint>, PHX::MDField<ScalarT,Cell,Node,VecDim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> >(BF, val_node, val_qp, numQPs, numNodes, vecDim));

#ifdef ALBANY_TIMER
PHX::Device::fence();
auto elapsed = std::chrono::high_resolution_clock::now() - start;
long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
std::cout<< "DOFVecInterpolation Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
}

// Specializations for Jacobian are identical for 3 Jacobian types

//**********************************************************************
template<typename Traits>
DOFVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
DOFVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolation Jacobian");
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}
//**********************************************************************
//Kokkos kernel for Jacobian
template <typename ScalarT, class Device, class MDFieldType, class MDFieldTypeFad1, class MDFieldTypeFad2>
class VecInterpolationJacob {
 MDFieldType  BF_;
 MDFieldTypeFad1  val_node_;
 MDFieldTypeFad2 U_;
 const int numNodes_;
 const int numQPs_;
 const int vecDims_;
 const int num_dof_;
 const int offset_; 

 public:
 typedef Device device_type;

 VecInterpolationJacob ( MDFieldType  &BF,
                         MDFieldTypeFad1  &val_node,
                         MDFieldTypeFad2 &U,
                         int numNodes,
                         int numQPs,
                         int vecDims,
                         int num_dof,
                         int offset)
  : BF_(BF)
  , val_node_(val_node)
  , U_(U)
  , numNodes_(numNodes)
  , numQPs_(numQPs)
  , vecDims_(vecDims)
  , num_dof_(num_dof)
  , offset_(offset){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
   const int neq = num_dof_ / numNodes_;
   for (int qp=0; qp < numQPs_; ++qp) {
      for (int vec=0; vec<vecDims_; vec++) {
           U_(i,qp,vec) = ScalarT(num_dof_, val_node_(i, 0, vec).val() * BF_(i, 0, qp));
          (U_(i,qp,vec)).fastAccessDx(offset_+vec) = val_node_(i, 0, vec).fastAccessDx(offset_+vec) * BF_(i, 0, qp);
           for (int node=1; node < numNodes_; ++node) {
            (U_(i,qp,vec)).val() += val_node_(i, node, vec).val() * BF_(i, node, qp);
            (U_(i,qp,vec)).fastAccessDx(neq*node+offset_+vec) += val_node_(i, node, vec).fastAccessDx(neq*node+offset_+vec) * BF_(i, node, qp);
           }
      }
    }
   }
};

//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits, FadType>::
evaluateFields(typename Traits::EvalData workset)
{
  int num_dof = val_node(0,0,0).size();
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
	val_qp(cell,qp,i) = ScalarT(num_dof, val_node(cell, 0, i).val() * BF(cell, 0, qp));
        (val_qp(cell,qp,i)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * BF(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          (val_qp(cell,qp,i)).val() += val_node(cell, node, i).val() * BF(cell, node, qp);
          (val_qp(cell,qp,i)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * BF(cell, node, qp);
        } 
      } 
    } 
  }
//Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(val_qp, val_node, BF);
#else
  Kokkos::parallel_for ( workset.numCells,  VecInterpolationJacob <ScalarT,  PHX::Device,  PHX::MDField<RealType,Cell,Node,QuadPoint>, PHX::MDField<ScalarT,Cell,Node,VecDim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> >(BF, val_node, val_qp, numNodes, numQPs, vecDim, num_dof, offset));
#endif

}

#ifdef ALBANY_SG
//**********************************************************************
template<typename Traits>
DOFVecInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, SGFadType>::
DOFVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolation SGJacobian");
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, SGFadType>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}
//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits, SGFadType>::
evaluateFields(typename Traits::EvalData workset)
{
  int num_dof = val_node(0,0,0).size();
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
	val_qp(cell,qp,i) = ScalarT(num_dof, val_node(cell, 0, i).val() * BF(cell, 0, qp));
        (val_qp(cell,qp,i)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * BF(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          (val_qp(cell,qp,i)).val() += val_node(cell, node, i).val() * BF(cell, node, qp);
          (val_qp(cell,qp,i)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * BF(cell, node, qp);
        } 
      } 
    } 
  }
//Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(val_qp, val_node, BF);
#else
  Kokkos::parallel_for ( workset.numCells,  VecInterpolationJacob <ScalarT,  PHX::Device,  PHX::MDField<RealType,Cell,Node,QuadPoint>, PHX::MDField<ScalarT,Cell,Node,VecDim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> >(BF, val_node, val_qp, numNodes, numQPs, vecDim, num_dof, offset));
#endif

}
#endif

#ifdef ALBANY_ENSEMBLE
//**********************************************************************
template<typename Traits>
DOFVecInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, MPFadType>::
DOFVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolation MPJacobian");
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, MPFadType>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}
//**********************************************************************
template<typename Traits>
void DOFVecInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits, MPFadType>::
evaluateFields(typename Traits::EvalData workset)
{
  int num_dof = val_node(0,0,0).size();
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
	val_qp(cell,qp,i) = ScalarT(num_dof, val_node(cell, 0, i).val() * BF(cell, 0, qp));
        (val_qp(cell,qp,i)).fastAccessDx(offset+i) = val_node(cell, 0, i).fastAccessDx(offset+i) * BF(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          (val_qp(cell,qp,i)).val() += val_node(cell, node, i).val() * BF(cell, node, qp);
          (val_qp(cell,qp,i)).fastAccessDx(neq*node+offset+i) += val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * BF(cell, node, qp);
        } 
      } 
    } 
  }
//Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(val_qp, val_node, BF);
#else

#ifdef ALBANY_TIMER
 auto start = std::chrono::high_resolution_clock::now();
#endif

  Kokkos::parallel_for ( workset.numCells,  VecInterpolationJacob <ScalarT,  PHX::Device,  PHX::MDField<RealType,Cell,Node,QuadPoint>, PHX::MDField<ScalarT,Cell,Node,VecDim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> >(BF, val_node, val_qp, numNodes, numQPs, vecDim, num_dof, offset));

#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "DOFVecInterpolation Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
}
#endif

}
