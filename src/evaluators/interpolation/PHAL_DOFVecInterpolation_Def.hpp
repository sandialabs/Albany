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
DOFVecInterpolationBase<EvalT, Traits, ScalarT>::
DOFVecInterpolationBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolationBase"+PHX::print<EvalT>());
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFVecInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
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
template<typename EvalT, typename Traits, typename ScalarT>
void DOFVecInterpolationBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;
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

   Kokkos::parallel_for ( workset.numCells,
       VecInterpolation<PHX::Device,
                        decltype(BF),
                        decltype(val_node),
                        decltype(val_qp) >(
                          BF, val_node, val_qp, numQPs, numNodes, vecDim));

#ifdef ALBANY_TIMER
PHX::Device::fence();
auto elapsed = std::chrono::high_resolution_clock::now() - start;
long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
std::cout<< "DOFVecInterpolationBase Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif

#endif
}

// Specialization for Jacobian evaluation taking advantage of known sparsity
// This assumes that the Mesh coordinates are not of FAD type
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
void FastSolutionVecInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  int num_dof = this->val_node(0,0,0).size();
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const int neq = workset.wsElNodeEqID.extent(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < this->numQPs; ++qp) {
      for (std::size_t i=0; i<this->vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        this->val_qp(cell,qp,i) = ScalarT(num_dof, this->val_node(cell, 0, i).val() * this->BF(cell, 0, qp));
        (this->val_qp(cell,qp,i)).fastAccessDx(offset+i) = this->val_node(cell, 0, i).fastAccessDx(offset+i) * this->BF(cell, 0, qp);
        for (std::size_t node=1; node < this->numNodes; ++node) {
          (this->val_qp(cell,qp,i)).val() += this->val_node(cell, node, i).val() * this->BF(cell, node, qp);
          (this->val_qp(cell,qp,i)).fastAccessDx(neq*node+offset+i) += this->val_node(cell, node, i).fastAccessDx(neq*node+offset+i) * this->BF(cell, node, qp);
        }
      }
    }
  }
//Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(val_qp, val_node, BF);
#else
  Kokkos::parallel_for(workset.numCells,
      VecInterpolationJacob<
        ScalarT,
        PHX::Device,
        decltype(this->BF),
        decltype(this->val_node),
        decltype(this->val_qp)>(
            this->BF, this->val_node, this->val_qp, this->numNodes, this->numQPs, this->vecDim, num_dof, offset));
#endif

}

}
