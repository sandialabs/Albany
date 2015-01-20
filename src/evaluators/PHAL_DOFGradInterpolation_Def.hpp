//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFGradInterpolation<EvalT, Traits>::
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
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            //ScalarT& gvqp = grad_val_qp(cell,qp,dim);
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
}

//**********************************************************************
template<typename Traits>
DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
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
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);


  int num_dof = val_node(0,0).size();
  int neq = num_dof / numNodes;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            //ScalarT& gvqp = grad_val_qp(cell,qp,dim);
            grad_val_qp(cell,qp,dim) = FadType(num_dof, val_node(cell, 0).val() * GradBF(cell, 0, qp, dim));
            (grad_val_qp(cell,qp,dim)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              (grad_val_qp(cell,qp,dim)).val() += val_node(cell, node).val() * GradBF(cell, node, qp, dim);
              (grad_val_qp(cell,qp,dim)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
}

//**********************************************************************
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
template <class DeviceType, class MDFieldType1, class MDFieldType2, class MDFieldType3 >
class GradInterpolation_noDeriv {
  MDFieldType1 grad_val_qp_;
  MDFieldType2 val_node_;
  MDFieldType3 GradBF_;
  const int numQPs_;
  const int numDims_;
  const int numNodes_;

  public:
  typedef DeviceType device_type;

  GradInterpolation_noDeriv (MDFieldType1 &grad_val_qp,
                             MDFieldType2 &val_node,
                             MDFieldType3 &GradBF,
                             int numQPs,
                             int numDims,
                             int numNodes)
                           : grad_val_qp_(grad_val_qp)
                           , val_node_(val_node)
                           , GradBF_(GradBF)
                           , numQPs_(numQPs)
                           , numDims_(numDims)
                           , numNodes_(numNodes){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
   for (int qp=0; qp < numQPs_; ++qp) {
       for (int dim=0; dim<numDims_; dim++) {
           grad_val_qp_(i,qp,dim) = val_node_(i, 0) * GradBF_(i, 0, qp, dim);
            for (int node= 1 ; node < numNodes_; ++node) {
              grad_val_qp_(i,qp,dim) += val_node_(i, node) * GradBF_(i, node, qp, dim);
          }
        }
      }  
 }
};
//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

//#ifdef NO_KOKKOS_ALBANY
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t dim=0; dim<numDims; dim++) {
            //MeshScalarT& gvqp = grad_val_qp(cell,qp,dim);
            grad_val_qp(cell,qp,dim) = val_node(cell, 0) * GradBF(cell, 0, qp, dim);
            for (std::size_t node= 1 ; node < numNodes; ++node) {
              grad_val_qp(cell,qp,dim) += val_node(cell, node) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
/*#else

  Kokkos::parallel_for ( workset.numCells,  GradInterpolation_noDeriv <  PHX::Device,  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>, PHX::MDField<RealType,Cell,Node>, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>  >( grad_val_qp, val_node, GradBF, numQPs, numDims, numNodes));

#endif*/
}

//**********************************************************************

}

