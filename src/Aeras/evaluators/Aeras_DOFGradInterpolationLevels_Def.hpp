//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFGradInterpolationLevels<EvalT, Traits>::
DOFGradInterpolationLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"),          dl->node_scalar_level),
  GradBF      (p.get<std::string>   ("Gradient BF Name"),       dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("Aeras::DOFGradInterpolationLevels"+PHX::typeAsString<EvalT>());

  //std::cout << "Aeras::DOFGradInterpolationLevels: " << numDims << " " << numQPs << " " << numLevels << std::endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolationLevels<EvalT, Traits>::
operator() (const int cell, const int qp, const int level) const{
  if (numDims==2){
    ScalarT grad_val0=0;
    ScalarT grad_val1=0;
    for (int node= 0 ; node < numNodes; ++node) {
      grad_val0 += val_node(cell, node, level) * GradBF(cell, node, qp, 0);
      grad_val1 += val_node(cell, node, level) * GradBF(cell, node, qp, 1);   
      }
    grad_val_qp(cell,qp,level,0)=grad_val0;
    grad_val_qp(cell,qp,level,1)=grad_val1;
  }
  else{
    for (int dim=0; dim<numDims; dim++) {
      ScalarT grad_val = 0;
      for (int node= 0 ; node < numNodes; ++node) {
        grad_val += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
      }
      grad_val_qp(cell,qp,level,dim) = grad_val;
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

  /*// OG debugging statements
  std::string	myName = this->getName();
  std::cout << "MY GRAD NAME IS ------------- " << myName <<"\n";
  val_node(23,0,0) = 0;
  val_node(23,1,0) = 0;
  val_node(23,2,0) = 0;
  val_node(23,3,0) = 0;
  val_node(23,4,0) = 1;
  val_node(23,5,0) = 2;
  val_node(23,6,0) = 3;
  val_node(23,7,0) = 4;
  val_node(23,8,0) = 0;
  val_node(23,9,0) = 0;
  val_node(23,10,0) = 0;
  val_node(23,11,0) = 0;
  val_node(23,12,0) = 0;
  val_node(23,13,0) = 0;
  val_node(23,14,0) = 0;
  val_node(23,15,0) = 0;

  std::cout << "Printing Geopotential on the first level ------------ \n";
  for (int qp=0; qp < numQPs; ++qp) {
    std::cout << "qp = " << qp << " value = "<< val_node(23,qp,0) <<"\n";
  }
  */

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim<numDims; dim++) {
          grad_val_qp(cell,qp,level,dim) = 0;
          for (int node= 0 ; node < numNodes; ++node) {
            grad_val_qp(cell,qp,level,dim) += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }

/*//OG debugging statements
  std::cout << "Printing Gradient on the first level ------------ \n";
  for (int qp=0; qp < numQPs; ++qp) {
    std::cout << "qp = " << qp << " value = "<< grad_val_qp(23,qp,0,0) << ", "<<grad_val_qp(23,qp,0,1) <<"\n";
  }
*/

#else
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  DOFGradInterpolationLevels_Policy range(
                {0,0,0}, {workset.numCells,numQPs,numLevels}, {256,1,1} );
#else
  DOFGradInterpolationLevels_Policy range ({(int)workset.numCells,
		(int)numQPs,(int)numLevels});
#endif
  Kokkos::Experimental::md_parallel_for(range,*this);
#endif
}

//**********************************************************************
//**********************************************************************

template<typename EvalT, typename Traits>
DOFGradInterpolationLevels_noDeriv<EvalT, Traits>::
DOFGradInterpolationLevels_noDeriv(Teuchos::ParameterList& p,
                             const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"),          dl->node_scalar_level),
  GradBF      (p.get<std::string>   ("Gradient BF Name"),       dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("Aeras::DOFGradInterpolationLevels_noDeriv"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationLevels_noDeriv<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFGradInterpolationLevels_noDeriv<EvalT, Traits>::
operator() (const int cell, const int qp, const int level) const{
  if(numDims==2){
    MeshScalarT gvqp0 = 0;
    MeshScalarT gvqp1 = 0;
    for (int node=0 ; node < numNodes; ++node) {
      gvqp0 += val_node(cell, node, level) * GradBF(cell, node, qp, 0);
      gvqp1 += val_node(cell, node, level) * GradBF(cell, node, qp, 1);
    }//end for
    grad_val_qp(cell,qp,level,0) = gvqp0;
    grad_val_qp(cell,qp,level,1) = gvqp1;
  }
  else{
   for (int dim=0; dim<numDims; dim++) {
     MeshScalarT gvqp =  0;
     for (int node=0 ; node < numNodes; ++node) {
       gvqp += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
     }
     grad_val_qp(cell,qp,level,dim)=gvqp;
   }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationLevels_noDeriv<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //Intrepid2 Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim<numDims; dim++) {
          typename PHAL::Ref<MeshScalarT>::type gvqp = grad_val_qp(cell,qp,level,dim) = 0;
          for (int node=0 ; node < numNodes; ++node) {
            gvqp += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }

#else
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  DOFGradInterpolationLevels_noDeriv_Policy range(
                {0,0,0}, {workset.numCells,numQPs,numLevels}, {256,1,1} );
#else 
  DOFGradInterpolationLevels_noDeriv_Policy range ({(int)workset.numCells,
                (int)numQPs,(int)numLevels});
#endif
  Kokkos::Experimental::md_parallel_for(range,*this);
#endif
}

//**********************************************************************

}

