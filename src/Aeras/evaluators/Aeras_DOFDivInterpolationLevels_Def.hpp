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
DOFDivInterpolationLevels<EvalT, Traits>::
DOFDivInterpolationLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node   (p.get<std::string>   ("Variable Name"),          dl->node_vector_level),
  GradBF     (p.get<std::string>   ("Gradient BF Name"),       dl->node_qp_gradient),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  div_val_qp (p.get<std::string>   ("Divergence Variable Name"), dl->qp_scalar_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addDependentField(jacobian_det);
  this->addDependentField(jacobian_inv);
  this->addEvaluatedField(div_val_qp);

  this->setName("Aeras::DOFDivInterpolationLevels"+PHX::typeAsString<EvalT>());

  Teuchos::ParameterList* xsa_params =
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");
  originalDiv = xsa_params->get<bool>("Original Divergence", true);

  std::cout << "ORIGINAL DIV ? " << originalDiv <<"\n";

  //OG Since there are a few evaluators that use div, it is possible to control
  //print statements with it:
  //myName = p.get<std::string>   ("Divergence Variable Name");
  //if(myName == "Utr1_divergence" )...
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFDivInterpolationLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(jacobian_inv, fm);
  this->utils.setFieldData(jacobian_det, fm);
  this->utils.setFieldData(div_val_qp,fm);

  refWeights         = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPs, 2);
  refPoints          = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, 2);
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  vcontra = Kokkos::createDynRankView(val_node.get_view(), "XXX", numNodes, 2);
#endif
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFDivInterpolationLevels<EvalT, Traits>::
operator() (const DOFDivInterpolationLevels_originalDiv_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
    for (int level=0; level < numLevels; ++level) {
      div_val_qp(cell,qp,level) = 0;
      for (int node= 0 ; node < numNodes; ++node) {
        for (int dim=0; dim<numDims; dim++) {
          div_val_qp(cell,qp,level) += val_node(cell,node,level,dim) * GradBF(cell,node,qp,dim);
        }
      }
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFDivInterpolationLevels<EvalT, Traits>::
operator() (const DOFDivInterpolationLevels_Tag& tag, const int& cell) const{
  Kokkos::DynRankView<ScalarT, PHX::Device> vcontra = createDynRankView(div_val_qp.get_view(), "vcontra", numNodes, numLevels, 2);
  for (std::size_t node=0; node < numNodes; ++node) {
    const MeshScalarT jinv00 = jacobian_inv(cell, node, 0, 0);
    const MeshScalarT jinv01 = jacobian_inv(cell, node, 0, 1);
    const MeshScalarT jinv10 = jacobian_inv(cell, node, 1, 0);
    const MeshScalarT jinv11 = jacobian_inv(cell, node, 1, 1);
    const MeshScalarT det_j  = jacobian_det(cell, node);

    for (int level=0; level < numLevels; ++level) {
      vcontra(node, level, 0) = det_j*(jinv00*val_node(cell, node, level, 0) + jinv01*val_node(cell, node, level, 1) );
      vcontra(node, level, 1) = det_j*(jinv10*val_node(cell, node, level, 0) + jinv11*val_node(cell, node, level, 1) );
    }
  }

  for (int qp=0; qp < numQPs; ++qp) {
    for (int level=0; level < numLevels; ++level) {
      div_val_qp(cell, qp, level) = 0;
      for (int node=0; node < numNodes; ++node) {
        div_val_qp(cell, qp, level) += vcontra(node, level, 0)*grad_at_cub_points(node, qp, 0)
                                    +  vcontra(node, level, 1)*grad_at_cub_points(node, qp, 1);
      }
      div_val_qp(cell, qp, level) /= jacobian_det(cell, qp);
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFDivInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if ( originalDiv ) {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        for (int level=0; level < numLevels; ++level) {
          div_val_qp(cell,qp,level) = 0;
          for (int node= 0 ; node < numNodes; ++node) {
            for (int dim=0; dim<numDims; dim++) {
              div_val_qp(cell,qp,level) += val_node(cell,node,level,dim) * GradBF(cell,node,qp,dim);
            }
          }
        }
      }
    }
  }//end of original div

  else {
    //rather slow, needs revision
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int level=0; level < numLevels; ++level) {
        for (std::size_t node=0; node < numNodes; ++node) {
          const MeshScalarT jinv00 = jacobian_inv(cell, node, 0, 0);
          const MeshScalarT jinv01 = jacobian_inv(cell, node, 0, 1);
          const MeshScalarT jinv10 = jacobian_inv(cell, node, 1, 0);
          const MeshScalarT jinv11 = jacobian_inv(cell, node, 1, 1);
          const MeshScalarT det_j  = jacobian_det(cell,node);

          vcontra(node, 0 ) = det_j*(jinv00*val_node(cell, node, level, 0) + jinv01*val_node(cell, node, level, 1) );
          vcontra(node, 1 ) = det_j*(jinv10*val_node(cell, node, level, 0) + jinv11*val_node(cell, node, level, 1) );
        }//end of nodal loop

        for (int qp=0; qp < numQPs; ++qp) {
          div_val_qp(cell, qp, level) = 0;
          for (int node=0; node < numNodes; ++node) {
            div_val_qp(cell, qp, level) += vcontra(node, 0)*grad_at_cub_points(node, qp, 0)
                                        +  vcontra(node, 1)*grad_at_cub_points(node, qp, 1);
          }
          div_val_qp(cell, qp, level) /= jacobian_det(cell,qp);
        }
      }//end level loop
    }//end of cell loop
  }//end of new div

#else
  if ( originalDiv ) {
    Kokkos::parallel_for(DOFDivInterpolationLevels_originalDiv_Policy(0,workset.numCells),*this);
  }
  else {
    Kokkos::parallel_for(DOFDivInterpolationLevels_Policy(0,workset.numCells),*this);
  }

#endif
}
}
