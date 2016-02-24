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
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<RealType,
		  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > > ("Intrepid2 Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<RealType,
		  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature")),
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

  refWeights        .resize(numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, 2);
  refPoints         .resize(numQPs, 2);
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  vcontra.resize(numNodes, 2);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFDivInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::set(div_val_qp, 0.0);

  if( originalDiv ){
	  for (int cell=0; cell < workset.numCells; ++cell){
		  for (int qp=0; qp < numQPs; ++qp)
			  for (int node= 0 ; node < numNodes; ++node)
				  for (int level=0; level < numLevels; ++level)
					  for (int dim=0; dim<numDims; dim++) {
						  div_val_qp(cell,qp,level) += val_node(cell,node,level,dim) * GradBF(cell,node,qp,dim);
						  //std::cout << "gradbf: " << cell << " " << node << " " << qp << " " << dim << " " << GradBF(cell,node,qp,dim) << std::endl;
						  //std::cout << "val_node " << val_node(cell,node,level,dim) << std::endl;
					  }
	  }

  }//end of original div
  else{
	  //rather slow, needs revision
	  for (int cell=0; cell < workset.numCells; ++cell){
		  for (int level=0; level < numLevels; ++level) {
			  for (std::size_t node=0; node < numNodes; ++node) {

				  const MeshScalarT jinv00 = jacobian_inv(cell, node, 0, 0);
				  const MeshScalarT jinv01 = jacobian_inv(cell, node, 0, 1);
				  const MeshScalarT jinv10 = jacobian_inv(cell, node, 1, 0);
				  const MeshScalarT jinv11 = jacobian_inv(cell, node, 1, 1);
				  const MeshScalarT det_j  = jacobian_det(cell,node);

				  vcontra(node, 0 ) = det_j*(
						  jinv00*val_node(cell, node, level, 0) + jinv01*val_node(cell, node, level, 1) );
				  vcontra(node, 1 ) = det_j*(
						  jinv10*val_node(cell, node, level, 0) + jinv11*val_node(cell, node, level, 1) );
				  for (int qp=0; qp < numQPs; ++qp) {
					  for (int node=0; node < numNodes; ++node) {
						  div_val_qp(cell, qp, level) += vcontra(node, 0)*grad_at_cub_points(node, qp, 0)
            	    						   +  vcontra(node, 1)*grad_at_cub_points(node, qp, 1);
					  }
				  }
				  for (int qp=0; qp < numQPs; ++qp)
					  div_val_qp(cell, qp, level) = div_val_qp(cell, qp, level)/jacobian_det(cell,qp);

			  }//end of nodal loop
		  }//end level loop
	  }//end of cell loop
  }//end of new div

}




}

