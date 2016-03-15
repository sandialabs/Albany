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
VorticityLevels<EvalT, Traits>::
VorticityLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node   (p.get<std::string>   ("Velx"),           dl->node_vector_level),
  GradBF     (p.get<std::string>   ("Gradient BF Name"),        dl->node_qp_gradient),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<RealType,
		  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > > ("Intrepid2 Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<RealType,
		  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature")),
  vort_val_qp (p.get<std::string>   ("Vorticity Variable Name"),dl->qp_scalar_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addDependentField(jacobian_det);
  this->addDependentField(jacobian);
  this->addEvaluatedField(vort_val_qp);

  this->setName("Aeras::VorticityLevels"+PHX::typeAsString<EvalT>());

  std::cout << "In Vorticity, name = " << this->getName() << "\n";
  //std::cout<< "Aeras::VorticityLevels: " << numNodes << " " << numDims << " " << numQPs << " " << numLevels << std::endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void VorticityLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_det, fm);
  this->utils.setFieldData(vort_val_qp,fm);

  refWeights        .resize(numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, 2);
  refPoints         .resize(numQPs, 2);
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  vco.resize(numNodes, 2);
}

//**********************************************************************
//Kokkos kernals
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void VorticityLevels<EvalT, Traits>::
operator() (const Vorticity_Tag& tag, const int & cell) const {

   
  for (int qp=0; qp < numQPs; ++qp) {
    for (int node= 0 ; node < numNodes; ++node) { 
      for (int level=0; level < numLevels; ++level) {
         vort_val_qp(cell,qp,level) = 0.0;
      }
    }
  }

  for (int qp=0; qp < numQPs; ++qp) {
    for (int node= 0 ; node < numNodes; ++node) { 
      for (int level=0; level < numLevels; ++level) {
         vort_val_qp(cell,qp,level) += (val_node(cell,node,level,1) * GradBF(cell,node,qp,0) 
                                     -  val_node(cell,node,level,0) * GradBF(cell,node,qp,1));
      }
    }
  }

}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void VorticityLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{


#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  PHAL::set(vort_val_qp, 0.0);

#define ORIGINALVORT 0
#if ORIGINALVORT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node= 0 ; node < numNodes; ++node) { 
        for (int level=0; level < numLevels; ++level) {
            vort_val_qp(cell,qp,level) += (val_node(cell,node,level,1) * GradBF(cell,node,qp,0) 
                                        -  val_node(cell,node,level,0) * GradBF(cell,node,qp,1));
        }
      }
    }
  }
#else
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int level=0; level < numLevels; ++level) {

	  for (std::size_t node=0; node < numNodes; ++node) {

		const MeshScalarT j00 = jacobian(cell, node, 0, 0);
		const MeshScalarT j01 = jacobian(cell, node, 0, 1);
		const MeshScalarT j10 = jacobian(cell, node, 1, 0);
		const MeshScalarT j11 = jacobian(cell, node, 1, 1);

		vco(node, 0 ) = j00*val_node(cell, node, level, 0) + j10*val_node(cell, node, level, 1);
		vco(node, 1 ) = j01*val_node(cell, node, level, 0) + j11*val_node(cell, node, level, 1);
	  }

	  for (std::size_t qp=0; qp < numQPs; ++qp) {
		for (std::size_t node=0; node < numNodes; ++node) {

		  vort_val_qp(cell,qp,level) += vco(node, 1)*grad_at_cub_points(node, qp,0)
              		                  - vco(node, 0)*grad_at_cub_points(node, qp,1);
		}
	    vort_val_qp(cell,qp,level) /= jacobian_det(cell,qp);
	  }
    }
  }

/*
if( this->getName() == "Aeras::VorticityLevels<Residual>"){
	  for (int cell=0; cell < 1; ++cell) {
	    for (int level=0; level < numLevels; ++level) {
	    	for (std::size_t node=0; node < numNodes; ++node) {
	    	   val_node(cell,node,level,0) = node;
	    	   val_node(cell,node,level,1) = node;
	    	}
	    }
	  }
	  for (int cell=0; cell < 1; ++cell) {
	    for (int level=0; level < numLevels; ++level) {
		  for (std::size_t node=0; node < numNodes; ++node) {

			const MeshScalarT j00 = jacobian(cell, node, 0, 0);
			const MeshScalarT j01 = jacobian(cell, node, 0, 1);
			const MeshScalarT j10 = jacobian(cell, node, 1, 0);
			const MeshScalarT j11 = jacobian(cell, node, 1, 1);

			vco(node, 0 ) = j00*val_node(cell, node, level, 0) + j10*val_node(cell, node, level, 1);
			vco(node, 1 ) = j01*val_node(cell, node, level, 0) + j11*val_node(cell, node, level, 1);
		  }

		  for (std::size_t qp=0; qp < numQPs; ++qp) {
			for (std::size_t node=0; node < numNodes; ++node) {

			  vort_val_qp(cell,qp,level) += vco(node, 1)*grad_at_cub_points(node, qp,0)
	              		                  - vco(node, 0)*grad_at_cub_points(node, qp,1);
			}
		    vort_val_qp(cell,qp,level) /= jacobian_det(cell,qp);
		  }
	    }
	  }
	  for (int level=0; level < numLevels; ++level) {
		  std::cout << "Vorticity DEBUG, level = " << level << "\n";
	      for (std::size_t node=0; node < numNodes; ++node) {
	    	 std::cout << "vort(" << node << ") = " << vort_val_qp(0,node,level) <<"\n";
	      }
	  }

}
*/

#endif


#else

  Kokkos::parallel_for(Vorticity_Policy(0,workset.numCells),*this);

#endif

}

}

