//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>

//#include "Intrepid_FunctionSpaceTools.hpp"
//#include "Intrepid_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  SurfaceHDiffusionDefResidual<EvalT, Traits>::
  SurfaceHDiffusionDefResidual(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness                          (p.get<double>("thickness")),
    cubature                           (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepidBasis                    (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    scalarGrad                       (p.get<std::string>("Surface Transport Gradient Name"),dl->qp_vector),
    surface_Grad_BF           (p.get<std::string>("Surface Scalar Gradient Operator Name"),dl->node_qp_gradient),
    refDualBasis                    (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal                         (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    refArea                             (p.get<std::string>("Reference Area Name"),dl->qp_scalar),
    transport_                        (p.get<std::string>("Transport Name"),dl->qp_scalar),
    nodal_transport_            (p.get<std::string>("Nodal Transport Name"),dl->node_scalar),
    dL_                                   (p.get<std::string>("Diffusion Coefficient Name"),dl->qp_scalar),
    eff_diff_                            (p.get<std::string>("Effective Diffusivity Name"),dl->qp_scalar),
    convection_coefficient_ (p.get<std::string>("Tau Contribution Name"),dl->qp_scalar),
    strain_rate_factor_         (p.get<std::string>("Strain Rate Factor Name"),dl->qp_scalar),
    hydro_stress_gradient_ (p.get<std::string>("Surface HydroStress Gradient Name"),dl->qp_vector),
    eqps_                               (p.get<std::string>("eqps Name"),dl->qp_scalar),
    deltaTime                         (p.get<std::string>("Delta Time Name"),dl->workset_scalar),
    element_length_                 (p.get<std::string>("Element Length Name"),dl->qp_scalar),
    transport_residual_         (p.get<std::string>("Residual Name"),dl->node_scalar),
    haveMech(false),
    stab_param_(p.get<RealType>("Stabilization Parameter"))
  {
    this->addDependentField(scalarGrad);
    this->addDependentField(surface_Grad_BF);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);    
    this->addDependentField(refArea);
    this->addDependentField(transport_);
    this->addDependentField(nodal_transport_);
    this->addDependentField(dL_);
    this->addDependentField(eff_diff_);
    this->addDependentField(convection_coefficient_);
    this->addDependentField(strain_rate_factor_);
    this->addDependentField(eqps_);
    this->addDependentField(hydro_stress_gradient_);
    this->addDependentField(element_length_);
    this->addDependentField(deltaTime);

    this->addEvaluatedField(transport_residual_);
  //  this->addEvaluatedField(transport_);

    this->setName("Transport Residual"+PHX::TypeString<EvalT>::value);

    if (p.isType<std::string>("DefGrad Name")) {
      haveMech = true;

      PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>
        tf(p.get<std::string>("DefGrad Name"), dl->qp_tensor);
      defGrad = tf;
      this->addDependentField(defGrad);

      PHX::MDField<ScalarT,Cell,QuadPoint>
        tj(p.get<std::string>("DetDefGrad Name"), dl->qp_scalar);
      J = tj;
      this->addDependentField(J);
    }

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;

#ifdef ALBANY_VERBOSE
    std::cout << "in Surface Scalar Residual" << std::endl;
    std::cout << " numPlaneNodes: " << numPlaneNodes << std::endl;
    std::cout << " numPlaneDims: " << numPlaneDims << std::endl;
    std::cout << " numQPs: " << numQPs << std::endl;
    std::cout << " cubature->getNumPoints(): " << cubature->getNumPoints() << std::endl;
    std::cout << " cubature->getDimension(): " << cubature->getDimension() << std::endl;
#endif

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // Allocate workspace
    artificalDL.resize(worksetSize, numQPs);
    stabilizedDL.resize(worksetSize, numQPs);
    flux.resize(worksetSize, numQPs, numDims);

    pterm.resize(worksetSize, numQPs);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);

    transportName = p.get<std::string>("Transport Name")+"_old";
    if (haveMech) eqpsName =p.get<std::string>("eqps Name")+"_old";
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceHDiffusionDefResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalarGrad,fm);
    this->utils.setFieldData(surface_Grad_BF,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(refArea,fm);
    this->utils.setFieldData(transport_, fm);
    this->utils.setFieldData(nodal_transport_, fm);
    this->utils.setFieldData(dL_, fm);
    this->utils.setFieldData(eff_diff_, fm);
    this->utils.setFieldData(convection_coefficient_, fm);
    this->utils.setFieldData(strain_rate_factor_, fm);
    this->utils.setFieldData(element_length_, fm);
    this->utils.setFieldData(deltaTime, fm);
    this->utils.setFieldData(transport_residual_,fm);

    if (haveMech) {
    	//NOTE: those are in surface elements
      this->utils.setFieldData(defGrad,fm);
      this->utils.setFieldData(J,fm);
      this->utils.setFieldData(eqps_, fm);
      this->utils.setFieldData(hydro_stress_gradient_, fm);
    }
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceHDiffusionDefResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
 //   typedef Intrepid::FunctionSpaceTools FST;
//    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    Albany::MDArray transportold = (*workset.stateArrayPtr)[transportName];
    Albany::MDArray scalarGrad_old = (*workset.stateArrayPtr)[CLGradName];
    Albany::MDArray eqps_old;
    if (haveMech) {
      eqps_old = (*workset.stateArrayPtr)[eqpsName];
    }

    ScalarT dt = deltaTime(0);
    ScalarT temp(0.0);
    ScalarT transientTerm(0.0);
    ScalarT stabilizationTerm(0.0);

    // compute artifical diffusivity

     // for 1D this is identical to lumped mass as shown in Prevost's paper.
     for (std::size_t cell=0; cell < workset.numCells; ++cell) {

       for (std::size_t pt=0; pt < numQPs; ++pt) {

    	 if (dt == 0){
    		 artificalDL(cell,pt) = 0;
    	 } else {
             temp = thickness*thickness /6.0*eff_diff_(cell,pt)/dL_(cell,pt)/dt;
             artificalDL(cell,pt) = stab_param_*temp*
        		 	 	 	 	 	 	    (0.5 + 0.5*std::tanh( (temp-1.0)/dL_(cell,pt)  ))*
                                            dL_(cell,pt);
    	 }
    	 	 stabilizedDL(cell,pt) = artificalDL(cell,pt)/( dL_(cell,pt) + artificalDL(cell,pt) );
       }
     }

     for (std::size_t cell=0; cell < workset.numCells; ++cell) {
       for (std::size_t pt=0; pt < numQPs; ++pt) {

    	  Intrepid::Tensor<ScalarT> F(numDims, &defGrad(cell, pt, 0, 0));
    	  Intrepid::Tensor<ScalarT> C_tensor_ = Intrepid::t_dot(F,F);
    	  Intrepid::Tensor<ScalarT> C_inv_tensor_ = Intrepid::inverse(C_tensor_);
    	  Intrepid::Vector<ScalarT> C_grad_(numDims, &scalarGrad(cell, pt, 0));
    	  Intrepid::Vector<ScalarT> C_grad_in_ref_ = Intrepid::dot(C_inv_tensor_, C_grad_ );

         for (std::size_t j=0; j<numDims; j++){
             flux(cell,pt,j) = (1-stabilizedDL(cell,pt))*C_grad_in_ref_(j);
         }

       }
     }

     // Initialize the residual
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t node(0); node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
          transport_residual_(cell, node) = 0;
          transport_residual_(cell, topNode) = 0;
        }
      }

      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
            for (std::size_t node(0); node < numPlaneNodes; ++node) {
              int topNode = node + numPlaneNodes;
              for (std::size_t pt=0; pt < numQPs; ++pt) {
                   for (std::size_t dim=0; dim <numDims; ++dim){

                       transport_residual_(cell, node) +=  flux(cell, pt, dim)*dt*
                        	                                                       surface_Grad_BF(cell, node, pt, dim)*
                        	                                                       refArea(cell,pt);

                      transport_residual_(cell, topNode) += flux(cell, pt, dim)*dt*
                    	                                                        	   surface_Grad_BF(cell, topNode, pt, dim)*
                    	                                                        	   refArea(cell,pt);
                   }
          	 }
        }
      }

    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {
        // initialize the residual
        int topNode = node + numPlaneNodes;

        for (std::size_t pt=0; pt < numQPs; ++pt) {

          // If there is no diffusion, then the residual defines only on the mid-plane value

            temp = 1.0/dL_(cell,pt) + artificalDL(cell,pt);

          // Local rate of change volumetric constraint term
            transientTerm = refValues(node,pt)*
                                          ( eff_diff_(cell,pt)*
                                          (transport_(cell, pt)-transportold(cell, pt) ))*
                                          refArea(cell,pt)*temp;

        	transport_residual_(cell, node) +=  transientTerm;

        	transport_residual_(cell, topNode) += transientTerm;

            if (haveMech) {
            	// Strain rate source term
            	transientTerm  = refValues(node,pt)*
                                            strain_rate_factor_(cell,pt)*
                                            (eqps_(cell,pt)- eqps_old(cell,pt))*
                                            refArea(cell,pt)*temp;

            	transport_residual_(cell, node) += transientTerm;

            	transport_residual_(cell, topNode) +=  transientTerm;

            	// hydrostatic stress term
        		for (std::size_t dim=0; dim < numDims; ++dim) {

        			    transport_residual_(cell, node) -= surface_Grad_BF(cell, node, pt, dim)*
        			    						  	  	                 convection_coefficient_(cell,pt)*transport_(cell,pt)*
        			    				                                 hydro_stress_gradient_(cell,pt, dim)*dt*
        			    				       	   	   	   	   	   	   	 refArea(cell,pt)*temp;

        				transport_residual_(cell, topNode) -= surface_Grad_BF(cell, topNode, pt, dim)*
        				                                                  convection_coefficient_(cell,pt)*transport_(cell,pt)*
        				                                                  hydro_stress_gradient_(cell,pt, dim)*dt*
        		                                                          refArea(cell,pt)*temp;
        		}
            }
        } // end integrartion point loop
      } //  end plane node loop
    } // end cell loop

    // Stabilization term (if needed)

    ScalarT CLPbar(0);
    ScalarT vol(0);

    for (std::size_t cell=0; cell < workset.numCells; ++cell){
      CLPbar = 0.0;
      vol = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        CLPbar += refArea(cell,qp)*
        		           (transport_(cell,qp) - transportold(cell, qp));
        vol  += refArea(cell,qp);
      }
      CLPbar /= vol;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        pterm(cell,qp) = CLPbar;
      }
    }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numPlaneNodes; ++node) {

    	  int topNode = node + numPlaneNodes;

    	  for (std::size_t qp=0; qp < numQPs; ++qp) {

    		  temp = 1.0/dL_(cell,qp) + artificalDL(cell,qp);

    		  stabilizationTerm= stab_param_*eff_diff_(cell, qp)*
                                              (-transport_(cell, qp)+transportold(cell, qp)+
                                             	pterm(cell,qp))*refValues(node,qp)*
                                             	refArea(cell,qp)*temp;

    		  transport_residual_(cell,node)       -= stabilizationTerm;
        	  transport_residual_(cell,topNode) -= stabilizationTerm;
        }
      }
    }


  }
  //**********************************************************************  
}
