//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  SurfaceTLPoroMassResidual<EvalT, Traits>::
  SurfaceTLPoroMassResidual(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")),
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    scalarGrad        (p.get<std::string>("Scalar Gradient Name"),dl->qp_vector),
    surface_Grad_BF     (p.get<std::string>("Surface Scalar Gradient Operator Name"),dl->node_qp_gradient),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    refArea        (p.get<std::string>("Reference Area Name"),dl->qp_scalar),
    porePressure       (p.get<std::string>("Pore Pressure Name"),dl->qp_scalar),
    nodalPorePressure       (p.get<std::string>("Nodal Pore Pressure Name"),dl->node_scalar),
    biotCoefficient      (p.get<std::string>("Biot Coefficient Name"),dl->qp_scalar),
    biotModulus       (p.get<std::string>("Biot Modulus Name"),dl->qp_scalar),
    kcPermeability       (p.get<std::string>("Kozeny-Carman Permeability Name"),dl->qp_scalar),
    deltaTime (p.get<std::string>("Delta Time Name"),dl->workset_scalar),
    poroMassResidual (p.get<std::string>("Residual Name"),dl->node_scalar),
    haveMech(false)
  {
    this->addDependentField(scalarGrad);
    this->addDependentField(surface_Grad_BF);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);    
    this->addDependentField(refArea);
    this->addDependentField(porePressure);
    this->addDependentField(nodalPorePressure);
    this->addDependentField(biotCoefficient);
    this->addDependentField(biotModulus);
    this->addDependentField(kcPermeability);
    this->addDependentField(deltaTime);

    this->addEvaluatedField(poroMassResidual);

    this->setName("Surface Scalar Residual"+PHX::TypeString<EvalT>::value);

    if (p.isType<string>("DefGrad Name")) {
      haveMech = true;

      PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>
        tf(p.get<string>("DefGrad Name"), dl->qp_tensor);
      defGrad = tf;
      this->addDependentField(defGrad);

      PHX::MDField<ScalarT,Cell,QuadPoint>
        tj(p.get<string>("DetDefGrad Name"), dl->qp_scalar);
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

    if (haveMech) {
      // Works space FCs
      C.resize(worksetSize, numQPs, numDims, numDims);
      Cinv.resize(worksetSize, numQPs, numDims, numDims);
      F_inv.resize(worksetSize, numQPs, numDims, numDims);
      F_invT.resize(worksetSize, numQPs, numDims, numDims);
      JF_invT.resize(worksetSize, numQPs, numDims, numDims);
      KJF_invT.resize(worksetSize, numQPs, numDims, numDims);
      Kref.resize(worksetSize, numQPs, numDims, numDims);
    }

    // Allocate workspace
    flux.resize(worksetSize, numQPs, numDims);
    fluxdt.resize(worksetSize, numQPs, numDims);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);

    porePressureName = p.get<std::string>("Pore Pressure Name")+"_old";
    if (haveMech) JName =p.get<std::string>("DetDefGrad Name")+"_old";
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceTLPoroMassResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalarGrad,fm);
    this->utils.setFieldData(surface_Grad_BF,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(refArea,fm);
    this->utils.setFieldData(porePressure, fm);
    this->utils.setFieldData(nodalPorePressure, fm);
    this->utils.setFieldData(biotCoefficient, fm);
    this->utils.setFieldData(biotModulus, fm);
    this->utils.setFieldData(kcPermeability, fm);
    this->utils.setFieldData(deltaTime, fm);
    this->utils.setFieldData(poroMassResidual,fm);
    if (haveMech) {
    	//NOTE: those are in surface elements
      this->utils.setFieldData(defGrad,fm);
      this->utils.setFieldData(J,fm);
    }
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceTLPoroMassResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    typedef Intrepid::FunctionSpaceTools FST;
    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    Albany::MDArray porePressureold = (*workset.stateArrayPtr)[porePressureName];
    Albany::MDArray Jold;
    if (haveMech) {
      Jold = (*workset.stateArrayPtr)[JName];
    }

    ScalarT dt = deltaTime(0);

   // Initialize the residual
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;
             poroMassResidual(cell, node) = 0;
             poroMassResidual(cell, topNode) = 0;
      }
    }

        // Compute pore fluid flux
       if (haveMech) {
       	// Put back the permeability tensor to the reference configuration
    	    RST::inverse(F_inv, defGrad);
           RST::transpose(F_invT, F_inv);
           FST::scalarMultiplyDataData<ScalarT>(JF_invT, J, F_invT);
           FST::scalarMultiplyDataData<ScalarT>(KJF_invT, kcPermeability, JF_invT);
           FST::tensorMultiplyDataData<ScalarT>(Kref, F_inv, KJF_invT);
           FST::tensorMultiplyDataData<ScalarT> (flux, Kref, scalarGrad); // flux_i = k I_ij p_j
       } else {
           FST::scalarMultiplyDataData<ScalarT> (flux, kcPermeability, scalarGrad); // flux_i = kc p_i
       }

       for (std::size_t cell=0; cell < workset.numCells; ++cell){
             for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t dim=0; dim <numDims; ++dim){
                 fluxdt(cell, qp, dim) = -flux(cell,qp,dim)*dt*refArea(cell,qp)*thickness;
               }
             }
       }
          FST::integrate<ScalarT>(poroMassResidual, fluxdt,
          		surface_Grad_BF, Intrepid::COMP_CPP, true); // "true" sums into


    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {
        // initialize the residual
        int topNode = node + numPlaneNodes;

        for (std::size_t pt=0; pt < numQPs; ++pt) {

          // If there is no diffusion, then the residual defines only on the mid-plane value

          // Diffusion term

    	     // orthogonal dimension  contribution
/*
    	  for (std::size_t nodeB(0); nodeB < numPlaneNodes; ++nodeB) {
    		 int topNodeB = nodeB + numPlaneNodes;
    		 for (int i(0); i < numDims; ++i ){
    			 for (int j(0); j < numDims; ++j ){

    			 poroMassResidual(cell, node) +=  refArea(cell, pt)*
    					                                                    refValues(node,pt)/
    					                                                    thickness*N(i)*
     	    		                                                        Kref(cell, pt, i,j)*
     	    		                                                        refValues(nodeB,pt)*
     	    		                                                        (nodalPorePressure(cell,topNodeB)-
     	    		                                                          nodalPorePressure(cell,nodeB))
     	    		                                                        *N(j)*dt;
    			 }
    		 }
    	  }

    	  for (std::size_t nodeB(0); nodeB < numPlaneNodes; ++nodeB) {
    	      		 int topNodeB = nodeB + numPlaneNodes;
    	      		 for (int i(0); i < numDims; ++i ){
    	      			 for (int j(0); j < numDims; ++j ){
    	      				for (int n(0); n < numPlaneDims; ++n){
    	      					poroMassResidual(cell, node) +=  refArea(cell, pt)*
    	      							                                                   refValues(node,pt)*N(i)*
    	       	    		                                                               Kref(cell, pt, i,j)*
    	       	    		                                                               refDualBasis(cell, pt, j, n)*
    	       	    		                                                               refGrads(nodeB,pt,n)*
                                                                                           0.5*(nodalPorePressure(cell,topNodeB)+
                                                                                                    nodalPorePressure(cell,nodeB))*dt;
    	      				}
					 }
				 }
		  }


    	  poroMassResidual(cell, topNode) =  -poroMassResidual(cell, node) ;

    	  for (std::size_t nodeB(0); nodeB < numPlaneNodes; ++nodeB) {
					 int topNodeB = nodeB + numPlaneNodes;
					 for (int m(0); m < numPlaneDims; ++m ){
							for (int i(0); i < numDims; ++i ){
								 for (int j(0); j < numDims; ++j ){
									 poroMassResidual(cell, node) -=  refArea(cell, pt)*
											                                                   refGrads(node, pt,m)*
											                                                   refDualBasis(cell, pt, m, i)*
																							   Kref(cell, pt, i,j)*
																							   (nodalPorePressure(cell,topNodeB)-
																								 nodalPorePressure(cell,nodeB))*
																							   N(j)*dt;
									 poroMassResidual(cell, topNode) -=  refArea(cell, pt)*
											                                                         refGrads(node, pt,m)*
											                                                         refDualBasis(cell, pt, m, i)*
									 																 Kref(cell, pt, i,j)*
									 																 (nodalPorePressure(cell,topNodeB)-
									 																  nodalPorePressure(cell,nodeB))
									 																 *N(j)*dt;
								}
							 }
					 }
    	  }

    	  // parallel direction contribution
    	  for (std::size_t nodeB(0); nodeB < numPlaneNodes; ++nodeB) {
    	      		 int topNodeB = nodeB + numPlaneNodes;
    	      		 for (int m(0); m < numPlaneDims; ++m ){
    	      			 for (int n(0); n < numPlaneDims; ++n){
    	      				for (int i(0); i < numDims; ++i ){
    	      				     for (int j(0); j < numDims; ++j ){
    	      				    	 poroMassResidual(cell, node) -=  refArea(cell, pt)*
    	      				    			                                                   thickness*
    	      				    			                                                   refGrads(node, pt,m)*
    	      				    			                                                   refDualBasis(cell, pt, m, i)*
    	       	    		                                                                   Kref(cell, pt, i,j)*
    	       	    		                                                                   refDualBasis(cell, pt, j, n)*
    	       	    		                                                                   refGrads(nodeB,pt,n)*
    	       	    		                                                                   0.5*(nodalPorePressure(cell,topNodeB)+
    	       	    		                                                                           nodalPorePressure(cell,nodeB))*dt;

    	      				    	 poroMassResidual(cell, topNode) -=  refArea(cell, pt)*
                                                                                                     thickness*
                                                                                                     refGrads(node, pt,m)*
                                                                                                     refDualBasis(cell, pt, m, i)*
                                                                                                     Kref(cell, pt, i,j)*
                                                                                                     refDualBasis(cell, pt, j, n)*
                                                                                                     refGrads(nodeB,pt,n)*
                                                                                                     0.5*(nodalPorePressure(cell,topNodeB)+
                                                                                                              nodalPorePressure(cell,nodeB))*dt;
    	      				     }
    	      				}
    	      			 }
    	      		 }
    	     }

*/

          // Local Rate of Change volumetric constraint term
           poroMassResidual(cell, node) -=
                         refValues(node,pt)*(
                         std::log(J(cell,pt)/Jold(cell, pt))*
                         biotCoefficient(cell,pt) +
                          (porePressure(cell, pt) - porePressureold(cell, pt))/ biotModulus(cell,pt)
                          ) *refArea(cell,pt)*thickness;

           poroMassResidual(cell, topNode) -=
        		           refValues(node,pt)*(
        		           std::log(J(cell,pt)/Jold(cell, pt))*
        		           biotCoefficient(cell,pt) +
        		           (porePressure(cell, pt) - porePressureold(cell, pt))/ biotModulus(cell,pt)
        		           ) *refArea(cell,pt)*thickness;


/*
          // Local Rate of Change pressure term
     	  for (std::size_t nodeB(0); nodeB < numPlaneNodes; ++nodeB) {
     		 int topNodeB = nodeB + numPlaneNodes;
     		 poroMassResidual(cell, node) -=0.5*refValues(node,pt)*refValues(nodeB,pt)*
                                     (nodalPorePressure(cell,nodeB)-porePressureold(cell, nodeB))/
                                     biotModulus(cell,pt)*refArea(cell,pt)*thickness;
     		 poroMassResidual(cell, topNode) -=0.5*refValues(node,pt)*refValues(nodeB,pt)*
                                     (nodalPorePressure(cell,topNodeB)-porePressureold(cell, topNodeB))/
                                     biotModulus(cell,pt)*refArea(cell,pt)*thickness;
     	  }
 */





        } // end integrartion point loop
      } //  end plane node loop

      // Stabilization term (if needed)
    } // end cell loop




  }
  //**********************************************************************  
}
