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
    scalarJump        (p.get<std::string>("Scalar Jump Name"),dl->qp_scalar),
    currentBasis   (p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    refArea        (p.get<std::string>("Reference Area Name"),dl->qp_scalar),
    porePressure       (p.get<std::string>("Pore Pressure Name"),dl->qp_scalar),
    biotCoefficient      (p.get<std::string>("Biot Coefficient Name"),dl->qp_scalar),
    biotModulus       (p.get<std::string>("Biot Modulus Name"),dl->qp_scalar),
    kcPermeability       (p.get<std::string>("Kozeny-Carman Permeability Name"),dl->qp_scalar),
    deltaTime (p.get<std::string>("Delta Time Name"),dl->workset_scalar),
    poroMassResidual (p.get<std::string>("Residual Name"),dl->node_scalar),
    haveMech(false)
  {
    this->addDependentField(scalarGrad);
    this->addDependentField(scalarJump);
    this->addDependentField(currentBasis);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);    
    this->addDependentField(refArea);
    this->addDependentField(porePressure);
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
    this->utils.setFieldData(scalarJump,fm);
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(refArea,fm);
    this->utils.setFieldData(porePressure, fm);
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

	// Put back the permeability tensor to the reference configuration
    RST::inverse(F_inv, defGrad);
    RST::transpose(F_invT, F_inv);
    FST::scalarMultiplyDataData<ScalarT>(JF_invT, J, F_invT);
    FST::scalarMultiplyDataData<ScalarT>(KJF_invT, kcPermeability, JF_invT);
    FST::tensorMultiplyDataData<ScalarT>(Kref, F_inv, KJF_invT);

     // Compute pore fluid flux
    if (haveMech) {
	    RST::inverse(F_inv, defGrad);
        RST::transpose(F_invT, F_inv);
        FST::scalarMultiplyDataData<ScalarT>(JF_invT, J, F_invT);
        FST::scalarMultiplyDataData<ScalarT>(KJF_invT, kcPermeability, JF_invT);
        FST::tensorMultiplyDataData<ScalarT>(Kref, F_inv, KJF_invT);
        FST::tensorMultiplyDataData<ScalarT> (flux, Kref, scalarGrad); // flux_i = k I_ij p_j
    } else {
        FST::scalarMultiplyDataData<ScalarT> (flux, kcPermeability, scalarGrad); // flux_i = kc p_i
    }


    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {
        // initialize the residual
        int topNode = node + numPlaneNodes;

        poroMassResidual(cell, node) = 0;
        poroMassResidual(cell, topNode) = 0;

        for (std::size_t pt=0; pt < numQPs; ++pt) {

            Intrepid::Vector<ScalarT> G_0(3, &refDualBasis(cell, pt, 0, 0));
            Intrepid::Vector<ScalarT> G_1(3, &refDualBasis(cell, pt, 1, 0));
            Intrepid::Vector<ScalarT> G_2(3, &refDualBasis(cell, pt, 2, 0));
            Intrepid::Vector<ScalarT> N(3, &refNormal(cell, pt, 0));

            // Need to inverse basis [G_0 ; G_1; G_2] and none of them should be normalized
            Intrepid::Tensor<ScalarT> gBasis(3, &refDualBasis(cell, pt, 0, 0));
            Intrepid::Tensor<ScalarT> invRefDualBasis(3);

            // This map the position vector from parent to current configuration in R^3
            gBasis = Intrepid::transpose(gBasis);
            invRefDualBasis = Intrepid::inverse(gBasis);

            Intrepid::Vector<ScalarT> invG_0(3, &invRefDualBasis(0, 0));
            Intrepid::Vector<ScalarT> invG_1(3, &invRefDualBasis(1, 0));
            Intrepid::Vector<ScalarT> invG_2(3, &invRefDualBasis(2, 0));

          // note: refArea = |J| * weight at integration point
         // note: Intergation point at mid-plane only.



          // Local Rate of Change volumetric constraint term
          poroMassResidual(cell, node) -= refValues(node,pt)*
            std::log(J(cell,pt)/Jold(cell, pt))*
            biotCoefficient(cell,pt)*refArea(cell,pt)*thickness;

          // Local Rate of Change pressure term
          poroMassResidual(cell, node) -= refValues(node,pt)*
            (porePressure(cell,pt)-porePressureold(cell, pt))/
            biotModulus(cell,pt)*refArea(cell,pt)*thickness;



          // If there is no diffusion, then the residual defines only on the mid-plane value

          // Diffusion term (parallel direction)
    	  for (int i(0); i < numDims; ++i ){
    	     for (int j(0); j< numPlaneDims; ++j ){


       	       poroMassResidual(cell, node) -=  refArea(cell, pt)
      	    		                                                *refGrads(node, pt, j)
      	    		                                                *invRefDualBasis(i,j)
      	    		                                                *flux(cell, pt, i)*dt;


        	  }
          }

      	 poroMassResidual(cell, topNode) =   poroMassResidual(cell, node);


      	//  Diffusion term (orthogonal direction) // WORK IN PROGRESS
        for (int k(0); k < numDims; ++k ){
       // 	for (int l(0); l < numDims; ++l ){

           	   poroMassResidual(cell, node) -= refValues(node,pt)
           			                                                          *refArea(cell, pt)
           			                                          //                /thickness
           			                                          //          *invRefDualBasis(l,k)
                                                              //              *invG_2(k)*invG_2(l)
           			                                                      *N(k)*flux(cell, pt, k)*dt;

               poroMassResidual(cell, topNode)       += refValues(node,pt)
                   			                                                          *refArea(cell, pt)
                   			                                          //                /thickness
                   			                                          //          *invRefDualBasis(l,k)
                                                                      //              *invG_2(k)*invG_2(l)
                   			                                                      *N(k)*flux(cell, pt, k)*dt;


     /*
      	   poroMassResidual(cell, node) -= refValues(node,pt)
      			                                                          *refArea(cell, pt)
      			                                                          /thickness
      			                                                          *Kref(cell, pt, k, l)
      			                                          //          *invRefDualBasis(l,k)
                                                         //              *invG_2(k)*invG_2(l)
      			                                                      *N(k)*N(l)
      			                                                      *scalarJump(cell,pt)*dt;

       	  poroMassResidual(cell, node)       -= refValues(node,pt)
       			                                                         *refArea(cell, pt)
      	      			                                                 /thickness
      	      			                                                 *Kref(cell, pt, k, l)
      	      			                                  //       *invRefDualBasis(l,k)
                                                          //       	*invG_2(k)*invG_2(l)
      	      			                                           *N(k)*N(l)
      	      			                                           *scalarJump(cell,pt)*dt;

      */

      //        	}
        }





        } // end integrartion point loop
      } //  end plane node loop

      // Stabilization term (if needed)
    } // end cell loop



  }
  //**********************************************************************  
}
