//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Tensor.h"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  SurfaceScalarGradient<EvalT, Traits>::
  SurfaceScalarGradient(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")), 
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
    intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
  //  cellType       (p.get<Teuchos::RCP<shards::CellTopology> >("Cell Type")),
    currentBasis   (p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    jump           (p.get<std::string>("Scalar Jump Name"),dl->qp_scalar),
    nodalScalar    (p.get<std::string>("Nodal Scalar Name"),dl->node_scalar),
    scalarGrad     (p.get<std::string>("Surface Scalar Gradient Name"),dl->qp_vector),
    weights        (p.get<std::string>("Weights Name"),dl->qp_scalar),
    weightedAverage(false),
    alpha(0.0)
  {
    if ( p.isType<string>("Weighted Volume Average J Name") )
      weightedAverage = p.get<bool>("Weighted Volume Average J");
    if ( p.isType<double>("Average J Stabilization Parameter Name") )
      alpha = p.get<double>("Average J Stabilization Parameter");

    this->addDependentField(currentBasis);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);
    this->addDependentField(jump);
    this->addDependentField(nodalScalar);

    this->addEvaluatedField(scalarGrad);
  //  this->addEvaluatedField(J);

    this->setName("Surface Scalar Gradient"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // temp space for midplane coords
    midplaneScalar.resize(worksetSize, numPlaneNodes, numDims);


    // Pre-Calculate reference element quantitites
    std::cout << "SurfaceScalarGradient Calling Intrepid to get reference quantities" << std::endl;
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceScalarGradient<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(jump,fm);
    this->utils.setFieldData(nodalScalar,fm);
    this->utils.setFieldData(scalarGrad,fm);

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceScalarGradient<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {

	 // compute mid-plane scalar value for the normal contribution calculation
     for (int cell(0); cell < midplaneScalar.dimension(0); ++cell) {
        // compute the mid-plane value
        for (int node(0); node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
            midplaneScalar(cell, node) = 0.5 * (nodalScalar(cell, node) + nodalScalar(cell, topNode));
        }
     }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < numQPs; ++pt) {


        LCM::Vector<ScalarT> g_0(3, &currentBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> g_1(3, &currentBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> g_2(3, &currentBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> G_2(3, &refNormal(cell, pt, 0));
        LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));


        LCM::Vector<ScalarT> scalarGradOrthogonal(0, 0, 0);
        LCM::Vector<ScalarT> scalarGradNormal(0, 0, 0);


        for (std::size_t i=0; i < numDims; ++i)
		 {
        	scalarGrad(cell,pt,i) = 0;

        	// normal contribution
        	for (int node(0); node < numPlaneNodes; ++node) {
			//  scalarGradNormal(i) += refGrads(node, pt, i)*midplaneScalar(cell,node)*(g_0(i)+g_1(i));
        	}

			// orthogonal contribution
              scalarGradOrthogonal(i) = jump(cell,pt)*g_2(i)/thickness;
              scalarGrad(cell, pt, i) =scalarGradOrthogonal(i) + scalarGradNormal(i);
        }

      }
    }

}
  //**********************************************************************  
}
