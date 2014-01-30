//*****************************************************************//
//    albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  SurfaceScalarGradient<EvalT, Traits>::
  SurfaceScalarGradient(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")), 
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
    intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    jump           (p.get<std::string>("Scalar Jump Name"),dl->qp_scalar),
    nodalScalar    (p.get<std::string>("Nodal Scalar Name"),dl->node_scalar),
    scalarGrad     (p.get<std::string>("Surface Scalar Gradient Name"),dl->qp_vector)
  {
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);
    this->addDependentField(jump);
    this->addDependentField(nodalScalar);

    this->addEvaluatedField(scalarGrad);

    this->setName("Surface Scalar Gradient"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;

#ifdef ALBANY_VERBOSE
    std::cout << "in Surface Gradient Jump" << std::endl;
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

    // Pre-Calculate reference element quantitites
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
    ScalarT midPlaneAvg;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < numQPs; ++pt) {

        Intrepid::Vector<MeshScalarT> G_0(3, &refDualBasis(cell, pt, 0, 0));
        Intrepid::Vector<MeshScalarT> G_1(3, &refDualBasis(cell, pt, 1, 0));
        Intrepid::Vector<MeshScalarT> G_2(3, &refDualBasis(cell, pt, 2, 0));
        Intrepid::Vector<MeshScalarT> N(3, &refNormal(cell, pt, 0));

        Intrepid::Vector<ScalarT> scalarGradPerpendicular(0, 0, 0);
        Intrepid::Vector<ScalarT> scalarGradParallel(0, 0, 0);

       // Need to inverse basis [G_0 ; G_1; G_2] and none of them should be normalized
        Intrepid::Tensor<MeshScalarT> gBasis(3, &refDualBasis(cell, pt, 0, 0));
        Intrepid::Tensor<MeshScalarT> invRefDualBasis(3);

        // This map the position vector from parent to current configuration in R^3
        gBasis = Intrepid::transpose(gBasis);
       invRefDualBasis = Intrepid::inverse(gBasis);

        Intrepid::Vector<MeshScalarT> invG_0(3, &invRefDualBasis(0, 0));
        Intrepid::Vector<MeshScalarT> invG_1(3, &invRefDualBasis(1, 0));
        Intrepid::Vector<MeshScalarT> invG_2(3, &invRefDualBasis(2, 0));

        // in-plane (parallel) contribution
        for (int node(0); node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
          midPlaneAvg = 0.5 * (nodalScalar(cell, node) + nodalScalar(cell, topNode));
          for (int i(0); i < numDims; ++i) {
            scalarGradParallel(i) += 
              refGrads(node, pt, 0) * midPlaneAvg * invG_0(i) +
              refGrads(node, pt, 1) * midPlaneAvg * invG_1(i);
          }
        }

        // normal (perpendicular) contribution
        for (int i(0); i < numDims; ++i) {
          scalarGradPerpendicular(i) = jump(cell,pt) / thickness *invG_2(i);
        }

        // assign components to MDfield ScalarGrad
        for (int i(0); i < numDims; ++i )
          scalarGrad(cell, pt, i) = scalarGradParallel(i) + scalarGradPerpendicular(i);
      }
    }
  }
  //**********************************************************************  
}
