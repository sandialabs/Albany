//*****************************************************************//
//    albany 2.0:  Copyright 2012 Sandia Corporation               //
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

        LCM::Vector<ScalarT> G_0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G_1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G_2(3, &refDualBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> N(3, &refNormal(cell, pt, 0));

        LCM::Vector<ScalarT> scalarGradPerpendicular(0, 0, 0);
        LCM::Vector<ScalarT> scalarGradParallel(0, 0, 0);

       // Need to inverse basis [G_0 ; G_1 G_2] and none of them should be normalized

        // in-plane (parallel) contribution
        for (int node(0); node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
          midPlaneAvg = 0.5 * (nodalScalar(cell, node) + nodalScalar(cell, topNode));
          scalarGradParallel += refGrads(node, pt, 0) * midPlaneAvg * G_0;
          scalarGradParallel += refGrads(node, pt, 1) * midPlaneAvg * G_1;
        }

        // normal (perpendicular) contribution
        scalarGradPerpendicular = jump(cell,pt)*N/thickness;

        // assign components to MDfield ScalarGrad
        for (int i(0); i < numDims; ++i )
          scalarGrad(cell, pt, i) = scalarGradParallel(i) + scalarGradPerpendicular(i);
      }
    }
  }
  //**********************************************************************  
}
