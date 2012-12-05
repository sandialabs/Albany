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
SurfaceScalarJump<EvalT, Traits>::
SurfaceScalarJump(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  cubature      (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")), 
  scalar        (p.get<std::string>("Scalar Name"),dl->node_scalar),
  scalarJump    (p.get<std::string>("Scalar Jump Name"),dl->qp_scalar),
  scalarAverage (p.get<std::string>("Scalar Average Name"),dl->qp_scalar)
{
  this->addDependentField(scalar);

  this->addEvaluatedField(scalarJump);
  this->addEvaluatedField(scalarAverage);

  this->setName("Surface Scalar Jump"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numDims = dims[2];

  numQPs = cubature->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims = numDims - 1;

#ifdef ALBANY_VERBOSE
    std::cout << "in Surface Scalar Jump" << std::endl;
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
  void SurfaceScalarJump<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalar,fm);
    this->utils.setFieldData(scalarJump,fm);
    this->utils.setFieldData(scalarAverage,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceScalarJump<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    ScalarT scalarA(0.0), scalarB(0.0);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < numQPs; ++pt) {
        scalarA = 0.0;
        scalarB = 0.0;
        for (std::size_t node=0; node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
          scalarA += refValues(node, pt) * scalar(cell, node);
          scalarB += refValues(node, pt) * scalar(cell, topNode);
        }
        scalarJump(cell,pt) = scalarB - scalarA;
        scalarAverage(cell,pt) = 0.5*(scalarB + scalarA);
      }
    }
  }

  //**********************************************************************
}

