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
SurfaceScalarJump(const Teuchos::ParameterList& p) :
  cubature      (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")), 
  scalar        (p.get<std::string>("Scalar Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  scalarJump          (p.get<std::string>("Scalar Jump Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(scalar);

  this->addEvaluatedField(scalarJump);

  this->setName("Surface Scalar Jump"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> nv_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  nv_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numDims = dims[2];

  Teuchos::RCP<PHX::DataLayout> qpv_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  qpv_dl->dimensions(dims);
  numQPs = dims[1];

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
  void SurfaceScalarJump<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalar,fm);
    this->utils.setFieldData(scalarJump,fm);
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

      }
    }
  }

  //**********************************************************************
}

