//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
SurfaceCohesiveResidual<EvalT, Traits>::
SurfaceCohesiveResidual(const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    cubature_(
        p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepid_basis_(
        p.get<Teuchos::RCP<Intrepid::Basis<RealType,
            Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    ref_area_(
        p.get<std::string>("Reference Area Name"), dl->qp_scalar),
    cohesiveTraction(
        p.get<std::string>("Cohesive Traction Name"), dl->qp_vector),
    force(
        p.get<std::string>("Surface Cohesive Residual Name"), dl->node_vector)
{
  this->addDependentField(ref_area_);
  this->addDependentField(cohesiveTraction);

  this->addEvaluatedField(force);

  this->setName("Surface Cohesive Residual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numDims = dims[2];

  dl->qp_vector->dimensions(dims);

  numQPs = cubature_->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims = numDims - 1;

  // Allocate Temporary FieldContainers
  refValues.resize(numPlaneNodes, numQPs);
  refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
  refPoints.resize(numQPs, numPlaneDims);
  refWeights.resize(numQPs);

  // Pre-Calculate reference element quantitites
  cubature_->getCubature(refPoints, refWeights);
  intrepid_basis_->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
  intrepid_basis_->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SurfaceCohesiveResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(cohesiveTraction, fm);
  this->utils.setFieldData(ref_area_, fm);
  this->utils.setFieldData(force, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SurfaceCohesiveResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Intrepid::Vector<ScalarT> f_plus(0, 0, 0);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {

      int topNode = node + numPlaneNodes;

      // initialize force vector
      f_plus.clear();

      for (int pt(0); pt < numQPs; ++pt) {

        // refValues(numPlaneNodes, numQPs) = shape function
        // refArea(numCells, numQPs) = |Jacobian|*weight
        f_plus(0) += cohesiveTraction(cell, pt, 0) * refValues(node, pt)
            * ref_area_(cell, pt);
        f_plus(1) += cohesiveTraction(cell, pt, 1) * refValues(node, pt)
            * ref_area_(cell, pt);
        f_plus(2) += cohesiveTraction(cell, pt, 2) * refValues(node, pt)
            * ref_area_(cell, pt);

        // output for debug, I'll keep it for now until the implementation fully verified
        // Q.Chen
        //std::cout << "refValues " << Sacado::ScalarValue<ScalarT>::eval(refValues(node,pt)) << std::endl;
        //std::cout << "refArea " << Sacado::ScalarValue<ScalarT>::eval(refArea(cell,pt)) << std::endl;

      } // end of pt loop

      force(cell, node, 0) = -f_plus(0);
      force(cell, node, 1) = -f_plus(1);
      force(cell, node, 2) = -f_plus(2);

      force(cell, topNode, 0) = f_plus(0);
      force(cell, topNode, 1) = f_plus(1);
      force(cell, topNode, 2) = f_plus(2);

    } // end of planeNode loop
  } // end of cell loop

}
//**********************************************************************
}
