//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl,
                      const Albany::MeshSpecsStruct* meshSpecs) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature      (p.get<Teuchos::RCP <Cogent::Integrator> >("Cubature")),
  weighted_measure (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  jacobian_det (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{

  elementBlockName = meshSpecs->ebName;

  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numCells = dim[0];
  numNodes = dim[1];
  numQPs   = dim[2];
  numDims  = dim[3];

  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  topoNames = cubature->getFieldNames();
  numTopos = topoNames.size();
  topoVals.resize(numNodes,numTopos);
  coordVals.resize(numNodes,numDims);

  // Allocate Temporary FieldContainers
  val_at_cub_points.resize(numNodes, numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  weights.resize(numQPs);
  jacobian.resize(numCells, numQPs, numDims, numDims);
  jacobian_inv.resize(numCells, numQPs, numDims, numDims);

  cubature->getStandardPoints(refPoints);

  cubature->getBasis()->getValues(val_at_cub_points, refPoints, Intrepid2::OPERATOR_VALUE);
  cubature->getBasis()->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  this->setName("ComputeBasisFunctions"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if( elementBlockName != workset.EBName ) return;

  /** The allocated size of the Field Containers must currently
    * match the full workset size of the allocated PHX Fields,
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int numCells = workset.numCells;
    */

  typedef typename Intrepid2::CellTools<MeshScalarT>   ICT;
  typedef Intrepid2::FunctionSpaceTools                IFST;

  Intrepid2::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, cubature->getBasis());
  ICT::setJacobianInv (jacobian_inv, jacobian);
  ICT::setJacobianDet (jacobian_det, jacobian);

  Teuchos::Array<Albany::MDArray> topo(numTopos);
  for(int itopo=0; itopo<numTopos; itopo++){
    topo[itopo] = (*workset.stateArrayPtr)[topoNames[itopo]];
  }
 
  for(int cell=0; cell<workset.numCells; cell++){

    for(int node=0; node<numNodes; node++)
      for(int dim=0; dim<numDims; dim++)
        coordVals(node,dim) = coordVec(cell,node,dim);

    for(int itopo=0; itopo<numTopos; itopo++)
      for(int node=0; node<numNodes; node++)
        topoVals(node,itopo) = topo[itopo](cell,node);

    // fix this.  refPoints is being passed back every call even though it doesn't change.
    // do you need to send coordVals?
    cubature->getCubature(weights, refPoints, topoVals, coordVals);

    for(int qp=0; qp<numQPs; qp++)
      weighted_measure(cell, qp) = weights(qp);
// weights are already in physical coordinates
//      weighted_measure(cell, qp) = jacobian_det(cell,qp)*weights(qp);
  }

//  IFST::computeCellMeasure<MeshScalarT> (weighted_measure, jacobian_det, weights);
  IFST::HGRADtransformVALUE<RealType>   (BF, val_at_cub_points);
  IFST::multiplyMeasure<MeshScalarT>    (wBF, weighted_measure, BF);
  IFST::HGRADtransformGRAD<MeshScalarT> (GradBF, jacobian_inv, grad_at_cub_points);
  IFST::multiplyMeasure<MeshScalarT>    (wGradBF, weighted_measure, GradBF);
}

//**********************************************************************
}
