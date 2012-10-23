//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  weighted_measure (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];


  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  // Allocate Temporary FieldContainers
  val_at_cub_points.resize(numNodes, numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);
  jacobian.resize(containerSize, numQPs, numDims, numDims);
  jacobian_inv.resize(containerSize, numQPs, numDims, numDims);
  jacobian_det.resize(containerSize, numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(val_at_cub_points, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

  this->setName("ComputeBasisFunctions"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
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

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */
  
  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!
  Intrepid::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);
  Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobian_det, jacobian);

  Intrepid::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
    (weighted_measure, jacobian_det, refWeights);
  Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
    (BF, val_at_cub_points);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wBF, weighted_measure, BF);
  Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
    (GradBF, jacobian_inv, grad_at_cub_points);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wGradBF, weighted_measure, GradBF);
}

//**********************************************************************
}
