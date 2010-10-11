/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p) :
  coordVec      (p.get<std::string>                   ("Coordinate Vector Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis") ),
  BF     (p.get<std::string>                   ("BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  GradBF  (p.get<std::string>                   ("Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  wGradBF (p.get<std::string>                   ("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") )
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  this->setName("ComputeBasisFunctions"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions
  numVertices = dims[1];
  numDims = dims[2];

  typename std::vector< typename PHX::template MDField<RealType,Cell,Node,QuadPoint>::size_type > dim;
  BF.dimensions(dim); //get dimensions
  numNodes = dim[1];
  numQPs = dim[2];
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
    */
  
  //int containerSize = workset.numCells;
  int containerSize = workset.worksetSize;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> val_at_cub_points(numNodes, numQPs);
  Intrepid::FieldContainer<RealType> grad_at_cub_points(numNodes, numQPs, numDims);
  Intrepid::FieldContainer<RealType> refPoints(numQPs, numDims);
  Intrepid::FieldContainer<RealType> refWeights(numQPs);
  Intrepid::FieldContainer<MeshScalarT> jacobian(containerSize, numQPs, numDims, numDims);
  Intrepid::FieldContainer<MeshScalarT> jacobian_inv(containerSize, numQPs, numDims, numDims);
  Intrepid::FieldContainer<MeshScalarT> jacobian_det(containerSize, numQPs);
  Intrepid::FieldContainer<MeshScalarT> weighted_measure(containerSize, numQPs);

  // Start computations
  cubature->getCubature(refPoints, refWeights);

  intrepidBasis->getValues(val_at_cub_points, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

//cout << "EvalT value = " <<  PHX::TypeString<EvalT>::value << endl;

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
