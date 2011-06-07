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

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
DislocationDensity<EvalT, Traits>::
DislocationDensity(const Teuchos::ParameterList& p) :
  Fp            (p.get<std::string>                   ("Fp Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  BF            (p.get<std::string>                   ("BF Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  GradBF        (p.get<std::string>                   ("Gradient BF Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  G             (p.get<std::string>                   ("Dislocation Density Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  this->addDependentField(Fp);
  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addEvaluatedField(G);

  // Get Dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dim;
  vector_dl->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];

  // Allocate Temporary FieldContainers
  BF_operator.resize(numQPs,numNodes);
  BF_inverse.resize(numNodes,numQPs);
  nodalFp.resize(numNodes, numDims, numDims);
  curlFp.resize(numQPs, numDims, numDims);
  A.resize(numNodes,numNodes);
  Ainv.resize(numNodes,numNodes);
    
  square = (numNodes == numQPs);

  this->setName("DislocationDensity"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DislocationDensity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Fp,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(G,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DislocationDensity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

//   /** The allocated size of the Field Containers must currently 
//     * match the full workset size of the allocated PHX Fields, 
//     * this is the size that is used in the computation. There is
//     * wasted effort computing on zeroes for the padding on the
//     * final workset. Ideally, these are size numCells.
//   //int containerSize = workset.numCells;
//     */
  
//   // setJacobian only needs to be RealType since the data type is only
//   //  used internally for Basis Fns on reference elements, which are
//   //  not functions of coordinates. This save 18min of compile time!!!
//   Intrepid::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);
//   Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);
//   Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobian_det, jacobian);

//   Intrepid::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
//     (weighted_measure, jacobian_det, refWeights);
//   Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
//     (BF, val_at_cub_points);
//   Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
//     (wBF, weighted_measure, BF);
//   Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
//     (GradBF, jacobian_inv, grad_at_cub_points);
//   Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
//     (wGradBF, weighted_measure, GradBF);

  for (int i=0; i < G.size() ; i++) G[i] = 0.0;

// construct the node --> point operator
  Intrepid::FieldContainer<double> BF_operator;
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    BF_operator.initialize(0.0);
    for (std::size_t node=0; node < numNodes; ++node) 
      for (std::size_t qp=0; qp < numQPs; ++qp) 
	BF_operator(qp,node) += BF(cell,node,qp);
    
    if (square) 
    {
      // compute mapping from point --> node
      Intrepid::RealSpaceTools<double>::inverse(BF_inverse,BF_operator);
    } 
    else 
    {
      // conmpute pseudo-inverse
      for (std::size_t i=0; i < numNodes; ++i) 
	for (std::size_t j=0; j < numNodes; ++j) 
	  for (std::size_t qp=0; qp < numQPs; ++qp) 
	    A(i,j) = BF_operator(qp,i) * BF_operator(qp,j);

      Intrepid::RealSpaceTools<double>::inverse(Ainv,A);

      for (std::size_t i=0; i < numNodes; ++i) 
	for (std::size_t j=0; j < numNodes; ++j) 
	  for (std::size_t qp=0; qp < numQPs; ++qp) 
	    BF_inverse(i,qp) = Ainv(i,j) * BF_operator(qp,j);
    }
    
    nodalFp.initialize(0.0);
    for (std::size_t node=0; node < numNodes; ++node) 
      for (std::size_t qp=0; qp < numQPs; ++qp) 
	for (std::size_t i=0; i < numDims; ++i) 
	  for (std::size_t j=0; j < numDims; ++j) 
	    nodalFp(node,i,j) += BF_inverse(node,qp) * Fp(cell,qp,i,j);

    // compute the curl using nodalFp
    curlFp.initialize(0.0);
    for (std::size_t node=0; node < numNodes; ++node) 
    {
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
	curlFp(qp,0,0) = nodalFp(node,0,2) * GradBF(cell,node,qp,1) - nodalFp(node,0,1) * GradBF(cell,node,qp,2);
	curlFp(qp,0,1) = nodalFp(node,1,2) * GradBF(cell,node,qp,1) - nodalFp(node,1,1) * GradBF(cell,node,qp,2);
	curlFp(qp,0,2) = nodalFp(node,2,2) * GradBF(cell,node,qp,1) - nodalFp(node,2,1) * GradBF(cell,node,qp,2);

	curlFp(qp,1,0) = nodalFp(node,0,0) * GradBF(cell,node,qp,2) - nodalFp(node,0,2) * GradBF(cell,node,qp,0);
	curlFp(qp,1,1) = nodalFp(node,1,0) * GradBF(cell,node,qp,2) - nodalFp(node,1,2) * GradBF(cell,node,qp,0);
	curlFp(qp,1,2) = nodalFp(node,2,0) * GradBF(cell,node,qp,2) - nodalFp(node,2,2) * GradBF(cell,node,qp,0);

	curlFp(qp,2,0) = nodalFp(node,0,1) * GradBF(cell,node,qp,0) - nodalFp(node,0,0) * GradBF(cell,node,qp,1);
	curlFp(qp,2,1) = nodalFp(node,1,1) * GradBF(cell,node,qp,0) - nodalFp(node,1,0) * GradBF(cell,node,qp,1);
	curlFp(qp,2,2) = nodalFp(node,2,1) * GradBF(cell,node,qp,0) - nodalFp(node,2,0) * GradBF(cell,node,qp,1);
      }
    }

    for (std::size_t qp=0; qp < numQPs; ++qp) 
      for (std::size_t i=0; i < numDims; ++i) 
	for (std::size_t j=0; j < numDims; ++j) 
	  for (std::size_t k=0; k < numDims; ++k) 
	    G(cell,qp,i,j) += Fp(cell,qp,i,k) * curlFp(qp,k,j);
  }
}

//**********************************************************************
}
