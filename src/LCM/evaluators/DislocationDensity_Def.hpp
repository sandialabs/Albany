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
  nodalFp.resize(numNodes,numDims,numDims);
  curlFp.resize(numQPs,numDims,numDims);
    
  square = (numNodes == numQPs);

  TEST_FOR_EXCEPTION( square == true, std::runtime_error, 
		      "Dislocation Density Calculation currently needs numNodes == numQPs" );


  this->setName("DislocationDensity"+PHAL::TypeString<EvalT>::value);
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

  Teuchos::SerialDenseMatrix<int, double> A;
  Teuchos::SerialDenseMatrix<int, double> X;
  Teuchos::SerialDenseMatrix<int, double> B;
  Teuchos::SerialDenseSolver<int, double> solver;

  A.shape(numNodes,numNodes);
  X.shape(numNodes,numNodes);
  B.shape(numNodes,numNodes);
  
  // construct Identity for RHS
  for (int i = 0; i < numNodes; ++i)
    B(i,i) = 1.0;

  for (int i=0; i < G.size() ; i++) G[i] = 0.0;

  // construct the node --> point operator
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    for (std::size_t node=0; node < numNodes; ++node) 
      for (std::size_t qp=0; qp < numQPs; ++qp) 
	A(qp,node) = BF(cell,node,qp);
    
    X = 0.0;

    solver.setMatrix( Teuchos::rcp( &A, false) );
    solver.setVectors( Teuchos::rcp( &X, false ), Teuchos::rcp( &B, false ) );

    // Solve the system A X = B to find A_inverse
    int status = 0;
    status = solver.factor();
    status = solver.solve();

    // compute nodal Fp
    nodalFp.initialize(0.0);
    for (std::size_t node=0; node < numNodes; ++node) 
      for (std::size_t qp=0; qp < numQPs; ++qp) 
	for (std::size_t i=0; i < numDims; ++i) 
	  for (std::size_t j=0; j < numDims; ++j) 
	    nodalFp(node,i,j) += X(node,qp) * Fp(cell,qp,i,j);

    // compute the curl using nodalFp
    curlFp.initialize(0.0);
    for (std::size_t node=0; node < numNodes; ++node) 
    {
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
	curlFp(qp,0,0) += nodalFp(node,0,2) * GradBF(cell,node,qp,1) - nodalFp(node,0,1) * GradBF(cell,node,qp,2);
	curlFp(qp,0,1) += nodalFp(node,1,2) * GradBF(cell,node,qp,1) - nodalFp(node,1,1) * GradBF(cell,node,qp,2);
	curlFp(qp,0,2) += nodalFp(node,2,2) * GradBF(cell,node,qp,1) - nodalFp(node,2,1) * GradBF(cell,node,qp,2);

	curlFp(qp,1,0) += nodalFp(node,0,0) * GradBF(cell,node,qp,2) - nodalFp(node,0,2) * GradBF(cell,node,qp,0);
	curlFp(qp,1,1) += nodalFp(node,1,0) * GradBF(cell,node,qp,2) - nodalFp(node,1,2) * GradBF(cell,node,qp,0);
	curlFp(qp,1,2) += nodalFp(node,2,0) * GradBF(cell,node,qp,2) - nodalFp(node,2,2) * GradBF(cell,node,qp,0);

	curlFp(qp,2,0) += nodalFp(node,0,1) * GradBF(cell,node,qp,0) - nodalFp(node,0,0) * GradBF(cell,node,qp,1);
	curlFp(qp,2,1) += nodalFp(node,1,1) * GradBF(cell,node,qp,0) - nodalFp(node,1,0) * GradBF(cell,node,qp,1);
	curlFp(qp,2,2) += nodalFp(node,2,1) * GradBF(cell,node,qp,0) - nodalFp(node,2,0) * GradBF(cell,node,qp,1);
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
