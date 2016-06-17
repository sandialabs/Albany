//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace HYD {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrideWResid<EvalT, Traits>::
HydrideWResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  BF          (p.get<std::string>                   ("BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  cDot        (p.get<std::string>                   ("C QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  cDotNode    (p.get<std::string>                   ("C QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  wGrad     (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  wResidual (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{

  lump = p.get<bool>("Lump Mass");

  this->addDependentField(wBF);
  this->addDependentField(BF);
  this->addDependentField(cDot);
  this->addDependentField(cDotNode);
  this->addDependentField(wGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(wResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("HydrideWResid"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrideWResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(cDot,fm);
  this->utils.setFieldData(cDotNode,fm);

  this->utils.setFieldData(wResidual,fm);
}

template<typename EvalT, typename Traits>
void HydrideWResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  FST::integrate(wResidual, wGrad, wGradBF, false); // "false" overwrites

  if(!lump){
    // Consistent mass matrix, the Intrepid2 way
    FST::integrate(wResidual, cDot, wBF, true); // "true" sums into

    // Consistent mass matrix, done manually
/*
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
      for (std::size_t node=0; node < numNodes; ++node)
       for (std::size_t qp=0; qp < numQPs; ++qp) 

         wResidual(cell, node) += cDot(cell, qp) * wBF(cell, node, qp);
*/
  }
  else {

   ScalarT diag;

    // Lumped mass matrix
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp) {

       diag = 0;
       for (std::size_t node=0; node < numNodes; ++node)

          diag += BF(cell, node, qp); // lump all the row onto the diagonal

       for (std::size_t node=0; node < numNodes; ++node)

          wResidual(cell, node) += diag * cDotNode(cell, node) * wBF(cell, node, qp);

     }

  }

}

//**********************************************************************
}

