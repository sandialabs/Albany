/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
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
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
CahnHillWResid<EvalT, Traits>::
CahnHillWResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  BF          (p.get<std::string>                   ("BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  rhoDot      (p.get<std::string>                   ("Rho QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rhoDotNode  (p.get<std::string>                   ("Rho QP Time Derivative Variable Name"),
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
  this->addDependentField(rhoDot);
  this->addDependentField(rhoDotNode);
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

  this->setName("CahnHillWResid"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillWResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(rhoDot,fm);
  this->utils.setFieldData(rhoDotNode,fm);

  this->utils.setFieldData(wResidual,fm);
}

template<typename EvalT, typename Traits>
void CahnHillWResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  FST::integrate<ScalarT>(wResidual, wGrad, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites

  if(!lump){
    // Consistent mass matrix, the Intrepid way
    FST::integrate<ScalarT>(wResidual, rhoDot, wBF, Intrepid::COMP_CPP, true); // "true" sums into

    // Consistent mass matrix, done manually
/*
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
      for (std::size_t node=0; node < numNodes; ++node)
       for (std::size_t qp=0; qp < numQPs; ++qp) 

         wResidual(cell, node) += rhoDot(cell, qp) * wBF(cell, node, qp);
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

          wResidual(cell, node) += diag * rhoDotNode(cell, node) * wBF(cell, node, qp);

     }

  }

}

//**********************************************************************
}

