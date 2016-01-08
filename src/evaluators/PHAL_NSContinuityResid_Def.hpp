//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
NSContinuityResid<EvalT, Traits>::
NSContinuityResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ), 
  VGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  rho         (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),

  CResidual   (p.get<std::string>                   ("Residual Name"), 
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  havePSPG(p.get<bool>("Have PSPG"))
{
  this->addDependentField(wBF);  
  this->addDependentField(VGrad);
  this->addDependentField(rho);
  if (havePSPG) {
    wGradBF = PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>(
      p.get<std::string>("Weighted Gradient BF Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") );
    TauM = PHX::MDField<ScalarT,Cell,QuadPoint>(
      p.get<std::string>("Tau M Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    Rm = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(
      p.get<std::string>("Rm Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(wGradBF);
    this->addDependentField(TauM);
    this->addDependentField(Rm);
  }
   
  this->addEvaluatedField(CResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  // Allocate workspace
  divergence.resize(dims[0], numQPs);

  this->setName("NSContinuityResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSContinuityResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(rho,fm);
  if (havePSPG) {
    this->utils.setFieldData(wGradBF,fm); 
    this->utils.setFieldData(TauM,fm);
    this->utils.setFieldData(Rm,fm);
  }

  this->utils.setFieldData(CResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSContinuityResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      divergence(cell,qp) = 0.0;
      for (std::size_t i=0; i < numDims; ++i) {
        divergence(cell,qp) += rho(cell,qp)*VGrad(cell,qp,i,i);
      }
    }
  }
  
  FST::integrate<ScalarT>(CResidual, divergence, wBF, Intrepid2::COMP_CPP,  
                          false); // "false" overwrites

  if (havePSPG) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
	for (std::size_t qp=0; qp < numQPs; ++qp) {               
	  for (std::size_t j=0; j < numDims; ++j) { 
	    CResidual(cell,node) += 
	      rho(cell,qp)*TauM(cell,qp)*Rm(cell,qp,j)*wGradBF(cell,node,qp,j);
	  }  
	}    
      }
    }
  }

}

//**********************************************************************
}

